from timm import create_model
from timm.layers.pos_embed import resample_abs_pos_embed
from flexivit_pytorch import pi_resize_patch_embed
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import gc
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" 

def create_vit_with_patch_size(new_patch_size):
    state_dict = create_model("vit_base_patch16_224", pretrained=True).state_dict()
    state_dict["patch_embed.proj.weight"] = pi_resize_patch_embed(
        patch_embed=state_dict["patch_embed.proj.weight"], new_patch_size=new_patch_size
    )
    image_size = 224
    grid_size = image_size // new_patch_size[0]
    state_dict["pos_embed"] = resample_abs_pos_embed(
        posemb=state_dict["pos_embed"], new_size=[grid_size, grid_size]
    )
    net = create_model(
        "vit_base_patch16_224", img_size=image_size, patch_size=new_patch_size
    )
    net.load_state_dict(state_dict, strict=True)

    return net

def train_flexified_vit_and_track_accuracy(rank, model, trainloader, testloader, epochs=10, label=None):
        torch.cuda.set_device(rank)
        model.cuda()
        model = DDP(model, device_ids=[rank], output_device=rank)
        if rank == 0:
            print(f"Training started for Label : {label}")
        criterion = nn.CrossEntropyLoss()
        criterion.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epoch_accuracies = []

        for epoch in range(epochs):
            model.train()
            if rank == 0:
                print(f"\n Training for epoch : {epoch}\n")
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            if rank == 0:
                print(f"Loss of Epoch {epoch} : {loss}")

            # Evaluation
            if epoch % 5 == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        images, labels = images.cuda(), labels.cuda()
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                

                epoch_accuracy = 100 * correct / total
                epoch_accuracies.append(epoch_accuracy)
                if rank == 0:
                    print(f'Epoch for {label} - {epoch + 1}, Accuracy: {epoch_accuracy}%')
        
        torch.save(model.state_dict(), f"saved_models/{label}_scratch_trained.pth")
        if rank == 0:
            print('Finished Training for patch size : ',label, "\n")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return epoch_accuracies

if __name__ == "__main__":
    dist.init_process_group("nccl", init_method="env://")
    local_rank = dist.get_rank()
    world_size = torch.cuda.device_count()

    if local_rank == 0:
        print("Devices :", torch.cuda.device_count())
    # net_32x32 = create_vit_with_patch_size((32, 32))
    # net_16x16 = create_vit_with_patch_size((16, 16))
    # net_8x8 = create_vit_with_patch_size((8, 8))
    # net_4x4 = create_vit_with_patch_size((4, 4))
    net_2x2 = create_vit_with_patch_size((2, 2))

    # Data augmentation and normalization for training
    # Just normalization for validation
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR10 training and test datasets
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)

    testset = datasets.CIFAR10(root='./data', train=False,
                            download=True, transform=transform)
    
    train_sampler =  DistributedSampler(trainset, num_replicas=world_size, rank=local_rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, sampler=train_sampler)

    test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=local_rank, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, sampler=test_sampler)

    

    # accuracies_32x32_full = train_flexified_vit_and_track_accuracy(net_32x32, trainloader, testloader, epochs=30, freeze_layers=False, label="32x32")
    # accuracies_16x16_full = train_flexified_vit_and_track_accuracy(net_16x16, trainloader, testloader, epochs=30, freeze_layers=False, label="16x16")
    # accuracies_8x8_full = train_flexified_vit_and_track_accuracy(net_8x8, trainloader, testloader, epochs=30, freeze_layers=False, label="8x8")
    # accuracies_4x4_full = train_flexified_vit_and_track_accuracy(net_4x4, trainloader, testloader, epochs=30, freeze_layers=False, label="4x4")

    accuracies_2x2_full = train_flexified_vit_and_track_accuracy(local_rank, net_2x2, trainloader, testloader, epochs=50, label="2x2")

    # Assuming accuracies for each model are already calculated
    epochs = range(1, 51)  # Assuming 10 epochs
    # plt.plot(epochs, accuracies_32x32_full, label='32x32')
    # plt.plot(epochs, accuracies_16x16_full, label='16x16')
    # plt.plot(epochs, accuracies_8x8_full, label='8x8')
    plt.plot(epochs, accuracies_2x2_full, label='2x2')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs for Different Patch Sizes')
    plt.legend()

    # Save the plot
    plt.savefig('vit_patch_size_accuracy_comparison_scratch.png')

    # Show the plot
    plt.show()