from timm import create_model
from timm.layers.pos_embed import resample_abs_pos_embed
from flexivit_pytorch import pi_resize_patch_embed
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

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

net_16x16 = create_vit_with_patch_size((16, 16))
net_8x8 = create_vit_with_patch_size((8, 8))

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

train_sampler = DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, num_workers=2, sampler=train_sampler)

test_sampler = DistributedSampler(testset, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2, sampler=test_sampler)


def distillation_loss(output, target, teacher_output, temperature, alpha):
    soft_target = nn.functional.softmax(teacher_output / temperature, dim=1)
    soft_output = nn.functional.log_softmax(output / temperature, dim=1)
    distillation = nn.functional.kl_div(soft_output, soft_target, reduction='batchmean')
    
    criterion = nn.CrossEntropyLoss()
    standard_loss = criterion(output, target)

    return alpha * distillation + (1 - alpha) * standard_loss

def train_student_with_distillation(rank, world_size, student_model, teacher_model, trainloader, testloader, epochs=10, temperature=3.0, alpha=0.5):
    student_model.train()
    teacher_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model.to(device)
    teacher_model.to(device)

    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    epoch_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"\n Training for epoch : {epoch}\n")
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            student_outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)

            loss = distillation_loss(student_outputs, labels, teacher_outputs, temperature, alpha)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print (f"Batch {i} Accuracy : {100 * correct / total}")
            del loss, student_outputs, teacher_outputs

        epoch_accuracy = 100 * correct / total
        epoch_accuracies.append(epoch_accuracy)
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Accuracy: {epoch_accuracy}%')

    torch.save(student_model.state_dict(), f"saved_models/kd_student_tuned.pth")
    torch.save(teacher_model.state_dict(), f"saved_models/kd_teacher_tuned.pth")
    print('Finished Training Student')
    return epoch_accuracies

accuracies_distillation = train_student_with_distillation(net_16x16, net_8x8, trainloader, testloader, epochs=30)
epochs = range(1, 51)  
plt.plot(epochs, accuracies_distillation, label='32x32 Distilled from 2x2')

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs for 32x32 Model Distilled from 2x2 Model')
plt.legend()

# Save the plot if needed
plt.savefig('distilled_model_accuracy.png')

# Show the plot
plt.show()
