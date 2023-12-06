import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from timm import create_model
from timm.layers.pos_embed import resample_abs_pos_embed
from flexivit_pytorch import pi_resize_patch_embed
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path, net):
    model = torch.load(model_path, map_location=device)
    net.load_state_dict(model, strict=True)
    net.to(device)
    net.eval()
    return net



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
    return net

def sample_images(dataset, samples_per_class):
    class_counts = {i: 0 for i in range(10)}  # CIFAR-10 has 10 classes
    sampled_indices = []

    for idx, (_, class_idx) in enumerate(dataset):
        if class_counts[class_idx] < samples_per_class:
            sampled_indices.append(idx)
            class_counts[class_idx] += 1
            if all(count == samples_per_class for count in class_counts.values()):
                break

    return sampled_indices

def unnormalize(tensor):
    for t, m, s in zip(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):  # mean and std
        t.mul_(s).add_(m)  # Unnormalize the channel
    return tensor

def get_attention_maps(model, dataloader, layer_num):
    attention_maps = []
    imgs = []

    # Define a hook
    def hook_fn(module, input, output):
        # Adjust this line based on the actual output structure
        print(output.size())
        attention_map = output[0] if isinstance(output, tuple) else output
        attention_maps.append(attention_map.detach().cpu())

    # Attach the hook to the desired attention layer
    handle = model.blocks[layer_num].attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            model(inputs)
            imgs.append(inputs.cpu())

    # Remove the hook
    handle.remove()
    return attention_maps, imgs

def visualize_attention_maps(attention_maps, imgs, patch_size, num_images=3):
    # Assuming square images and square patch size
    num_patches_side = (224 // patch_size) ** 2
    fig, axs = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i in range(num_images):
        img_unnorm = unnormalize(imgs[0][i].clone())  # Unnormalize the image
        img = transforms.ToPILImage()(img_unnorm).convert("RGB")
        attn_map = attention_maps[0][i].mean(dim=1)  # Average over heads
        print(attn_map.size())
        attn_map = attn_map[1:].reshape(int(np.sqrt(num_patches_side)), int(np.sqrt(num_patches_side)))
        attn_map = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=img.size, mode='bilinear', align_corners=False).squeeze()

        axs[i, 0].imshow(img)
        axs[i, 0].set_title('Original Image')

        axs[i, 1].imshow(img)
        axs[i, 1].imshow(attn_map, cmap='jet', alpha=0.5)
        axs[i, 1].set_title('Attention Map')

    plt.savefig("attention_map_32.png")
    # plt.show()


transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

sampled_indices = sample_images(testset, 50)

random_loader = DataLoader(Subset(testset, np.random.choice(len(testset), 3)), batch_size=3, shuffle=False)


model_path = "saved_models/32x32_fine_tuned.pth"
net = create_vit_with_patch_size((32, 32))
model = load_model(model_path, net=net)

layer_num = 11
attention_maps, imgs = get_attention_maps(model, random_loader, layer_num)
visualize_attention_maps(attention_maps, imgs, patch_size=32)


