import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from timm import create_model
from timm.layers.pos_embed import resample_abs_pos_embed
from flexivit_pytorch import pi_resize_patch_embed

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
    net = create_model("vit_base_patch16_224", img_size=image_size, patch_size=new_patch_size)
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

def get_attention_weights(model, dataloader, layer_num):
    attention_maps = []

    def hook_fn(module, input, output):
        # We take the attention weights and aggregate over the batch
        attention_maps.append(output.detach().cpu())

    handle = model.blocks[layer_num].attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, label = data
            inputs = inputs.to(device)
            model(inputs)  # This will trigger the hook

    handle.remove()

    # Aggregate attention maps across the batch
    aggregated_attention = torch.mean(torch.stack(attention_maps), dim=0)
    return aggregated_attention

def compute_distance_matrix(patch_size, num_patches):
    """Helper function to compute distance matrix."""
    length = int(np.sqrt(num_patches))  
    distance_matrix = np.zeros((num_patches, num_patches))

    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue
            xi, yi = divmod(i, length)
            xj, yj = divmod(j, length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix

def compute_mean_attention_dist(patch_size, attention_weights):
    # Assuming attention_weights shape = (batch_size, num_heads, num_patches + 1, num_patches + 1)
    # Remove the CLS token and average over the batch
    attention_weights = attention_weights[:, :, 1:, 1:]
    num_patches = attention_weights.shape[-1]
    distance_matrix = compute_distance_matrix(patch_size, num_patches)

    # Compute mean distances
    mean_distances = np.sum(
        attention_weights.numpy() * distance_matrix[None, None, :, :], axis=-1
    ) / num_patches

    return np.mean(mean_distances, axis=1)  # Average across patches

# Load your data and model
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
sampled_indices = sample_images(testset, 50)
sampled_dataset = Subset(testset, sampled_indices)
sampled_loader = DataLoader(sampled_dataset, batch_size=4, shuffle=False)

model_path = "saved_models/8x8_fine_tuned.pth"
net = create_vit_with_patch_size((8, 8))
model = load_model(model_path, net=net)

# Choose a layer number to extract attention from
layer_num = 11  # Example: choose a layer

# Get attention weights
attention_weights = get_attention_weights(model, sampled_loader, layer_num)
mean_distances = compute_mean_attention_dist(8, attention_weights.numpy())  # Assuming patch size is 8

print("Mean attention distances:", mean_distances)
