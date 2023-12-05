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

def get_features(model, dataloader, layer_num):
    features = []
    labels = []
    print("Fetching features..")

    # Define a hook
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())  # Move feature data to CPU

    # Attach the hook to the desired layer
    handle = model.blocks[layer_num].register_forward_hook(hook_fn)

    with torch.no_grad():
        for data in dataloader:
            inputs, label = data
            inputs = inputs.to(device)  # Move inputs to GPU
            label = label.to(device)  # Move labels to GPU
            model(inputs)  # This will trigger the hook
            labels.append(label.cpu())  # Move label data to CPU

    # Remove the hook
    handle.remove()
    print("Finished fetching features..")
    return torch.cat(features), torch.cat(labels)

def visualize_tsne(features, labels, perplexity=30, n_iter=1000):
    # Reshape features to 2D array: [n_samples, n_features]
    n_samples, n_patches, n_features = features.shape
    features_2d = features.reshape(n_samples, n_patches * n_features)
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(features_2d)
    plt.figure(figsize=(10, 10))
    for class_idx in range(10):  # CIFAR-10 has 10 classes
        indices = labels == class_idx
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=str(class_idx))
    plt.title('t-SNE Visualization of ViT Features')
    plt.savefig('tsne_vit_features_32.png', dpi=300, bbox_inches='tight')
    # plt.show()


transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

sampled_indices = sample_images(testset, 50)

sampled_dataset = Subset(testset, sampled_indices)
sampled_loader = DataLoader(sampled_dataset, batch_size=4, shuffle=False)


model_path = "saved_models/8x8_fine_tuned_kd.pth"
net = create_vit_with_patch_size((32, 32))
model = load_model(model_path, net=net)

layer_num = 11
features, labels = get_features(model, sampled_loader, layer_num)
visualize_tsne(features.numpy(), labels.numpy())
print("Visualized TSNE")


