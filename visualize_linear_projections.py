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
from sklearn.preprocessing import MinMaxScaler



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

def scale_projections(projections: np.ndarray):
    projection_dim = projections.shape[-1]
    patch_h, patch_w, patch_channels = projections.shape[:-1]

    scaled_projections = MinMaxScaler().fit_transform(
        projections.reshape(-1, projection_dim)
    )
    scaled_projections = scaled_projections.reshape(
        patch_h, patch_w, patch_channels, -1
    )
    return scaled_projections


def display_projections(scaled_projections: np.ndarray, save_plot=None):
    fig, axes = plt.subplots(nrows=8, ncols=16, figsize=(13, 8))
    img_count = 0
    limit = 128

    for i in range(8):
        for j in range(16):
            if img_count < limit:
                axes[i, j].imshow(scaled_projections[..., img_count])
                axes[i, j].axis("off")
                img_count += 1

    fig.tight_layout()
    fig.savefig(save_plot, dpi=300, bbox_inches="tight")
    # plt.show()


model_path = "saved_models/8x8_fine_tuned.pth"
net = create_vit_with_patch_size((8, 8))
model = load_model(model_path, net=net)

patch_embedding_weights = model.patch_embed.proj.weight.data.cpu().numpy().transpose(2, 3, 1, 0)

scaled_projections = scale_projections(patch_embedding_weights)
display_projections(scaled_projections=scaled_projections, save_plot="linear_proj_8.png")
print("Visualized linear projections")


