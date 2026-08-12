
# Encoder-Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE with SOM-idea neighborhood update.

    Inputs:
    - n_e : number of embeddings (should be a square number for 2D SOM grid)
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term
    - som_grid_shape: tuple (height, width) for 2D SOM grid
    - neighbor_sigma: controls strength of neighbor update (Gaussian kernel)
    """

    def __init__(self, n_e, e_dim, beta, som_grid_shape=None, neighbor_sigma=1.0):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        # SOM parameters
        self.som_grid_shape = som_grid_shape
        self.neighbor_sigma = neighbor_sigma

        if som_grid_shape is not None:
            self.grid_coords = self._create_som_grid(som_grid_shape)

    def _create_som_grid(self, shape):
        h, w = shape
        coords = torch.stack(torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing='ij'), dim=-1
        ).reshape(-1, 2)  # shape [n_e, 2]
        return coords.float()

    def _compute_som_kernel(self, winner_idx):
        """Compute Gaussian neighborhood weights from winner index."""
        if self.grid_coords is None:
            return None

        winner_coord = self.grid_coords[winner_idx]  # [2]
        distances = torch.sum((self.grid_coords - winner_coord)**2, dim=1)  # [n_e]
        kernel = torch.exp(-distances / (2 * self.neighbor_sigma**2))  # [n_e]
        return kernel / kernel.sum()  # Normalize

    def forward(self, z, sel_idx=0):
        B, C, H, W = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # Compute distances to codebook
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        sorted_indices = torch.argsort(d, dim=1)
        min_encoding_indices = sorted_indices[:, sel_idx].unsqueeze(1)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(B, H, W, C)

        loss1 = torch.mean((z_q.detach() - z)**2)
        loss2 = torch.mean((z_q - z.detach())**2)
        loss = loss1 + self.beta * loss2

        # SOM-style neighborhood update (during training only)
        if self.training and self.som_grid_shape is not None:
            with torch.no_grad():
                for idx in min_encoding_indices.view(-1).unique():
                    kernel = self._compute_som_kernel(idx.item()).to(z.device)  # [n_e]
                    selected = (min_encoding_indices.view(-1) == idx).nonzero(as_tuple=True)[0]
                    if selected.numel() == 0:
                        continue
                    z_avg = torch.mean(z_flattened[selected], dim=0)  # [e_dim]
                    delta = kernel[:, None] * (z_avg[None, :] - self.embedding.weight.data)
                    self.embedding.weight.data += 0.1 * delta  # update rate can be tuned

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        encoding_indices = min_encoding_indices.view(B, H, W)

        return loss, z_q, encoding_indices
    
#######################################################################

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - res_h_dim : the hidden dimension of the residual block

    Depthwise Separable Convolution-BatchNorm-SiLU archtecture
    """

    def __init__(self, in_dim, res_h_dim, depthwise=True):
        super(ResidualLayer, self).__init__()
        # why using sigmoid?

        if depthwise:
            self.res_block = nn.Sequential(
                DepthwiseSeparableConv2d(in_dim, res_h_dim),
                nn.BatchNorm2d(res_h_dim),
                nn.SiLU(True), 
                DepthwiseSeparableConv2d(res_h_dim, in_dim),
                nn.BatchNorm2d(in_dim),
                nn.SiLU(True), 
            )
        else:
            self.res_block = nn.Sequential(
                nn.Conv2d(in_dim, res_h_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(res_h_dim),
                nn.SiLU(True),
                nn.Conv2d(res_h_dim, in_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_dim),
                nn.SiLU(True),
            )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    
    def __init__(self, 
                 h_dim, res_dim, 
                 n_layers, 
                 depthwise=True):
        super(ResidualStack, self).__init__()

        modules = [ResidualLayer(h_dim, res_dim, depthwise) for _ in range(n_layers)]
        self.stack = nn.Sequential(*modules)
            
    def forward(self, x):
        return self.stack(x)

#######################################################################

class Encoder(nn.Module):
    """
    Input is RGB image
    5 blocks : 224 -> 112 -> 56  -> 28  -> 14
    #kernels : 32  -> 64  -> 128 -> 256 -> 512
    res_dim  : 64  -> 128 -> 256 -> 128 -> 256 (expand then compress design)
    """
    def __init__(self, config):
        super(Encoder, self).__init__()

        if config["depthwise"]:
            modules = [
                DepthwiseSeparableConv2d(3, config["nkernels"][0]),
                nn.BatchNorm2d(config["nkernels"][0]),
                nn.SiLU(True),
            ]
        else:
            modules = [
                nn.Conv2d(3, config["nkernels"][0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(config["nkernels"][0]),
                nn.SiLU(True),
            ]
        for i in range(config["nblocks"]):
            modules.append(
                ResidualStack(
                    h_dim=config["nkernels"][i], 
                    res_dim=config["res_dim"][i], 
                    n_layers=config["nlayers"][i], 
                    depthwise=config["depthwise"])
            )
            if i < config["nblocks"]-1:
                modules.append(nn.MaxPool2d(2, stride=2)) 
                modules.append(nn.Conv2d(config["nkernels"][i], config["nkernels"][i+1], kernel_size=1)) # adjust num kernels

        self.stack = nn.Sequential(*modules)

    def forward(self, x): # -> (B, 512,14,14)
        return self.stack(x)

class Decoder(nn.Module):
    """
    Decode RGB image from output of Encoder
    5 blocks : 14  -> 28  -> 56  -> 112 -> 224
    #kernels : 512 -> 256 -> 128 -> 64  -> 32
    res_h_dim: 256 -> 128 -> 256 -> 128 -> 64
    """
    def __init__(self, config):
        super(Decoder, self).__init__()

        modules = []
        for i in range(config["nblocks"]):
            modules.append(
                ResidualStack(
                    h_dim=config["nkernels"][i], 
                    res_dim=config["res_dim"][i], 
                    n_layers=config["nlayers"][i],
                    depthwise=config["depthwise"])
            )
            if i < config["nblocks"]-1:
                modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
                modules.append(nn.Conv2d(config["nkernels"][i], config["nkernels"][i+1], kernel_size=1)) # adjust num kernels
        
        modules.append(nn.Conv2d(config["nkernels"][-1], 3, kernel_size=1, bias=False))
        modules.append(nn.Sigmoid())

        self.stack = nn.Sequential(*modules)

    def forward(self, x): # -> (B, 3, 224,224)
        return self.stack(x)

#######################################################################

from copy import deepcopy
from pytorch_msssim import ms_ssim

class VQVAE(nn.Module):

    def __init__(self, config):
        super(VQVAE, self).__init__()

        self.config = deepcopy(config)
        self.encoder = Encoder(config["encoder"])

        self.pre_vq = nn.Conv2d(
            config["encoder"]["nkernels"][-1],
            config["embedding_dim"],
            kernel_size=1, bias=False)

        self.vq = VectorQuantizer(
            config["nembeddings"],
            config["embedding_dim"], 
            config["beta"],
            config["som_grid_shape"],
            config["neighbor_sigma"]
            )

        self.post_vq = nn.Conv2d(
            config["embedding_dim"],
            config["decoder"]["nkernels"][0],
            kernel_size=1, bias=False)

        self.decoder = Decoder(config["decoder"])

    def forward(self, x):

        z_e = self.encoder(x)
        z_e = self.pre_vq(z_e)
        
        embedding_loss, z_q, _ = self.vq(z_e)
        z_d = self.post_vq(z_q)

        x_hat = self.decoder(z_d)
        
        data_range = x.max() - x.min()
        
        perceptual_loss = 1 - ms_ssim(x_hat, x, data_range=data_range, size_average=True)
        
        l1_loss = F.l1_loss(x_hat, x)
        lambda_l1 = 0.5  # tune later
        
        recon_loss = perceptual_loss + lambda_l1 * l1_loss

        loss = recon_loss + embedding_loss
        
        return {
            "recon": x_hat, 
            "loss": loss, 
            "recon_loss": recon_loss, 
            "emb_loss": embedding_loss
        }

from PIL import Image
import torchvision.transforms as transforms

# Assume `tensor` is your input tensor of shape [C, H, W] with values in [0, 1] 
def tensor_to_image(tensor):
    # Convert tensor to PIL Image
    tensor = tensor.cpu().detach()  # Ensure tensor is on CPU and detached from graph
    tensor = tensor.clamp(0, 1)  # Ensure values are in [0, 1]
    tensor = tensor.mul(255).byte()  # Scale to [0, 255] and convert to uint8
    tensor = tensor.permute(1, 2, 0)  # Reshape from [C, H, W] to [H, W, C]
    numpy_image = tensor.numpy()  # Convert to NumPy array
    image = Image.fromarray(numpy_image)  # Convert to PIL Image
    return image

#######################################################################
import os

# os.makedirs(f"output/{modname}", exist_ok=True)

def continue_training(model):
    modname = model.config["modname"]
    os.makedirs(f"output/{modname}", exist_ok=True)
    filename = f"{modname}.pth"
    if not os.path.isfile(filename): 
        return model, 0
            
    # exist, then load and continue
    loaded_dict = torch.load(filename, weights_only=True)   
    model.load_state_dict(loaded_dict["model"])
    model.config = deepcopy(loaded_dict["config"])
    epoch = loaded_dict["epoch"]

    return model, epoch

#######################################################################
from lion_pytorch import Lion
import pickle

def train(model, data_loader, device, nepochs, save_interval=500):
    
    optimizer = Lion(model.parameters(), lr=2e-4, weight_decay=2e-4*10.0, use_triton=True)
    model, bepoch = continue_training(model)
    modname = model.config["modname"]
    log_path = f"output/{modname}/{modname}_loss_log.pkl"
    loss_log = {"epoch": [], "loss": [], "recon_loss": [], "emb_loss": []}

    for epoch in range(bepoch, nepochs):
        print("="*20)
        print(f"Epoch: {epoch}")

        closs, crecon_loss, cembedding_loss = 0, 0, 0

        for batch_idx, (images,_) in enumerate(data_loader):
            images = images.to(device)
            optimizer.zero_grad()
            out = model(images)
            out["loss"].backward()
            optimizer.step()
            
            torch.cuda.empty_cache()  # Clear cache after each backward pass

            closs += out["loss"].item()
            if "recon_loss" in out:
                crecon_loss += out["recon_loss"].item()
            if "emb_loss" in out:
                cembedding_loss += out["emb_loss"].item()
                
            if (batch_idx % save_interval == 0) or (batch_idx == len(data_loader)-1):
                """
                save model and print values
                """
                torch.save({
                    "model": model.state_dict(),
                    "config": model.config,
                    "epoch": epoch
                }, 
                f"output/{modname}/{modname}.pth")

                msg = f'Update {batch_idx}: Loss {closs/(batch_idx+1.):.5f}'
                if "recon_loss" in out:
                    msg += f', Recon loss {crecon_loss/(batch_idx+1.):.5f}'
                if "emb_loss" in out: ## suppose to be 'vq_loss' , 'emb_loss' should be used for vq_loss calculation
                    msg += f', Embedding loss {cembedding_loss/(batch_idx+1.):.5f}'
                print(msg)
        
        # Save log to pickle
        loss_log["epoch"].append(epoch)
        loss_log["loss"].append(closs / len(data_loader))
        loss_log["recon_loss"].append(crecon_loss / len(data_loader))
        loss_log["emb_loss"].append(cembedding_loss / len(data_loader))

        with open(log_path, "wb") as f:
            pickle.dump(loss_log, f)
        
        # Save image conditionally
        if epoch in (0, 99) or (epoch + 1) % 1000 == 0:
            img = tensor_to_image(images[0])
            rec = tensor_to_image(out["recon"][0])
            out_img = Image.new('RGB', (224*2, 224))
            out_img.paste(img, (0, 0))
            out_img.paste(rec, (224, 0))
            out_img.save(f"output/{modname}/{modname}_out_{epoch+1}.png") 

#######################################################################
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import pandas as pd

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) 
                           if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, image

# Example usage
def get_dataset(folder_path, image_size=(128, 128), csv_path=None, label_col=None):
    """
    Create a dataset with standard transformations.
    
    Args:
        folder_path (str): Path to image folder.
        image_size (tuple): Desired image size (height, width).
    
    Returns:
        Dataset: PyTorch Dataset for the images.
    """
    # Define transformations: resize, convert to tensor, normalize
    transform = Transforms.Compose([
        Transforms.Resize(image_size),  # Resize to fixed size
        Transforms.ToTensor()           # Convert to tensor (C, H, W), scales to [0,1]
    ])
    
    return ImageFolderDataset(folder_path, transform=transform)

#######################################################################
BaseEncoder = {
    "nblocks": 5,
    "nkernels": [32,  64, 128, 256, 512],
    "res_dim":  [64, 128, 256, 128, 256],
    "nlayers":  [ 2,   2,   2,   2,   2],
    "depthwise": True
}
BaseDecoder = {
    "nblocks": 5,
    "nkernels": [512, 256, 128,  64, 32],
    "res_dim":  [256, 128, 256, 128, 64],
    "nlayers":  [ 2,   2,   2,   2,   2],
    "depthwise": True
}
VQVAEConfig = {
    "encoder": BaseEncoder,
    "decoder": BaseDecoder,
    "nembeddings": 512, # 512 , 1024
    "embedding_dim": 64, # 8
    "beta": 0.25, # 0.25
    "som_grid_shape": (32,16) , # 32*32 for 1024 , 32*16 for 512
    "neighbor_sigma": 0.1 # fixed rate
}

from torch.utils.data import DataLoader

def trainae(folder_path, config):
    dataset = get_dataset(folder_path, image_size=(224, 224))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16) # nworker = ncpu
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VQVAE(config)
    
    model = model.to(device)
    train(model, dataloader, device, nepochs=5000, save_interval=500)

import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('conv', type=int, help='') # classical or depthwise
    parser.add_argument('name', type=str, help='') # saved name
    parser.add_argument('train', type=str, help='') # training images folder
    args = parser.parse_args()

    config = VQVAEConfig
    
    config["modname"] = args.name

    config["encoder"]["depthwise"] = (args.conv!=0) # 0 = normal conv
    config["decoder"]["depthwise"] = (args.conv!=0)

    trainae(args.folder, config)     