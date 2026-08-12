import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from lion_pytorch import Lion
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image


# PixelCNNPrior Model
class PixelCNNPrior(nn.Module):
    def __init__(self, shape, n_embeddings, embedding_dim, hidden_channels=64):
        super().__init__()
        H, W = shape
        self.shape = shape
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(1, hidden_channels, 7, padding=3)
        self.layers = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
            for _ in range(7)
        ])
        self.out = nn.Conv2d(hidden_channels, n_embeddings, 1)

    def forward(self, x):
        # x: (B, H, W) of discrete indices
        x = F.one_hot(x, num_classes=self.n_embeddings).float()  # (B,H,W,K)
        x = x.permute(0, 3, 1, 2)  # (B,K,H,W)
        x = self.conv1(x[:, :1])   # crude mask (could be improved)

        for layer in self.layers:
            x = torch.relu(layer(x))

        return self.out(x)  # (B,K,H,W)

    @torch.no_grad()
    def sample(self, device, n=16, temp=1.0):
        B, H, W = n, *self.shape
        samples = torch.zeros(B, H, W, dtype=torch.long, device=device)
        for i in range(H):
            for j in range(W):
                logits = self.forward(samples)
                probs = F.softmax(logits[:, :, i, j] / temp, dim=-1)
                samples[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)
        return samples


# Training function
def train_pixelcnn(prior, train_loader, device, epochs):
    optimizer = Lion(prior.parameters(), lr=2e-4, weight_decay=2e-4, use_triton=True)
    prior.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            logits = prior(batch)  # (B,K,H,W)
            loss = F.cross_entropy(
                logits.permute(0, 2, 3, 1).reshape(-1, prior.n_embeddings),
                batch.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss {total_loss / len(train_loader):.4f}")

# Main script
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "sample"], required=True)
    parser.add_argument("--latent_h", type=int, default=14)
    parser.add_argument("--latent_w", type=int, default=14)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--vqvae_ckpt", type=str, default="")
    parser.add_argument("--image_folder", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------- TRAIN MODE -----------------
    if args.mode == "train":
        if args.vqvae_ckpt and args.image_folder:
            print(f"Using VQ-VAE ({args.vqvae_ckpt}) to encode {args.image_folder}")

            from VQSOM import VQVAE, VQVAEConfig

            config = VQVAEConfig.copy()
            config["encoder"]["depthwise"] = False
            config["decoder"]["depthwise"] = False

            vqvae = VQVAE(config).to(device)
            vqvae.load_state_dict(torch.load(args.vqvae_ckpt, map_location=device))
            vqvae.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            dataset = datasets.ImageFolder(args.image_folder, transform=transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            all_indices = []
            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(device)
                    z = vqvae.encoder(x)
                    _, _, indices = vqvae.vq(z, None)
                    all_indices.append(indices.cpu())

            data = torch.cat(all_indices, dim=0)  # (N,H,W)
            print("Encoded latents shape:", data.shape)
            
            if args.embedding_dim is None:
                args.embedding_dim = vqvae.vq.embedding.weight.shape[1]
                print(f"[INFO] Inferred embedding_dim from VQ-VAE: {args.embedding_dim}")

        elif args.dataset:
            print(f"Loading precomputed latent indices from: {args.dataset}")
            data = torch.load(args.dataset)

        else: ###
            print("Using dummy random dataset [for debugging].")
            data = torch.randint(0, args.n_embeddings, (1000, args.latent_h, args.latent_w))

        train_loader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=True)

        prior = PixelCNNPrior(
            (args.latent_h, args.latent_w),
            args.n_embeddings,
            args.embedding_dim
        ).to(device)

        # optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr)

        train_pixelcnn(prior, train_loader, device, args.epochs)
        torch.save(prior.state_dict(), "pixelcnn_prior_test.pt") #pixelcnn_train_img_prior.pt , pixelcnn_test_img_prior.pt
        print(f"[DEBUG] Using embedding_dim={args.embedding_dim}, n_embeddings={args.n_embeddings}")
        print("Prior saved to pixelcnn_prior_test.pt")

# ----------------- SAMPLE MODE -----------------
    elif args.mode == "sample":
        print("Loading model from pixelcnn_prior_test.pt...")

        # Load trained PixelCNNPrior
        prior = PixelCNNPrior(
            (args.latent_h, args.latent_w),
            args.n_embeddings,
            args.embedding_dim
        ).to(device)
        prior.load_state_dict(torch.load("pixelcnn_prior_test.pt", map_location=device))
        prior.eval()

        # Sample from the PixelCNN prior
        samples = prior.sample(device, n=16, temp=args.temp)  # (B,H,W)
        print("Generated samples shape:", samples.shape)

        # Save sampled discrete latent indices
        torch.save(samples, "pixelcnn_samples_test.pt")
        print("Samples saved to pixelcnn_samples_test.pt")

        # If VQ-VAE checkpoint is provided, decode samples into images
        if args.vqvae_ckpt:
            print("Decoding samples with VQ-VAE...")

            from VQSOM import VQVAE, VQVAEConfig  # Adjust if filename is different

            config = VQVAEConfig.copy()
            config["encoder"]["depthwise"] = False
            config["decoder"]["depthwise"] = False

            # Load VQ-VAE
            vqvae = VQVAE(config).to(device)
            state_dict = torch.load(args.vqvae_ckpt, map_location=device)

            # Try to load just the model weights
            if isinstance(state_dict, dict) and "model" in state_dict:
                vqvae.load_state_dict(state_dict["model"])
            else:
                vqvae.load_state_dict(state_dict)

            vqvae.eval()
            
            if args.embedding_dim is None:
                args.embedding_dim = vqvae.vq.embedding.weight.shape[1]
                print(f"[INFO] Inferred embedding_dim from VQ-VAE: {args.embedding_dim}")

            # Convert sampled indices to quantized embeddings
            with torch.no_grad():
                if not hasattr(vqvae.vq, "get_codebook_entry"):
                    def get_codebook_entry(indices):
                        z_q = vqvae.vq.embedding(indices)  # (B, H, W, D)
                        return z_q.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
                    vqvae.vq.get_codebook_entry = get_codebook_entry

                z_q = vqvae.vq.get_codebook_entry(samples)  # should be (B, 64, H, W)
                print("[DEBUG] z_q shape before post_vq:", z_q.shape)

                print(f"Expected embedding_dim: {args.embedding_dim}")
                print(f"vqvae.vq.embedding.weight.shape: {vqvae.vq.embedding.weight.shape}")

                z_q = vqvae.post_vq(z_q)  # should be [B, 512, H, W]
                print("[DEBUG] z_q shape after post_vq:", z_q.shape)
                
                # Decode to image
                recon = vqvae.decoder(z_q)  # (B, 3, H_out, W_out)

                # Normalize if in range [-1, 1]
                recon = (recon + 1) / 2
                recon.clamp_(0, 1)

                # Save to image grid
                save_image(recon, "pixelcnn_samples_test.png", nrow=4)
                print("Reconstructions saved to pixelcnn_samples_test.png")
        else:
            print("VQ-VAE checkpoint not provided — skipping decoding.")


if __name__ == "__main__":
    main()

# python pixelCNN.py --mode train --epochs 1000 --batch_size 64 --lr 3e-4 --latent_h 14 --latent_w 14 --dataset /latents.pt
# python pixelCNN.py --mode sample --latent_h 14 --latent_w 14 --temp 1.0 --vqvae_ckpt /trained_weight.pth
