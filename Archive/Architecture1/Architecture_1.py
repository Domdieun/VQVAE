#!/usr/bin/env python3

"""
Extensive VQ-VAE Architecture for scRNA-seq Clustering.
Optimized for the 10x Genomics PBMC 68k dataset.

Features:
- Deep MLP architecture with 1D Residual Connections
- Layer Normalization (optimal for single-cell sparsity)
- Exponential Moving Average (EMA) Vector Quantization
- PyTorch Lightning integration for scalable training
"""

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl


# 1. Building Blocks

class ResidualBlock1D(nn.Module):
    """Residual connection for tabular/1D data to prevent vanishing gradients."""

    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor):
        return x + self.block(x)


class Encoder(nn.Module):
    """Compresses the high-dimensional gene expression matrix into a dense latent space."""

    def __init__(self, input_dim: int, hidden_dims: list, embedding_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                #ResidualBlock1D(h_dim)
            ])
            in_dim = h_dim

        # Final projection to embedding space
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class Decoder(nn.Module):
    """Reconstructs the original gene expression profile from the quantized latent vectors."""

    def __init__(self, embedding_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        in_dim = embedding_dim

        # Reverse the hidden dimensions for the decoder
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                ResidualBlock1D(h_dim)
            ])
            in_dim = h_dim

        # Output layer (no activation, assuming MSE loss on log1p normalized data)
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.decoder(x)



# 2. Vector Quantization (EMA Approach)

class EMAVectorQuantizer(nn.Module):
    """
    EMA Vector Quantizer optimized for 1D tabular inputs.
    Prevents codebook collapse by updating centroids via moving averages.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int, commitment_cost: float = 0.25, decay: float = 0.99,
                 epsilon: float = 1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize the codebook
        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', embed.clone())

    def forward(self, x: torch.Tensor):
        # x is of shape [batch_size, embedding_dim]
        # Calculate Euclidean distances: (x-y)^2 = x^2 - 2xy + y^2
        distances = (torch.sum(x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding ** 2, dim=0)
                     - 2 * torch.matmul(x, self.embedding))

        # Find closest cluster centers
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the embeddings
        quantized = torch.matmul(encodings, self.embedding.t())

        # Use EMA to update the embedding vectors (only during training)
        if self.training:
            with torch.no_grad():
                self.cluster_size.data.mul_(self.decay).add_(
                    encodings.sum(0), alpha=1 - self.decay
                )

                # Laplace smoothing
                n = self.cluster_size.sum()
                cluster_size = (
                        (self.cluster_size + self.epsilon)
                        / (n + self.num_embeddings * self.epsilon) * n
                )

                embed_sum = torch.matmul(encodings.t(), x)
                self.ema_w.data.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)
                self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(0))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss, encoding_indices



# 3. Complete VQ-VAE Model & Lightning Wrapper

class scRNA_VQVAE(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256],
                 embedding_dim: int = 64, num_embeddings: int = 512,
                 lr: float = 1e-3, quantize_on_epoch: int = 0):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.quantize_on_epoch = quantize_on_epoch

        # Networks
        self.encoder = Encoder(input_dim, hidden_dims, embedding_dim)
        self.vq_layer = EMAVectorQuantizer(embedding_dim, num_embeddings)
        self.decoder = Decoder(embedding_dim, hidden_dims, input_dim)

    def forward(self, x):
        enc = self.encoder(x)

        # Delayed quantization: allow autoencoder to warm up first
        if self.current_epoch >= self.quantize_on_epoch:
            quantized, vq_loss, encodings = self.vq_layer(enc)
        else:
            quantized = enc
            vq_loss = torch.tensor(0.0, device=x.device)
            encodings = None

        dec = self.decoder(quantized)
        return dec, vq_loss, encodings

    def training_step(self, batch, batch_idx):
        x = batch
        dec, vq_loss, _ = self(x)

        # Mean Squared Error for log1p normalized data
        reconstruction_loss = F.mse_loss(dec, x)
        total_loss = reconstruction_loss + vq_loss

        self.log("train_recon_loss", reconstruction_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_vq_loss", vq_loss, on_step=False, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        # AdamW is often better than standard Adam for models with LayerNorm and Residuals
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer



# Testing Block (Run to verify dimensions)
if __name__ == "__main__":
    # Mock parameters matching your dataset context
    MOCK_BATCH_SIZE = 256
    MOCK_GENES = 32738  # Replace with actual input_dimension from your dataloader script

    print(f"Testing Architecture initialization for {MOCK_GENES} genes...")

    model = scRNA_VQVAE(
        input_dim=MOCK_GENES,
        hidden_dims=[512, 256],  # Deep narrowing layers
        embedding_dim=64,  # Latent space dimensions
        num_embeddings=512  # Number of clusters (codebook size)
    )

    mock_data = torch.rand(MOCK_BATCH_SIZE, MOCK_GENES)
    reconstruction, vq_loss, indices = model(mock_data)

    print(f"Input shape:  {mock_data.shape}")
    print(f"Output shape: {reconstruction.shape}")
    print(f"VQ Loss:      {vq_loss.item():.4f}")
    print("Architecture verified successfully!")