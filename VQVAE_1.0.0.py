import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import the dataloader AND the exact number of genes
from PBMC_UMI import dataloader, input_dimension

# ==========================================
# 1. The Architecture
# ==========================================
class Encoder(nn.Module):
    # Changed input_dim to dynamically accept the variable
    def __init__(self, input_dim=input_dimension, hidden_dim=512, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

# ... (Keep the VectorQuantizer exactly the same) ...

class Decoder(nn.Module):
    # Changed output_dim to dynamically accept the variable
    def __init__(self, latent_dim=64, hidden_dim=512, output_dim=input_dimension):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class VQVAE(nn.Module):
    # Changed input_dim to dynamically accept the variable
    def __init__(self, input_dim=input_dimension, hidden_dim=512, latent_dim=64, num_embeddings=512):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, cluster_indices = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, cluster_indices

# ... (Keep the Training Loop exactly the same, EXCEPT this one line):
# Change the model initialization in the training loop to this:
    model = VQVAE(input_dim=input_dimension, hidden_dim=512, latent_dim=64, num_embeddings=512).to(device)