"""
NOT FINISHED YET

train_model_2.py

Training loop for the VQ-VAE on PBMC 68k scRNA-seq data.
Plugs directly into the DataLoader built in load_data.py and the
model defined in VQVAE.py.

Run:
    python train_model_2.py
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt


from PBMC.load_data import dataloader, input_dimension
from Architecture_2 import VQVAE


# 1. HYPERPARAMETERS

# Tune these for optimisation. Defaults were determined from literature with exception of num_embeddings

HIDDEN_DIM      = 512    # width of encoder/decoder hidden layers
LATENT_DIM      = 64     # size of each codebook vector (z_e & z_q dimension)
NUM_EMBEDDINGS  = 20    # K: number of codebook entries
COMMITMENT_COST = 0.25   # beta: weight on commitment loss in Vector quantiser

LEARNING_RATE   = 1e-3   # Adam initial learning rate
NUM_EPOCHS      = 20     # total training epochs (increase for better results)

# Device: use GPU if available, otherwise CPU -> will change after access to server
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")


# 2. MODEL, OPTIMISER, (SCHEDULER if needed)
model = VQVAE(
    input_dim       = input_dimension,
    hidden_dim      = HIDDEN_DIM,
    latent_dim      = LATENT_DIM,
    num_embeddings  = NUM_EMBEDDINGS,
    commitment_cost = COMMITMENT_COST,
).to(DEVICE)

print(f"\nModel architecture:\n{model}\n")

# Count and print total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}\n")

# Adam is standard for VQ-VAE; weight_decay adds mild L2 regularisation
# which helps prevent overfitting on the decoder's large output layer.
optimiser = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Reduce learning rate by 50% if val loss doesn't improve for n epochs.
# 'patience=5' gives the model a few epochs to recover before stepping down.
scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)


# 3. LOSS FUNCTION

def compute_loss(x: torch.Tensor, x_recon: torch.Tensor, vq_loss: torch.Tensor):
    """
    Total VQ-VAE loss = Reconstruction loss + VQ loss.

    Reconstruction loss (MSE):
        Measures how well the decoder recovers the original gene expression
        from the quantized latent code. MSE is appropriate here because the
        data is log-normalised (continuous, non-negative, roughly Gaussian).

    VQ loss:
        Already computed inside VectorQuantizer.forward(): it combines the
        codebook loss (pulls codebook towards encoder) and the commitment
        loss (pulls encoder towards codebook). I just add it here.
    """
    recon_loss = F.mse_loss(x_recon, x)   # scalar: mean over batch and genes
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss


# 4. TRAINING LOOP

# Track losses across epochs for plotting later
history = {"total": [], "recon": [], "vq": []}

print("Starting training...")

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()

    # Accumulators - later sum over batches then divide at the end of the epoch
    epoch_total = 0.0
    epoch_recon = 0.0
    epoch_vq    = 0.0

    for batch_idx, x_batch in enumerate(dataloader):
        # x_batch: (256, n_genes) float32 tensor, already prepared by load_data.py

        x_batch = x_batch.to(DEVICE)   # move to GPU if available

        # forward pass
        x_recon, vq_loss, _ = model(x_batch)
        # x_recon   : reconstructed gene expression (256, n_genes)
        # vq_loss   : codebook + commitment loss scalar
        # _         : (encoding_indices) — not needed during training so omitted with '_'
        # compute loss
        total_loss, recon_loss = compute_loss(x_batch, x_recon, vq_loss)

        # backward pass
        optimiser.zero_grad()  # clear gradients from previous step
        total_loss.backward()  # compute gradients via autograd
        optimiser.step()  # update weights

        # Accumulate batch losses (detach to free computation graph)
        epoch_total += total_loss.item()
        epoch_recon += recon_loss.item()
        epoch_vq += vq_loss.item()

    #At the end of epochs
    #Average losses over the number of batches in the epoch
    n_batches = len(dataloader)
    avg_total = epoch_total / n_batches
    avg_recon = epoch_recon / n_batches
    avg_vq = epoch_vq / n_batches

    # History store
    history["total"].append(avg_total)
    history["recon"].append(avg_recon)
    history["vq"].append(avg_vq)

    # Step the learning-rate scheduler using total loss
    scheduler.step(avg_total)

    # Print a summary every epoch (add a modulo condition to print less often)
    print(
        f"Epoch [{epoch:>3}/{NUM_EPOCHS}] | "
        f"Total: {avg_total:.4f} | "
        f"Recon: {avg_recon:.4f} | "
        f"VQ: {avg_vq:.4f}"
    )
# 5. SAVE MODEL CHECKPOINT

checkpoint_path = "PBMC/vqvae_checkpoint.pth"

# We save the full model state so it can be reloaded without re-training.
# Also saving hyperparameters so the exact architecture can be reconstructed.
torch.save(
    {
        "epoch":            NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimiser_state":  optimiser.state_dict(),
        "hyperparameters": {
            "input_dim":       input_dimension,
            "hidden_dim":      HIDDEN_DIM,
            "latent_dim":      LATENT_DIM,
            "num_embeddings":  NUM_EMBEDDINGS,
            "commitment_cost": COMMITMENT_COST,
        },
        "final_losses": {
            "total": history["total"][-1],
            "recon": history["recon"][-1],
            "vq":    history["vq"][-1],
        },
    },
    checkpoint_path,
)
print(f"\nModel checkpoint saved to '{checkpoint_path}'")

# 6. PLOT TRAINING CURVES

epochs_range = range(1, NUM_EPOCHS + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("VQ-VAE Training Curves — PBMC 68k", fontsize=14)

# Total loss
axes[0].plot(epochs_range, history["total"], color="royalblue")
axes[0].set_title("Total Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# Reconstruction loss
axes[1].plot(epochs_range, history["recon"], color="darkorange")
axes[1].set_title("Reconstruction Loss (MSE)")
axes[1].set_xlabel("Epoch")

# VQ loss
axes[2].plot(epochs_range, history["vq"], color="forestgreen")
axes[2].set_title("VQ Loss (Codebook + Commitment)")
axes[2].set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("PBMC/training_curves.png", dpi=150)
plt.show()
print("Training curves saved to 'PBMC/training_curves.png'")

# 7. CODEBOOK USAGE CHECK

# After training, a healthy VQ-VAE should use many (ideally most) of its K
# codebook entries. If only a few entries are used, the codebook has 'collapsed'
# — a common failure mode. Check this before doing any downstream analysis.

model.eval()
all_indices = []

with torch.no_grad():
    for x_batch in dataloader:
        x_batch = x_batch.to(DEVICE)
        indices = model.encode(x_batch)          # (batch,) — discrete codes
        all_indices.append(indices.cpu())

all_indices = torch.cat(all_indices)             # (n_cells,)
unique_codes = all_indices.unique().numel()

print(f"\nCodebook usage: {unique_codes} / {NUM_EMBEDDINGS} entries used "
      f"({100 * unique_codes / NUM_EMBEDDINGS:.1f}%)")

if unique_codes < NUM_EMBEDDINGS * 0.5:
    print("WARNING: Less than 50% of codebook used — consider reducing "
          "NUM_EMBEDDINGS or adding codebook reset / EMA updates.")
else:
    print("Codebook utilisation looks healthy!")









