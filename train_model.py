#!/usr/bin/env python3

"""
train_model.py

This script connects the DataLoader to the VQ-VAE architecture
and runs the actual training loop across multiple epochs.
"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

# Import your data and architecture
# (Make sure your files are named load_data.py and architecture.py)
from PBMC.load_data import dataloader, input_dimension
from Architecture_1 import scRNA_VQVAE


def main():
    print(f"Initializing VQ-VAE for {input_dimension} genes...")

    # 1. Initialize the Model
    # We pass the exact number of genes from your dataloader
    model = scRNA_VQVAE(
        input_dim=input_dimension,
        hidden_dims=[512, 256],
        embedding_dim=64,
        num_embeddings=512,
        lr=1e-3,
        quantize_on_epoch=2  # Wait 2 epochs before snapping to clusters to let the network warm up
    )

    # 2. Setup the Checkpoint Saver
    # This automatically saves the best version of your model during training
    checkpoint_callback = ModelCheckpoint(
        dirpath='./saved_models/',
        filename='vqvae-{epoch:02d}-{train_total_loss:.2f}',
        save_top_k=1,
        monitor='train_total_loss',
        mode='min'
    )

    # 3. Initialize the PyTorch Lightning Trainer
    print("Spinning up the Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=20,  # How many times it loops through the entire dataset
        accelerator='auto',  # Automatically uses your GPU if you have one, or falls back to CPU
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # 4. START TRAINING!
    print("Starting the training loop. Watch the loss go down!")
    trainer.fit(model=model, train_dataloaders=dataloader)

    print("\nTraining Complete! Best model saved to the './saved_models/' folder.")


if __name__ == "__main__":
    main()
