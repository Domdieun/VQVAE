def train_and_evaluate(
    hidden_dim,
    latent_dim,
    num_embeddings,
    commitment_cost,
    learning_rate,
    batch_size,
    num_epochs,
):
    # 1. train model
    # 2. save checkpoint
    # 3. extract cluster assignments
    # 4. compare to annotations
    # 5. return metrics dict
    return {
        "val_total_loss": ...,
        "val_recon_loss": ...,
        "val_vq_loss": ...,
        "overall_purity": ...,
        "ari": ...,
        "nmi": ...,
        "codebook_usage": ...,
    }