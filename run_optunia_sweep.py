import optuna
from train_one_trial import train_and_evaluate

def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512])
    latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64])
    num_embeddings = trial.suggest_categorical("num_embeddings", [10, 20, 30, 50])
    commitment_cost = trial.suggest_categorical("commitment_cost", [0.1, 0.25, 0.5])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 5e-4, 1e-3])
    batch_size = trial.suggest_categorical("batch_size", [128, 256])

    results = train_and_evaluate(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=20,
    )

    trial.set_user_attr("overall_purity", results["overall_purity"])
    trial.set_user_attr("ari", results["ari"])
    trial.set_user_attr("nmi", results["nmi"])
    trial.set_user_attr("codebook_usage", results["codebook_usage"])

    return results["val_total_loss"]


study = optuna.create_study(
    direction="minimize",
    study_name="vqvae_pbmc",
    storage="sqlite:///vqvae_optuna.db",
    load_if_exists=True,
)

study.optimize(objective, n_trials=30)

print("Best trial:")
print(study.best_trial.number)
print(study.best_trial.value)
print(study.best_trial.params)
print(study.best_trial.user_attrs)