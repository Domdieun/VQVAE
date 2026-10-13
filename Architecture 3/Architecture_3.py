
"""
Architecture_3.py

Architecture 3: Architecture 1 backbone with Exponential Moving Average codebook updates.

Everything outside the vector quantizer follows the original Architecture 1:

    normalized expression
    -> two-layer encoder
    -> z_e
    -> nearest codebook vector z_q
    -> two-layer decoder
    -> reconstructed normalized expression

Key EMA difference
The codebook vectors are NOT optimized with gradient descent.

For each minibatch, the quantizer records:

    N_k = number of encoder vectors assigned to code k
    S_k = sum of encoder vectors assigned to code k

It updates running statistics:

    ema_count_k <- decay * ema_count_k
                   + (1 - decay) * N_k

    ema_sum_k   <- decay * ema_sum_k
                   + (1 - decay) * S_k

The codebook vector is then:

    e_k <- ema_sum_k / smoothed_ema_count_k

Only the commitment loss sends gradients into the encoder:

    vq_loss = commitment_cost * MSE(z_e, stop_gradient(z_q))

The straight-through estimator still passes decoder gradients to z_e.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """
    Vector quantizer where the codebook is updated using
    Exponential Moving Average (EMA), rather than gradient descent.

    Main steps during training:

    1. The encoder produces continuous latent vectors z_e.
    2. For every z_e, find the nearest codebook vector.
    3. Assign each z_e to that code.
    4. Count how many z_e vectors were assigned to each code.
    5. Sum the z_e vectors assigned to each code.
    6. Update running EMA counts and EMA sums.
    7. Recalculate each codebook vector as:

           codebook vector
           = EMA sum / EMA count

    Therefore, each codebook vector behaves approximately like a
    slowly moving centroid of the encoder outputs assigned to it.

    Args:
        num_embeddings:
            Number of codebook vectors K.

        embedding_dim:
            Dimension D of every codebook vector.

        commitment_cost:
            Weight beta on the encoder commitment loss.

        decay:
            EMA decay. Values close to 1 update the codebook more slowly.

        epsilon:
            Numerical stability value used when normalizing EMA counts.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1.")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.epsilon = float(epsilon)

        # nn.Embedding is being used here as a convenient matrix containing all codebook vectors.
        # Shape: embedding.weight = (num_embeddings, embedding_dim)
        # Example:
        #     K = 20
        #     D = 64
        # gives:
        #     embedding.weight.shape = (20, 64)
        # Each row is one codebook vector:
        #     e_0
        #     e_1
        #     e_2

        self.embedding = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
        )
        # Before training, the codebook needs starting positions.
        # The values are randomly drawn from: [-1/K, +1/K]

        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / self.num_embeddings,
            1.0 / self.num_embeddings,
        )

        # The codebook must never receive gradients from the optimiser.
        self.embedding.weight.requires_grad_(False)

        # Buffers are saved in model.state_dict() but are not trainable
        # parameters. This allows checkpointing.

        # EMA needs to remember information from previous minibatches.
        # Two main statistics are required:
        # 1. How often each code has been selected
        # 2. The sum of the z_e vectors assigned to each code

        # ema_cluster_size[k] stores the EMA estimate of how many
        # encoder vectors are being assigned to code k.
        # Initially:
        #     [0, 0, 0, ..., 0]
        # Shape:
        #     (num_embeddings,)
        # Example for 4 codes:
        #     [0, 0, 0, 0]
        # Later it might become something like:
        #     [8.4, 15.7, 3.2, 6.9]
        # These are not necessarily integers because they are exponentially averaged values.

        self.register_buffer(
            "ema_cluster_size",
            torch.zeros(self.num_embeddings),
        )

        # For each code k, we also need to remember the running
        # sum of encoder representations assigned to that code.
        # detach() ensures this copy is not connected to autograd.
        # clone() creates an independent tensor
        self.register_buffer(
            "ema_embedding_sum",
            self.embedding.weight.detach().clone(),
        )
        #records how many times _ema_update() has run.useful for debugging, checking inspection etc.
        self.register_buffer(
            "ema_update_steps",
            torch.zeros((), dtype=torch.long), # does not directly affect the ema equation
        )

    def _squared_distances(
        self,
        z_e: torch.Tensor,
    ) -> torch.Tensor:
        #calculate the squared magnitude of its latent vector for every codebook vector
        z_e_squared = (z_e ** 2).sum(dim=1, keepdim=True)
        embedding_squared = (
            self.embedding.weight ** 2
        ).sum(dim=1)
        #Matrix multiplication compares every encoder vector with every codebook vector.
        #Therefore every row gives the comparison of one cell against every possible code.
        cross_term = z_e @ self.embedding.weight.t()

        # Example:
        #
        #             code0 code1 code2
        #     cell0     2     8     4
        #     cell1     7     1     6
        #
        # Cell 0 would choose code 0.
        # Cell 1 would choose code 1.

        return (
            z_e_squared
            + embedding_squared
            - 2.0 * cross_term
        )

    @torch.no_grad()
    def _ema_update(
        self,
        z_e: torch.Tensor,
        encoding_indices: torch.Tensor,
    ) -> None:
        """
        Update EMA counts, EMA sums, and codebook vectors.

        z_e is detached before this method is called.
        """

# Convert assignments into one-hot vectors
# encoding_indices contains the selected code for each cell.
        # eg. coding_indices = [0, 2, 0, 1]
        # After one-hot encoding:
        #             code0 code1 code2
        #     cell0     1     0     0
        #     cell1     0     0     1
        #     cell2     1     0     0
        #     cell3     0     1     0
        # Shape: (batch_size, num_embeddings)

        encodings = F.one_hot(
            encoding_indices,
            num_classes=self.num_embeddings,
        ).to(dtype=z_e.dtype)

#count how many cells use each code
        #eg. code 0 -> 2 cells
        #     code 1 -> 1 cell
        #     code 2 -> 1 cell
        #     batch_cluster_size = [2, 1, 1]

        batch_cluster_size = encodings.sum(dim=0)

        #Sum z_e vectors belonging to each code
        batch_embedding_sum = encodings.t() @ z_e

#EMA UPDATE OF CLUSTER COUNTS
        # ema_cluster_size =
        #     decay * old_ema_cluster_size + (1 - decay) * batch_cluster_size

        # With decay = 0.99:
        #     new_count =
        #         0.99 * historical_count  + 0.01 * current_batch_count

        # The "_" in mul_() and add_() means the tensor is modified
        # IN PLACE rather than creating a new tensor.

        self.ema_cluster_size.mul_(self.decay).add_(
            batch_cluster_size,
            alpha=1.0 - self.decay,
        )
        #EMA UPDATE OF LATENT SUMS
        # ema_embedding_sum =
        #     decay * old_ema_embedding_sum + (1 - decay) * batch_embedding_sum
        # Therefore the model gradually remembers the location
        # of all the encoder vectors that have previously used
        # each code.

        self.ema_embedding_sum.mul_(self.decay).add_(
            batch_embedding_sum,
            alpha=1.0 - self.decay,
        )

        # Record that one EMA update has occurred.
        self.ema_update_steps.add_(1)

        #Find the total effective number of assignments
        # eg. ema_cluster_size = [4.2, 8.1, 3.7]
        # total_count = 16.0

        total_count = self.ema_cluster_size.sum()

        # smoothing keeps denominators positive while preserving the
        # total effective count.
        # This looks complicated, but its purpose is simple:
        # Prevent codebook vectors with extremely small counts
        # from causing:
        #     embedding_sum / 0
        # or:
        #     embedding_sum / extremely_tiny_number
        # which could produce:
        #     inf
        #     huge values
        #     NaN
        # epsilon = 1e-5 gives every cluster a tiny amount
        # of numerical protection.
        # The rescaling by total_count tries to preserve
        # approximately the same overall effective count.
        smoothed_cluster_size = (
            (
                self.ema_cluster_size + self.epsilon
            )
            / (
                total_count
                + self.num_embeddings * self.epsilon
            )
            * total_count
        ).clamp_min(self.epsilon)

        #CALCULATE THE NEW CODEBOOK CENTROIDS

        # Why unsqueeze(1)?
        # smoothed_cluster_size originally has shape:(K,) eg [5, 10, 3]
        # But ema_embedding_sum has: (K, D)
        # unsqueeze(1) changes counts to: (K, 1)
        # allowing PyTorch broadcasting:
        #     code0 vector / code0 count
        #     code1 vector / code1 count

        updated_embeddings = (
            self.ema_embedding_sum
            / smoothed_cluster_size.unsqueeze(1)
        )


        #IDENTIFY CODES THAT HAVE EVER BEEN USED - figure codes that have been used and not been selected
        # A never-used code has count exactly zero. Keep its initialized vector
        # instead of dividing its initial EMA sum by a tiny denominator.
        ever_used = self.ema_cluster_size > 0

        #DON'T DESTROY NEVER-USED CODES
        # torch.where creates a full replacement tensor. This avoids boolean
        # advanced indexing, which would otherwise update a temporary copy
        # rather than the original codebook parameter.

        # torch.where(condition, A, B)
        # means:
        #     if condition is True: use A
        #     otherwise: use B

        # USED code:
        #     replace with EMA-updated centroid
        # UNUSED code:
        #     retain its old codebook vector

        # This is important because a never-used code has essentially
        # zero EMA count.
        # Without this protection, dividing by the tiny epsilon value
        # could produce a meaningless codebook location.

        replacement = torch.where(
            ever_used.unsqueeze(1),
            updated_embeddings,
            self.embedding.weight.data,
        )
        #ACTUALLY REPLACE THE CODEBOOK
        # This copies the newly calculated EMA centroids into the actual codebook.
        # After this point:
        #     self.embedding.weight contains the newly updated code vectors.
        self.embedding.weight.data.copy_(replacement)

    def forward(
        self,
        z_e: torch.Tensor,
    ):
        if z_e.ndim != 2:
            raise ValueError(
                "VectorQuantizerEMA expects z_e with shape "
                "(batch, embedding_dim)."
            )

        if z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected latent dimension {self.embedding_dim}, "
                f"received {z_e.shape[1]}."
            )

        distances = self._squared_distances(z_e)
        encoding_indices = distances.argmin(dim=1)

        # Use the current codebook values for this minibatch.
        z_q = self.embedding(encoding_indices)

        # EMA updates happen only during training, never during validation,
        # testing, cluster extraction, or UMAP creation.
        if self.training:
            self._ema_update(
                z_e=z_e.detach(),
                encoding_indices=encoding_indices,
            )

        # EMA replaces the original gradient-based codebook loss. Only the
        # encoder commitment term remains in vq_loss.
        commitment_loss = F.mse_loss(
            z_e,
            z_q.detach(),
        )
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-through estimator:
        # forward value equals z_q, backward gradient flows through z_e.
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, vq_loss, encoding_indices

    @torch.no_grad()
    def codebook_usage_from_indices(
        self,
        indices: torch.Tensor,
    ) -> dict:
        counts = torch.bincount(
            indices.view(-1),
            minlength=self.num_embeddings,
        )

        unique_codes = int((counts > 0).sum().item())

        probabilities = counts.float()
        probabilities = probabilities / probabilities.sum().clamp_min(1.0)

        nonzero = probabilities > 0
        perplexity = torch.exp(
            -(
                probabilities[nonzero]
                * probabilities[nonzero].log()
            ).sum()
        ).item()

        return {
            "counts": counts,
            "unique_codes": unique_codes,
            "usage_percent": (
                100.0 * unique_codes / self.num_embeddings
            ),
            "perplexity": float(perplexity),
        }


class Encoder(nn.Module):
    """
    Architecture_3 two-hidden-layer encoder (based on Architecture 1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 64,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Architecture_3 symmetric MSE decoder (based on Architecture 1).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 512,
        output_dim: int | None = None,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(z_q)


class VQVAE(nn.Module):
    """
    Architecture_3 with an EMA-updated vector quantizer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 64,
        num_embeddings: int = 20,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()

        self.input_dim = int(input_dim)

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        self.vq = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            decay=ema_decay,
            epsilon=ema_epsilon,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.vq(z_e)
        x_recon = self.decoder(z_q)

        return x_recon, vq_loss, encoding_indices

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        z_e = self.encoder(x)
        _z_q, _vq_loss, indices = self.vq(z_e)

        return indices

    def encode_continuous(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(x)

    def decode(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        z_q = self.vq.embedding(indices)
        return self.decoder(z_q)
