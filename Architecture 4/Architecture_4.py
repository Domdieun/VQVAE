"""
Architecture_4.py

A Vector Quantized Variational Autoencoder (VQ-VAE) for single-cell RNA-seq
with orthogonal codebook vector regularisation.

THIS VERSION USES MSE RECONSTRUCTION LOSS.

New addition: - Orthogonal codebook regularisation encourages different codebook vectors to be dissimilar / close to orthogonal.
- This is inspired by implementations such as lucidrains/vector-quantize-pytorch,

- Orthogonal regularisation is added inside the VQ auxiliary loss:
      VQ loss = codebook loss
                + beta * commitment loss
                + orthogonal_reg_weight * orthogonal_loss

Use normalized/log1p expression matrices for this architecture, not raw-count
matrices intended for Negative Binomial loss.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Learnable VQ-VAE codebook with optional orthogonal regularisation.

    Args:
        num_embeddings:
            K, number of codebook vectors.
        embedding_dim:
            D, dimensionality of each codebook vector.
        commitment_cost:
            Beta term on the encoder commitment loss.
        orthogonal_reg_weight:
            Weight of the orthogonal codebook regularisation term.
            0.0 disables it.
        orthogonal_reg_max_codes:
            Optional maximum number of codebook vectors to include in the
            orthogonal loss. Useful only for large codebooks.
        orthogonal_reg_active_codes_only:
            If True, regularise only codebook vectors selected in the current
            batch. For small codebooks, False is usually better.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_max_codes: Optional[int] = None,
        orthogonal_reg_active_codes_only: bool = False,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def orthogonal_regularization_loss(
        self,
        encoding_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Penalise cosine-similarity between different codebook vectors.

        Let E be the selected codebook matrix with shape K x D.
        1. L2-normalise rows of E.
        2. Compute Gram matrix G = E_norm @ E_norm.T.
        3. Penalise mean squared error from the identity matrix.

        For normalized vectors, the diagonal is already close to 1. The useful
        effect is mainly on off-diagonal terms, encouraging different codebook
        vectors to be decorrelated.
        """
        if self.orthogonal_reg_weight <= 0:
            return self.embedding.weight.new_tensor(0.0)

        codebook = self.embedding.weight

        if self.orthogonal_reg_active_codes_only:
            if encoding_indices is None:
                raise ValueError(
                    "encoding_indices must be provided when "
                    "orthogonal_reg_active_codes_only=True"
                )
            unique_indices = torch.unique(encoding_indices)
            codebook = codebook[unique_indices]

        if (
            self.orthogonal_reg_max_codes is not None
            and codebook.shape[0] > self.orthogonal_reg_max_codes
        ):
            perm = torch.randperm(codebook.shape[0], device=codebook.device)
            selected = perm[: self.orthogonal_reg_max_codes]
            codebook = codebook[selected]

        if codebook.shape[0] <= 1:
            return codebook.new_tensor(0.0)

        codebook_normalized = F.normalize(codebook, p=2, dim=1, eps=1e-8)
        gram_matrix = codebook_normalized @ codebook_normalized.t()

        identity = torch.eye(
            gram_matrix.shape[0],
            device=gram_matrix.device,
            dtype=gram_matrix.dtype,
        )

        return F.mse_loss(gram_matrix, identity)

    def forward(self, z_e: torch.Tensor):
        """
        Quantize encoder output.

        Returns:
            z_q:
                Straight-through quantized latent tensor.
            vq_loss:
                Codebook + commitment + weighted orthogonal loss.
            encoding_indices:
                Selected codebook index per cell.
            loss_dict:
                Detached individual VQ loss components for logging.
        """
        z_e_sq = (z_e ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.embedding.weight ** 2).sum(dim=1)
        cross = z_e @ self.embedding.weight.t()
        distances = z_e_sq + e_sq - 2 * cross

        encoding_indices = distances.argmin(dim=1)
        z_q = self.embedding(encoding_indices)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        orthogonal_loss = self.orthogonal_regularization_loss(
            encoding_indices=encoding_indices
        )

        weighted_orthogonal_loss = self.orthogonal_reg_weight * orthogonal_loss

        vq_loss = (
            codebook_loss
            + self.commitment_cost * commitment_loss
            + weighted_orthogonal_loss
        )

        z_q = z_e + (z_q - z_e).detach()

        loss_dict = {
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "orthogonal_loss": orthogonal_loss.detach(),
            "weighted_orthogonal_loss": weighted_orthogonal_loss.detach(),
            "vq_loss_total": vq_loss.detach(),
        }

        return z_q, vq_loss, encoding_indices, loss_dict


class Encoder(nn.Module):
    """
    Encoder with two hidden layers:
        input_dim -> hidden_dim -> hidden_dim -> latent_dim
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, latent_dim: int = 64):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Decoder with two hidden layers:
        latent_dim -> hidden_dim -> hidden_dim -> output_dim

    Final ReLU keeps MSE reconstructions non-negative, matching log-normalized
    expression matrices.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 512, output_dim: int = None):
        super().__init__()

        if output_dim is None:
            raise ValueError("output_dim must be provided")

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

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.net(z_q)


class VQVAE(nn.Module):
    """
    Full VQ-VAE model:
        Encoder -> VectorQuantizer -> Decoder
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 64,
        num_embeddings: int = 20,
        commitment_cost: float = 0.25,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_max_codes: Optional[int] = None,
        orthogonal_reg_active_codes_only: bool = False,
    ):
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_max_codes=orthogonal_reg_max_codes,
            orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
        )
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices, loss_dict = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, encoding_indices, loss_dict

    def encode(self, x: torch.Tensor):
        z_e = self.encoder(x)
        _, _, encoding_indices, _ = self.vq(z_e)
        return encoding_indices

    def encode_latent(self, x: torch.Tensor):
        """Return continuous encoder latent vectors before quantization."""
        return self.encoder(x)

    def decode(self, indices: torch.Tensor):
        z_q = self.vq.embedding(indices)
        return self.decoder(z_q)
