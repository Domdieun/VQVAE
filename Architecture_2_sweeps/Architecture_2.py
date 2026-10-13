"""
Architecture_2_1.py

A Vector Quantized Variational Autoencoder (VQ-VAE) for single-cell RNA-seq
using Negative Binomial reconstruction loss.

This is a Negative-Binomial version of Architecture_1.py.

Main differences from the MSE version:
1. The decoder no longer outputs direct reconstructed expression values.
2. The decoder outputs mu, the predicted mean count for each gene.
3. The model also learns theta, the inverse dispersion parameter for each gene.
4. The forward pass returns:
       mu, theta, vq_loss, encoding_indices
   instead of:
       x_recon, vq_loss, encoding_indices

Important:
Negative Binomial loss should be used with raw count data, not log-normalized data.

Training loss should be computed as:
    recon_loss = negative_binomial_loss(x, mu, theta)
    total_loss = recon_loss + vq_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. NEGATIVE BINOMIAL LOSS

def negative_binomial_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Negative Binomial negative log-likelihood loss.

    Args:
        x:
            Observed gene expression raw counts.
            Shape: (batch, n_genes)

        mu:
            Predicted mean count for each gene from the decoder
            Shape: (batch, n_genes)
            Must be positive.

        theta:
            Inverse dispersion parameter.
            Shape can be:
                (n_genes,)
            or:
                (batch, n_genes)
            Must be positive.

        eps:
            Small value for numerical stability.

    Returns:
        Scalar loss value.

    Notes:
        The Negative Binomial distribution is commonly used for overdispersed
        count data, including single-cell RNA-seq raw counts. Unlike in MSE, the goal is not to make the decoder output
        match x directly with squared error. Instead, the decoder predicts the mean of a probability distribution
        : How likely is the observed count x under a negative binomial distribution with mean mu and dispersion theta?

        A simple Poisson model assumes: variance = mean
        in scRNA-seq, usually variance > mean
        The negative binomial handles this by adding a dispersion parameter, theta

        Higher theta means lower dispersion.
        Lower theta means higher dispersion.

        variance = mu + mu^2 / theta

        The code needs to implement the log of this probability: P(x|mu, theta)
        P(x | mu, theta) = Γ(x + theta) / [Γ(theta) Γ(x + 1)] * (theta / (theta + mu))^theta * (mu / (theta + mu))^x

        where:
        x      = observed count
        mu     = predicted mean
        theta  = dispersion
        Γ      = gamma function

        hence: log P(x | mu, theta)

        log(a × b × c) = log(a) + log(b) + log(c)

        So the log-likelihood becomes:
        log P(x | mu, theta) =
        log Γ(x + theta)
        - log Γ(theta)
        - log Γ(x + 1)
        + theta × [log(theta) - log(theta + mu)]
        + x × [log(mu) - log(theta + mu)]

        Higher log-likelihood -> the model predicts a distribution that fits the observed count well
        Lower log-likelihood -> the observed count was unlikely under the model prediction
        we want to maximise log likelihood
    """

    # Ensure numerical stability
    mu = torch.clamp(mu, min=eps) # prevents mu from being zero or negative (since log(0) is undefined), must be positive
    theta = torch.clamp(theta, min=eps) # same thing for theta, makes sure theta > 0

    #calculation of log-likelihood means
    log_likelihood = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    )

    #maximize log-likelihood == minimise negative log-likelihood -> this is why we multiply by -1
    #log_likelihood = high, negative_log_likelihood = low
    #log_likelihood = very negative, negative_log_likelihood = large positive
    negative_log_likelihood = -log_likelihood

    return negative_log_likelihood.mean()


# 2. VECTOR QUANTIZER

class VectorQuantizer(nn.Module):
    """
    One of the VQ-VAE components that maintains a learnable codebook of K
    embedding vectors.

    Each encoder output is snapped to its nearest codebook entry.
    The codebook has shape:
        K x D

    Args:
        num_embeddings:
            K, the number of codebook vectors.

        embedding_dim:
            D, the dimensionality of each codebook vector.
            Must match the encoder output dimension.

        commitment_cost:
            Beta, weight on the commitment loss term.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Common VQ-VAE initialization.
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / num_embeddings,
            1.0 / num_embeddings,
        )

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e:
                Encoder output.
                Shape: (batch, embedding_dim)

        Returns:
            z_q:
                Quantized vectors, same shape as z_e.

            vq_loss:
                Scalar VQ loss.

            encoding_indices:
                Selected codebook index per sample.
                Shape: (batch,)
        """

        z_e_sq = (z_e ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.embedding.weight ** 2).sum(dim=1)
        cross = z_e @ self.embedding.weight.t()

        distances = z_e_sq + e_sq - 2 * cross

        encoding_indices = distances.argmin(dim=1)

        z_q = self.embedding(encoding_indices)

        # Keep these as MSE.
        # These losses are between latent vectors, not gene count distributions.
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, encoding_indices


# 3. ENCODER

class Encoder(nn.Module):
    """
    Maps raw gene-count vectors to a continuous latent space z_e.

    Architecture:
        Linear -> BatchNorm -> ReLU
        Linear -> BatchNorm -> ReLU
        Linear projection

    Args:
        input_dim:
            Number of genes.

        hidden_dim:
            Width of hidden layers.

        latent_dim:
            Output size. Must match VectorQuantizer embedding_dim.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 4. DECODER

class Decoder(nn.Module):
    """
    Maps quantized latent vectors z_q back to Negative Binomial mean parameters.

    Important:
    The decoder output is not a direct reconstruction anymore.
    It is mu, the predicted mean count per gene.

    Since mu must be positive, the final raw decoder output is passed through:
        softplus(raw_mu) + eps

    Args:
        latent_dim:
            Latent/codebook vector dimension.

        hidden_dim:
            Width of hidden layers.

        output_dim:
            Number of genes.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 512,
        output_dim: int = None,
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
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        raw_mu = self.net(z_q)

        # softplus makes mu positive but smoother than ReLU.
        mu = F.softplus(raw_mu) + 1e-4

        return mu


# 5. VQ-VAE

class VQVAE(nn.Module):
    """
    Full VQ-VAE model with Negative Binomial output.

    Flow:
        input x
        -> encoder
        -> continuous latent z_e
        -> vector quantizer
        -> quantized latent z_q
        -> decoder
        -> mu

    The model also learns theta, the inverse dispersion parameter.

    Args:
        input_dim:
            Number of input genes.

        hidden_dim:
            Hidden layer width.

        latent_dim:
            Latent/codebook vector dimension.

        num_embeddings:
            Number of codebook vectors.

        commitment_cost:
            Beta for VQ commitment loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 64,
        num_embeddings: int = 20,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
        )

        # Learn one inverse-dispersion value per gene.
        # We store it in unconstrained log/raw space, then transform with softplus.
        self.log_theta = nn.Parameter(torch.zeros(input_dim))

    def get_theta(self) -> torch.Tensor:
        """
        Return positive inverse-dispersion parameter theta.

        Shape:
            (n_genes,)
        """

        theta = F.softplus(self.log_theta) + 1e-4
        return theta

    def forward(self, x: torch.Tensor):
        """
        Run the full VQ-VAE forward pass.

        Args:
            x:
                Raw count matrix batch.
                Shape: (batch, n_genes)

        Returns:
            mu:
                Predicted mean counts.
                Shape: (batch, n_genes)

            theta:
                Learned inverse dispersion.
                Shape: (n_genes,)

            vq_loss:
                Vector quantization loss.

            encoding_indices:
                Selected codebook index per cell.
                Shape: (batch,)
        """

        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.vq(z_e)

        mu = self.decoder(z_q)
        theta = self.get_theta()

        return mu, theta, vq_loss, encoding_indices

    def encode(self, x: torch.Tensor):
        """
        Map each input sample to its discrete codebook assignment.

        Returns:
            encoding_indices:
                Tensor of shape (batch,) containing one codebook index per cell.
        """

        z_e = self.encoder(x)
        _, _, encoding_indices = self.vq(z_e)

        return encoding_indices

    def decode(self, indices: torch.Tensor):
        """
        Decode directly from codebook indices.

        For the Negative Binomial model, this returns mu rather than x_recon.

        Args:
            indices:
                Tensor of shape (batch,) containing codebook indices.

        Returns:
            mu:
                Predicted mean counts.
                Shape: (batch, n_genes)
        """

        z_q = self.vq.embedding(indices)
        mu = self.decoder(z_q)

        return mu