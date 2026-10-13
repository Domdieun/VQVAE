"""
Architecture_5.py

A scVI-inspired Negative-Binomial / Zero-Inflated Negative-Binomial VQ-VAE
for single-cell RNA-seq that explicitly accounts for library size.
https://github.com/romain-lopez/scVI-reproducibility

The model keeps the usual VQ-VAE structure:

    raw counts x
        -> encoder
        -> continuous latent z_e
        -> vector quantizer / codebook
        -> quantized latent z_q
        -> decoder
        -> likelihood parameters
        -> NB or ZINB reconstruction loss

library size adjust
1. Decoder produces raw gene scores.
2. Softmax turns scores into gene proportions.
3. Library size multiplies proportions into expected counts.
4. NB/ZINB loss compares expected counts to real raw counts.
5. Backpropagation adjusts decoder weights.
6. Over training, genes that are highly expressed in a cell type get larger proportions for those cells.

The important scVI-inspired change is in the decoder.

Older/direct NB decoder idea:

    z_q -> decoder -> mu

That means the decoder directly predicts the Negative-Binomial mean count for
all genes. In that setup, the latent representation may need to encode both:
    1. biological composition / relative gene-expression pattern
    2. technical total count depth / library size

This version separates those two things:
    z_q -> decoder -> px_scale
    mu = library_size * px_scale

where:
    px_scale:
        A gene-proportion vector for each cell. It sums to 1 across genes.
    library_size:
        The observed total count depth of each cell, usually x.sum(dim=1).
        library_size = total raw counts in that cell
    mu:
        The expected raw count for each gene under the NB/ZINB likelihood.

So the decoder learns the relative expression composition, while the observed
library size supplies the total count scale.

    library_size = x.sum(dim=1)

This only makes sense if x contains raw counts. If x contains log1p-normalized
values, then x.sum(dim=1) is not the true library size.

The encoder does apply log1p(x) internally for numerical stability, but the
reconstruction likelihood is still computed against the raw counts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

'''

In single-cell RNA-seq, each entry x[cell, gene] is a count. Counts are non-negative integers and are often overdispersed: 
their variance is larger than their mean. A Poisson likelihood usually cannot model this well.

The Negative Binomial distribution is commonly used because it has both:
     mu:    the mean count
     theta: inverse dispersion

Lower theta means more dispersion / noisier counts.
Higher theta makes the distribution closer to Poisson-like behaviour.

This model can use either:
     gene_likelihood = "nb"
     Standard Negative Binomial likelihood.

     gene_likelihood = "zinb"
       Zero-Inflated Negative Binomial likelihood, where zeros can come either
       from the NB distribution itself or from an additional dropout/extra-zero
       process.
In both cases, the loss is a negative log-likelihood. Lower is better.
'''

def nb_log_prob(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the elementwise Negative Binomial log-probability.

    This function returns one log-likelihood value for every cell-gene entry.
    It does NOT reduce/average the result. The reduction happens later in
    negative_binomial_loss().

    Parameterisation used here:
        mean = mu
        inverse dispersion = theta
    Shapes:
        x:
            Raw count matrix with shape:
                (batch_size, n_genes)

        mu:
            Predicted NB mean with shape:
                (batch_size, n_genes)

            In this architecture, mu is not predicted directly. It is computed as:
                mu = library_size * px_scale

        theta:
            Inverse-dispersion parameter. Usually shape:
                (n_genes,)

            PyTorch broadcasting then applies the same gene-wise theta to every
            cell in the batch. It could also be shape (batch_size, n_genes) if
            you later changed the model to predict cell-gene-specific theta.

    Returns:
        log_likelihood:
            Tensor of shape (batch_size, n_genes).
    """

    # Counts should not be negative.
    x = torch.clamp(x, min=0.0)

    # mu and theta must be strictly positive. The log-probability contains log(mu),
    # log(theta), and log(theta + mu), so zeros would produce -inf/NaN values.
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)

    # Negative Binomial log-probability under the mean / inverse-dispersion
    # parameterisation.

    # For each cell-gene count x_ij:
    #
    #   log p(x_ij | mu_ij, theta_j)
    #
    # The lgamma terms are the combinatorial/count part of the NB probability.
    # The theta and x terms describe the probability mass controlled by mu/theta.
    log_likelihood = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    )

    return log_likelihood


def negative_binomial_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mean Negative Binomial negative log-likelihood.

    nb_log_prob(...) gives log p(x | mu, theta). Training normally minimizes a
    loss, so we take the negative log-likelihood:

        loss = -log p(x | mu, theta)

    The .mean() averages over all cells and genes in the batch.
    """

    return -nb_log_prob(x=x, mu=mu, theta=theta, eps=eps).mean()


def zero_inflated_negative_binomial_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    zi_logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mean Zero-Inflated Negative Binomial negative log-likelihood.

    ZINB extends NB with an additional probability of extra zeros.

    zi_logits:
        Raw logits for the extra-zero/dropout probability.
        If:
            pi = sigmoid(zi_logits)
        then pi is the model's extra-zero probability for each cell-gene entry.

    For x == 0:
        There are two possible explanations:
            1. The value is an extra zero with probability pi.
            2. The value came from the NB distribution and happened to be zero.

        So:
            p(x = 0) = pi + (1 - pi) * NB(0 | mu, theta)

    For x > 0:
        The value cannot have come from the extra-zero component, so:
            p(x > 0) = (1 - pi) * NB(x | mu, theta)

    This function computes the log-probability in a numerically stable way.
    """

    # First compute the NB log-probability for every entry, whether zero or not.
    nb_case = nb_log_prob(x=x, mu=mu, theta=theta, eps=eps)

    # Stable versions of:

    #   log_pi           = log(sigmoid(zi_logits))
    #   log_one_minus_pi = log(1 - sigmoid(zi_logits))

    # Directly computing sigmoid then log can be unstable for large positive or
    # negative logits. softplus gives a stable equivalent expression.
    log_pi = -F.softplus(-zi_logits)
    log_one_minus_pi = -F.softplus(zi_logits)

    # For zero observations, compute:

    # log[ pi + (1 - pi) * NB(0 | mu, theta) ]

    # In log-space, this is:

    # logsumexp( log_pi, log_one_minus_pi + log_NB_zero )
    # logsumexp is used for numerical stability.
    zero_case = torch.logsumexp(
        torch.stack(
            [
                log_pi,
                log_one_minus_pi + nb_case,
            ],
            dim=0,
        ),
        dim=0,
    )

    # For non-zero observations, the extra-zero component cannot explain the
    # count. The probability is therefore:
    #
    # (1 - pi) * NB(x | mu, theta)
    #
    # and in log-space:
    #
    #   log(1 - pi) + log NB(x | mu, theta)
    non_zero_case = log_one_minus_pi + nb_case

    # Use the zero formula for x == 0 and the non-zero formula for x > 0.
    # Because x may be stored as float tensors, x < eps is used instead of x == 0.
    log_likelihood = torch.where(x < eps, zero_case, non_zero_case)

    # Return mean negative log-likelihood over cells and genes.
    return -log_likelihood.mean()


def reconstruction_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    zi_logits: torch.Tensor = None,
    gene_likelihood: str = "nb",
) -> torch.Tensor:

    gene_likelihood = gene_likelihood.lower()

    if gene_likelihood == "nb":
        return negative_binomial_loss(x=x, mu=mu, theta=theta)

    if gene_likelihood == "zinb":
        if zi_logits is None:
            raise ValueError("zi_logits must be provided when gene_likelihood='zinb'.")
        return zero_inflated_negative_binomial_loss(
            x=x,
            mu=mu,
            theta=theta,
            zi_logits=zi_logits,
        )

    raise ValueError(f"Unknown gene_likelihood: {gene_likelihood}")


class VectorQuantizer(nn.Module):
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

        # Small uniform initialization, as commonly used for VQ-VAE codebooks.
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / num_embeddings,
            1.0 / num_embeddings,
        )

    def forward(self, z_e: torch.Tensor):

        z_e_sq = (z_e ** 2).sum(dim=1, keepdim=True)

        e_sq = (self.embedding.weight ** 2).sum(dim=1)

        cross = z_e @ self.embedding.weight.t()

        distances = z_e_sq + e_sq - 2 * cross

        encoding_indices = distances.argmin(dim=1)

        z_q = self.embedding(encoding_indices)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, encoding_indices

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # Dropout regularization. During training, randomly zeroes some
            # hidden units to reduce overfitting.
            nn.Dropout(dropout_rate),

            # Final projection into latent space.
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(torch.clamp(x, min=0.0))

        # Output is the continuous latent representation z_e.
        return self.net(x)

# This architecture:
#
#     z_q -> decoder -> px_scale_logits
#     px_scale = softmax(px_scale_logits)
#     mu = library_size * px_scale
#
# Because softmax makes each px_scale row sum to 1, px_scale is a composition or
# probability vector over genes.
#
# Therefore:
#
#     mu[cell, :].sum() = library_size[cell]
#
# up to numerical precision.
#
# This means the model is asked to learn how the observed total count depth should
# be distributed across genes, rather than learning total count depth from z_q.


class DecoderSCVILibrarySize(nn.Module):
    """
    Decoder that separates expression composition from library size.

    It predicts:

        px_scale:
            Gene proportions. For each cell, this is a probability vector over
            genes and sums to 1.

        zi_logits:
            Dropout / extra-zero logits for ZINB. These are only required when
            using gene_likelihood="zinb".

    The NB/ZINB mean is
        mu = observed_library_size * px_scale

    Shapes:
        z_q:
            (batch_size, latent_dim)

        library_size:
            (batch_size,)

        px_scale:
            (batch_size, n_genes)

        mu:
            (batch_size, n_genes)

        zi_logits:
            (batch_size, n_genes)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        output_dim: int = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if output_dim is None:
            raise ValueError("output_dim must be set to the number of genes.")

        # Shared decoder trunk.
        # This maps the quantized latent vector z_q into a hidden representation h.
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Gene-composition head.
        # Produces one unnormalized logit per gene.
        # Softmax is applied in forward() to convert logits to proportions.
        self.px_scale_decoder = nn.Linear(hidden_dim, output_dim)

        # Zero-inflation/dropout head.
        # Produces one extra-zero logit per gene.
        # Used only for ZINB loss; for NB loss it is computed but ignored.
        self.dropout_decoder = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z_q: torch.Tensor,
        library_size: torch.Tensor,
    ):
        # Decode quantized latent vector into hidden representation.
        h = self.shared(z_q)

        # Predict unnormalized gene-composition logits.
        px_scale_logits = self.px_scale_decoder(h)

        # Convert logits to proportions across genes.
        # dim=1 means the softmax is applied across the gene dimension.
        # For each cell:
        #
        #     px_scale[cell, :].sum() == 1
        #
        # This is what makes px_scale a relative expression composition.
        px_scale = F.softmax(px_scale_logits, dim=1)

        # Library size should be positive.
        # If a cell somehow has zero total counts, clamp it to 1 to prevent a
        # zero mean vector mu. A zero mu would break log(mu) in the NB likelihood.
        library_size = torch.clamp(library_size, min=1.0)

        # Convert gene proportions into expected raw counts.

        # library_size shape before unsqueeze:(batch_size,)

        # library_size.unsqueeze(1) shape: (batch_size, 1)

        # px_scale shape: (batch_size, n_genes)
        # Broadcasting gives: mu[cell, gene] = library_size[cell] * px_scale[cell, gene]
        # Therefore each cell's predicted mean counts sum to its library size.
        mu = library_size.unsqueeze(1) * px_scale

        # Predict ZINB extra-zero logits.
        # These are used by zero_inflated_negative_binomial_loss().
        zi_logits = self.dropout_decoder(h)

        return mu, px_scale, zi_logits


class VQVAE(nn.Module):
    """
    scVI-inspired VQ-VAE for raw-count scRNA-seq data.

    Full flow:

        raw counts x
            -> infer library_size = x.sum(dim=1), unless provided separately
            -> log1p(x) inside encoder
            -> z_e
            -> vector quantizer
            -> z_q + encoding_indices
            -> decoder predicts px_scale
            -> mu = library_size * px_scale
            -> theta = softplus(log_theta)
            -> NB/ZINB likelihood in training script

    The model learns one inverse-dispersion theta per gene:

        theta.shape = (n_genes,)

    This is a common/simple setting: each gene has its own dispersion, shared
    across cells.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 16,
        num_embeddings: int = 32,
        commitment_cost: float = 0.25,
        dropout_rate: float = 0.1,
        gene_likelihood: str = "nb",
    ):
        super().__init__()

        # Store/check which likelihood the model is intended to use.
        # The actual loss choice is handled by reconstruction_loss(), but keeping
        # this attribute is useful for configs/checkpoints.
        gene_likelihood = gene_likelihood.lower()
        if gene_likelihood not in {"nb", "zinb"}:
            raise ValueError("gene_likelihood must be 'nb' or 'zinb'.")

        self.input_dim = input_dim
        self.gene_likelihood = gene_likelihood

        # Encoder: raw counts -> log1p(raw counts) -> continuous latent z_e.
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )

        # Vector quantizer: continuous latent z_e -> nearest codebook vector z_q.
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder: quantized latent z_q + library_size -> mu, px_scale, zi_logits.
        self.decoder = DecoderSCVILibrarySize(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            dropout_rate=dropout_rate,
        )

        # Learn one inverse-dispersion parameter per gene.
        #
        # We store log_theta as an unconstrained parameter because neural network
        # parameters can be any real number. Then get_theta() transforms it into a
        # positive value using softplus.
        #
        # Initial log_theta = 0 means:
        #
        #     theta = softplus(0) + 1e-4 ~= 0.6932
        self.log_theta = nn.Parameter(torch.zeros(input_dim))

    def get_theta(self) -> torch.Tensor:
        """
        Return positive gene-wise inverse-dispersion parameters.

        softplus(x) = log(1 + exp(x))

        softplus is used because theta must be positive for the NB distribution.
        The small +1e-4 prevents theta from becoming extremely close to zero.
        """

        return F.softplus(self.log_theta) + 1e-4

    def infer_library_size(self, x: torch.Tensor) -> torch.Tensor:
        """
        Infer observed library size from raw counts.

        For each cell:

            library_size[cell] = sum of raw counts across genes

        Shape:
            x:
                (batch_size, n_genes)

            library_size:
                (batch_size,)

        This function assumes x is raw counts. If x is log-normalized or otherwise
        normalized, this inferred library size is not meaningful.
        """

        library_size = x.sum(dim=1)
        return torch.clamp(library_size, min=1.0)

    def forward(
        self,
        x: torch.Tensor,
        library_size: torch.Tensor = None,
    ):

        # infer library size from x.
        # This is correct only when x contains raw counts.
        if library_size is None:
            library_size = self.infer_library_size(x)

        z_e = self.encoder(x)

        z_q, vq_loss, encoding_indices = self.vq(z_e)

        # Decode quantized latent vectors into:
        #     px_scale:  relative gene-expression proportions
        #     mu:        expected raw counts after library-size scaling
        #     zi_logits: ZINB extra-zero logits
        mu, px_scale, zi_logits = self.decoder(
            z_q=z_q,
            library_size=library_size,
        )

        # Positive gene-wise inverse-dispersion parameter.
        theta = self.get_theta()

        return {
            "mu": mu,
            "theta": theta,
            "px_scale": px_scale,
            "zi_logits": zi_logits,
            "vq_loss": vq_loss,
            "encoding_indices": encoding_indices,
            "z_e": z_e,
            "z_q": z_q,
        }

    def encode(self, x: torch.Tensor):

        z_e = self.encoder(x)
        _, _, encoding_indices = self.vq(z_e)
        return encoding_indices

    def encode_latent(self, x: torch.Tensor):
        """
        Return continuous and quantized latent representations.

            z_e:
                Continuous encoder output.
            z_q:
                Quantized codebook vector after nearest-neighbour lookup.
            encoding_indices:
                Codebook index for each cell.

        This is useful for UMAP/t-SNE visualization or checking whether z_e and
        z_q contain meaningful biological structure.
        """

        z_e = self.encoder(x)
        z_q, _, encoding_indices = self.vq(z_e)
        return z_e, z_q, encoding_indices

    def decode(
        self,
        indices: torch.Tensor,
        library_size: torch.Tensor,
    ):

        # Look up the codebook vectors corresponding to the requested indices.
        z_q = self.vq.embedding(indices)

        # Decode into proportions and library-size-scaled means.
        mu, px_scale, zi_logits = self.decoder(
            z_q=z_q,
            library_size=library_size,
        )

        theta = self.get_theta()

        return {
            "mu": mu,
            "theta": theta,
            "px_scale": px_scale,
            "zi_logits": zi_logits,
        }
