"""
Architecture_1.py

A baseline Vector Quantized Variational Autoencoder (VQ-VAE) for single-cell
RNA sequencing data (PBMC 68k). Designed to plug directly into the
load_data_split_1.py pipeline.

Architecture overview:
Input (n_genes) --> Encoder --> z_e --> VectorQuantizer --> z_q --> Decoder --> Reconstruction
"""

import torch # for multidimensional arrays (tensors) and maths operations
import torch.nn as nn # imports neural network module (layers, loss functions etc.)
import torch.nn.functional as F # imports functional modules (activation functions, loss metrics etc.)



# 1. VECTOR QUANTIZER
class VectorQuantizer(nn.Module):
    """
    One of VQVAE components that maintains a learnable 'codebook' of K embedding vectors.
    Each encoder output is snapped to its nearest codebook entry.
    The vector codebook has a dimension of K x D.

    Args:
        num_embeddings (int): K — the number of codebook vectors (vocabulary size).
        embedding_dim  (int): D — the dimensionality of each codebook vector.
                              Must match the encoder output dimension size in order for it to work.
        commitment_cost (float): Beta — weight on the commitment loss term that
                                 encourages the encoder to stay close to the
                                 chosen codebook vector (for now I use 0.25 as default standards).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__() #calls the constructor of parent class (nn.Module)

        self.num_embeddings  = num_embeddings   # K: codebook size
        self.embedding_dim   = embedding_dim    # D: vector dimension
        self.commitment_cost = commitment_cost  # beta: 0.25

        # self.embedding: the codebook of shape (K, D).
        # Each row represents one discrete "word" (vector) in our latent vocabulary.
        # by default nn.Embeddings draws numbers from a Standard Normal Distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # nn.init.uniform_(x,y,z) : 'uniform_' means 'in-place operation'
            # -> tells matrix to reset the values of x (self.embedding.weight) from normal distribution
            # -> takes tensor x and fill cell with numbers between y and z.
        # Initialise codebook weights uniformly in [-1/K, 1/K] - common in vqvae to prevent codebook collapse
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: Encoder output, shape (batch, embedding_dim) #eg. (256, 64)

        Returns:
            z_q        : Quantized vectors (same shape as z_e), with straight-
                         through gradient trick applied.
            loss       : Scalar VQ loss = codebook loss + commitment loss.
            encoding_indices: The codebook index chosen per sample, shape.
        """

        # Distance computation -> euclidean distance: ||z_e - e_k||^2 for every (output, codebook entry) pair.
        # Expanding: ||z_e||^2 - 2·z_e·e_k^T + ||e_k||^2
        #   z_e_sq (encoder output squared summed across dimensions): (batch, 1)
        #   e_sq (codebook vectors squared summed across dimensions)  : (1, K)
        #   cross (2(encoder output * codebook vectors)) : (batch, K)  via matrix multiply

        z_e_sq = (z_e ** 2).sum(dim=1, keepdim=True)          # (B, 1)
        e_sq   = (self.embedding.weight ** 2).sum(dim=1)       # (K, )
        cross  = z_e @ self.embedding.weight.t() # (B, K), @:matrix multiplication, transposed from shape (K,D)-> (D,K)

        # Result: (B,K) distance matrix where each entry is ||z_e[i] - e_k||^2
        distances = z_e_sq + e_sq - 2 * cross                  # (B, K)

        # Nearest-neighbour lookup - scan horizontally for cells (dim=1)
        # Pick the codebook index with the smallest distance for each sample.
        encoding_indices = distances.argmin(dim=1)             # (B,)

        #z_q: retrieve the corresponding codebook vectors.
        z_q = self.embedding(encoding_indices)                 # (B, D)

        # VQ losses
        # Codebook loss: updates codebook vectors to represent encoder outputs better.
        #   sg(·) = stop_gradient — detach() in PyTorch.
        #   z_e.detach() : encoder output is fixed, not updated, only affecting z_q
        codebook_loss   = F.mse_loss(z_q, z_e.detach())

        # Commitment loss: moves encoder outputs towards chosen codebook vectors.
        #   z_q.detach() : codebook is fixed, not updated, only affecting z_e
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        # Combine the losses together but put commitment_cost (eg.0.25) to slow down the commitment process
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-Through Estimator (STE)
        # The argmin is not differentiable. STE hack: copy gradients from
        # z_q back to z_e by replacing z_q with z_e in the forward pass
        # while keeping z_q's value.
        z_q = z_e + (z_q - z_e).detach()                      # (B, D)

        return z_q, vq_loss, encoding_indices




# 2. ENCODER

class Encoder(nn.Module):
    """
    Maps raw gene-expression vectors to a continuous latent space z_e.

    Architecture: Linear --> BN --> ReLU (×2 layers) --> Linear projection
    Batch Norm is important for scRNA-seq because library-size differences
    cause large scale variation between cells, even after normalization.

    Args:
        input_dim   : Number of genes.
        hidden_dim  : Width of intermediate layers.
        latent_dim  : Output size — must match VectorQuantizer's embedding_dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, latent_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1: full gene space -> hidden
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # Layer 2: hidden -> hidden (adds representational depth)
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # Projection to latent dimension (no activation — raw z_e)
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_genes)  -->  z_e: (batch, latent_dim)
        return self.net(x)


# 3. DECODER

class Decoder(nn.Module):
    """
    Maps quantized latent vectors z_q back to gene-expression space.

    Architecture mirrors the Encoder (symmetric). The final activation is
    ReLU because log-normalised gene expression values are non-negative.

    Args:
        latent_dim  : Must match Encoder's latent_dim / VQ's embedding_dim.
        hidden_dim  : intermediate layers
        output_dim  : Number of genes — same as Encoder's input_dim.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 512, output_dim: int = None):
        super().__init__()

        self.net = nn.Sequential(
            # Expand from latent back to hidden
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            # Project to gene space
            nn.Linear(hidden_dim, output_dim),

            # ReLU: reconstructed expression values cannot be negative.
            nn.ReLU(),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: (batch, latent_dim)  -->  x_recon: (batch, n_genes)
        return self.net(z_q)



# 4. VQ-VAE
class VQVAE(nn.Module):
    """
    Full VQ-VAE model: Encoder -> VectorQuantizer -> Decoder.

    This class glues together the three major parts of the architecture:

    1. Encoder
       Maps the original high-dimensional input x into a lower-dimensional
       continuous latent representation z_e.

    2. VectorQuantizer
       Compares each encoder output z_e to all codebook vectors and replaces it
       with the nearest codebook entry. This produces:
       - z_q: the quantized latent vector
       - vq_loss: codebook + commitment loss
       - encoding_indices: the discrete codebook ID chosen for each sample

    3. Decoder
       Reconstructs the original gene-expression vector from the quantized
       latent representation z_q.

    So the overall information flow is:
        input x -> encoder -> continuous latent z_e -> quantizer ->
        discrete/quantized latent z_q -> decoder -> reconstruction x_recon

    This is useful because the model learns a compressed latent space while
    forcing that latent space to use a finite set of learned codebook vectors.
    That gives you both:
    - continuous neural feature extraction from the encoder
    - discrete latent assignments from the codebook

    Args:
        input_dim       : Number of input genes.
        hidden_dim      : Hidden layer width for encoder/decoder (default 512).
        latent_dim      : Latent / codebook vector dimension (default 64).
        num_embeddings  : Codebook size K (default 20).
        commitment_cost : Beta for VQ commitment loss (default 0.25).
    """

    def __init__(
        self,
        input_dim:       int,
        hidden_dim:      int   = 512,
        latent_dim:      int   = 64,
        num_embeddings:  int   = 20,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.encoder    = Encoder(input_dim, hidden_dim, latent_dim)
        self.vq         = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder    = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        """
        Run the full VQ-VAE forward pass on a batch of inputs.

        Step-by-step:
        1. `self.encoder(x)` computes a continuous latent vector z_e for each
           sample in the batch.
        2. `self.vq(z_e)` replaces each z_e with the nearest codebook vector,
           giving a quantized latent vector z_q and the associated VQ loss.
        3. `self.decoder(z_q)` reconstructs the original input from the
           quantized latent representation.

        Returns:
            x_recon         : Reconstructed gene expression, shape (B, n_genes).
            vq_loss         : Scalar vector-quantization loss. This combines:
                              - codebook loss: updates the codebook vectors
                              - commitment loss: encourages encoder outputs to
                                stay close to the selected codebook entries
            encoding_indices: Integer codebook index selected for each sample,
                              shape (B,). This is the model's discrete latent
                              assignment per input cell.
        """
        z_e = self.encoder(x)                           # continuous latent representation
        z_q, vq_loss, encoding_indices = self.vq(z_e)  # nearest codebook vectors + VQ loss
        x_recon = self.decoder(z_q)                    # reconstruction from quantized latent
        return x_recon, vq_loss, encoding_indices

    def encode(self, x: torch.Tensor):
        """
        Map each input sample to its discrete codebook assignment.

        This is an inference helper for when you do not need the reconstruction
        and only want to know which codebook entry each sample uses.

        What it does:
        1. Passes the input through the encoder to produce z_e.
        2. Finds the nearest codebook vector for each z_e.
        3. Returns only the selected codebook indices.

        What it does NOT return:
        - not the reconstructed gene-expression vector
        - not the continuous encoder output z_e
        - not the quantized latent vector z_q

        Practical interpretation:
        If the codebook has K entries, each input cell is assigned one integer
        in the range [0, K-1]. You can think of this as the cell's discrete
        latent label or codebook cluster.

        This is useful for:
        - checking how often each codebook entry is used
        - exporting one discrete cluster/code per cell
        - downstream analyses that compare learned latent groups to annotations

        Returns:
            encoding_indices: Tensor of shape (B,) containing one codebook index
                              per input sample.
        """
        z_e = self.encoder(x)
        _, _, encoding_indices = self.vq(z_e)
        return encoding_indices

    def decode(self, indices: torch.Tensor):
        """
        Reconstruct gene-expression vectors directly from codebook indices.

        This method skips the encoder entirely. Instead of starting from an
        input x, it starts from already known discrete codebook IDs.

        What it does:
        1. Looks up the embedding vector associated with each codebook index.
        2. Treats those embedding vectors as quantized latent vectors z_q.
        3. Passes them through the decoder to produce reconstructed
           gene-expression profiles.

        This is useful when you want to ask:
        "What does codebook entry 7 decode to in input space?"
        or
        "What reconstruction do I get if I feed these discrete latent indices
        into the decoder?"

        Important detail:
        `decode()` reconstructs from the codebook embeddings themselves, not
        from the original continuous encoder outputs. So this reflects the
        decoder's view of the discrete latent state after quantization.

        Args:
            indices: Tensor of shape (B,) containing integer codebook indices.

        Returns:
            Reconstructed gene-expression tensor of shape (B, n_genes).
        """
        z_q = self.vq.embedding(indices)  # lookup the codebook vector for each index
        return self.decoder(z_q)
