#!/usr/bin/env python3

"""
Vector Quantization block of a VQ-VAE, where embeddings are snapped to one from a codebook
Based on TF implementation from Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
Pytorch version written by Dick de Ridder, minor adaptations by Mathijs Balk
"""

import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """ module to learn discrete representations based on vector quantization
    """
    # See Section 3 of "Neural Discrete Representation Learning" and:
    # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.
    
    def __init__(self, embedding_dim:int, num_embeddings:int, 
                 limit:float= 1e-2, seed:int = None):
        """ module to learn discrete representations based on vector quantization
        embedding_dim: dimensionality in which the datapoints will be quantized
        limit: boundaries between which centroids can initiate away from the origin
        seed: seed to randomly generate initial positions of codebook vectors
        """
        
        super().__init__()

        # Dictionary embeddings, initialize with small uniform random values
        self.embedding_dim  = embedding_dim 
        self.num_embeddings = num_embeddings

        # Randomly initializing codebook vectors
        self.limit = limit
        self.seed = seed
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)
        vectors = torch.FloatTensor(embedding_dim, num_embeddings)
        vectors.uniform_(-self.limit, self.limit, generator=self.generator)
        self.register_parameter("vectors", nn.Parameter(vectors))


    def forward(self, x:torch.tensor):
        """ for batch of points in latent space, find nearest codebook vector and output its position
        x: batch of data to flatten, quantize to codebook vectors and update codebook based on
        """
        flat_x = x.reshape(-1,self.embedding_dim) # Reshape to batchsize x embedding_dim

        # Calculate Euclidean distances between input and dictionary
        distances = (flat_x**2).sum(1,keepdim=True) - (2 * flat_x @ self.vectors) + (self.vectors**2).sum(0,keepdim=True)

        # For each input x, find closest embedding quantized_x in dictionary
        encoding_indices = distances.argmin(1)
                    
        quantized_x = F.embedding(encoding_indices, self.vectors.transpose(0,1))

        if self.use_ema:          
            dictionary_loss = torch.zeros(1, device = x.device)
        else:
            # See second term of Equation (3): treat encoder output as constant, try to get embeddings close to it
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()

        # See third term of Equation (3): treat embeddings as constant, force encoder to commit to a representation
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()

        # Straight-through gradient: will be calculated on encoder output, not on quantization step. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()
    
        return (quantized_x, dictionary_loss, commitment_loss, encoding_indices.view(x.shape[0], -1))
