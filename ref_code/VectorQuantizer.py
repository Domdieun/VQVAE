#!/usr/bin/env python3

"""
Vector Quantization block of a VQ-VAE, where embeddings are snapped to one from a codebook
Based on TF implementation from Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
Pytorch version written by Dick de Ridder, adaptations by Mathijs Balk
"""

import torch
from torch import nn
import torch.nn.functional as F


class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.

    def __init__(self, decay, shape):
        """ 
        decay: 
        shape:
        """

        super().__init__()

        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average


class VectorQuantizer(nn.Module):
    """ module to learn discrete representations based on vector quantization
    """
    # See Section 3 of "Neural Discrete Representation Learning" and:
    # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.
    
    def __init__(self, embedding_dim:int, num_embeddings:int, 
                 use_ema:bool = False, ema_decay:float = 0.99,
                 limit:float= 1e-2, seed:int = None):
        """ module to learn discrete representations based on vector quantization
        embedding_dim: dimensionality in which the datapoints will be quantized
        num_embeddings: size of VQ codebook, nr of representatives/vectors to store
        use_ema: update the codebook with rolling mean instead of dictionary loss
        ema_decay: sensitivity of rolling average to new updates
        limit: boundaries between which centroids can initiate away from the origin
        seed: seed to randomly generate initial positions of codebook vectors
        """
        
        super().__init__()

        # Dictionary embeddings, initialize with small uniform random values
        self.embedding_dim  = embedding_dim 
        self.num_embeddings = num_embeddings

        # Randomly initializing codebook vectors
        self.limit = limit
        self.seed = seed if seed is not None else random.randint(0, 1e-2)
        self.generator = torch.Generator().manual_seed(self.seed)
        vectors = torch.FloatTensor(embedding_dim, num_embeddings)
        vectors.uniform_(-self.limit, self.limit, generator=self.generator)

        # Nature of codebook depends on learning method
        self.epsilon = 1e-5 
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        if not use_ema:
             self.register_parameter("vectors", nn.Parameter(vectors))
        else:
            self.register_buffer("vectors", vectors)
            self.N_i_ts = SonnetExponentialMovingAverage(ema_decay, (num_embeddings,))
            self.m_i_ts = SonnetExponentialMovingAverage(ema_decay, vectors.shape)
            self.v_i_ts = SonnetExponentialMovingAverage(ema_decay, (num_embeddings,))

 
    def update_codebook_ema(self, flat_x:torch.tensor, encoding_indices:torch.tensor):
        """ update rolling averages that determine positioning of the centroids
        flat_x: (batch_size, embedding_dim) input data represented as points in latent space
        encoding_indices:
        """
        with torch.no_grad():
            # See Appendix A.1 of "Neural Discrete Representation Learning".        
            encoding_indices_onehot = torch.eye(self.num_embeddings, device = encoding_indices.device)[encoding_indices.detach()] # made with ChatGPT

            # Updated exponential moving average of the cluster counts.
            # See Equation (6).
            n_i_ts = encoding_indices_onehot.sum(0)
            self.N_i_ts(n_i_ts)

            # Exponential moving average of the embeddings. 
            # See Equation (7).
            embed_sums = flat_x.transpose(0, 1) @ encoding_indices_onehot
            self.m_i_ts(embed_sums)
            
            # for stability, add epsilon
            # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
            N_i_ts_sum = self.N_i_ts.average.sum()
            N_i_ts_stable = ((self.N_i_ts.average + self.epsilon)
                            / (N_i_ts_sum + self.num_embeddings * self.epsilon) 
                            * N_i_ts_sum)
            
            # 
            # See Equation (8).       
            self.vectors = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)


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
    
        if self.use_ema and self.training: # forward without setting to .eval() will update codebook if EMA
            self.update_codebook_ema(flat_x, encoding_indices)
                            
        return (quantized_x, dictionary_loss, commitment_loss, encoding_indices.view(x.shape[0], -1))
