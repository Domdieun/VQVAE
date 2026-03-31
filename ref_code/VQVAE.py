#!/usr/bin/env python3

"""
Autoencoder with vector quantization in latent space and 2 hidden layers before latent
Based on TF implementation from Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
Pytorch version written by Dick de Ridder
Last adaptations by Mathijs Balk (Lightning & logging)
"""

import torch
from torch import nn
from VQVAE.VectorQuantizer import VectorQuantizer

# Based on Van den Oord et al., Neural Discrete Representation Learning


class VQVAE(nn.Module):
    def __init__(self, encoder:nn.Module, decoder:nn.Module, 
                 vectorquantizer:VectorQuantizer, use_quantizer = True):
        """ initiates a VQ-VAE of MLPs w/ 2 hidden layers in the encoder and decoder
        encoder: NN to learn represenation of data in latent space with
        vectorquantizer: module to learn discrete representations in the latent space
        decoder: NN to reconstruct input from positions of nearest codebook vectors
        use_quantizer: whether or not to use VQ before passing on to the decoder
        """
        super().__init__()

        self.use_quantizer = use_quantizer

        self.encoder = encoder
        self.vq = vectorquantizer
        self.decoder = decoder
    
    def quantize(self, enc:torch.tensor):
        """ replaces point with nearest representative in codebook if use_quantizer
        enc: input mapped to the embedding space, low-dimensional representation
        returns: z_quantized, discrete representation to map back to input space w/ the decoder
                 dictionary loss, eucl dist to the nearest representative in codebook
                 commitment loss, same as dict loss but to update encoder instead of codebook
                 encoding indices, for all datapoints in batch, idx in codebook they quantized to
        """
        if self.use_quantizer:
            (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.vq(enc)
        else:
            z_quantized      = enc
            dictionary_loss  = torch.zeros(1, device = enc.device)
            commitment_loss  = torch.zeros(1, device = enc.device)
            encoding_indices = []
        return (z_quantized, dictionary_loss, commitment_loss, encoding_indices)

    
    def forward(self, x:torch.tensor):
        """ steps in a forward pass: map to latent space, quantize (if use_quantizer), reconstruct
        x: batch of input data, should be compatible with encoder input / decoder output shape
        returns: dec, input reconstruction from representations in latent space
                 z_quantized, representations in latent space, discrete if use_quantizer
                 dictionary loss, eucl dist to the nearest representative in codebook
                 commitment loss, same as dict loss but to update encoder instead of codebook
        """
        enc = self.encoder(x) # note enc and z
        
        # Quantize output of encoder, if needed; otherwise uses the encoder output
        z_quantized, dictionary_loss, commitment_loss, _ = self.quantize(enc) #note _
        dec = self.decoder(z_quantized)
        
        return dec, z_quantized, dictionary_loss, commitment_loss
