#Under Architecture_2.py, under class VQVAE, def forward function
'''
def encode(self, x: torch.Tensor):
    """
    Convenience method: returns only the discrete codebook indices.
    Useful at inference time to get cell 'barcodes' in latent space.
    """
    z_e = self.encoder(x)
    _, _, encoding_indices = self.vq(z_e)
    return encoding_indices


def decode(self, indices: torch.Tensor):
    """
    Convenience method: reconstruct from codebook indices directly.
    """
    z_q = self.vq.embedding(indices)
    return self.decoder(z_q)

'''

