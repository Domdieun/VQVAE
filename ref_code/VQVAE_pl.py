#!/usr/bin/env python3

"""

"""

from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl


class VQVAE_Learner(pl.LightningModule):

    def __init__(self, vqvae, lr, quantize_on, w_commitment):
        """
        vqvae: 
        quantize_on: int; epoch number on which to start vector quantizing
        lr: learning rate to use for training
        w_commitment: float; weight to shrink role of commitment loss in total loss,
                             affecting learning of how to map point to latent space
        """
        super().__init__()

        self.vqvae = vqvae
        self.vqvae.use_quantizer = False
        self.quantize_on = quantize_on
        
        self.lr = lr
        self.w_commitment = w_commitment

  
    def forward(self, x):
        return self.vqvae(x)
    

    def training_step(self, batch, batch_idx):
        """ doc
        """

        self.vqvae.use_quantizer = False if self.current_epoch < self.quantize_on else True
    
        x = batch.to(self.device)
        dec, z_quantized, dictionary_loss, commitment_loss = self(x)
        reconstruction_loss = F.mse_loss(dec, x, reduction = 'mean')
        total_loss = (reconstruction_loss + dictionary_loss + 
                      self.w_commitment * commitment_loss)
        
        self.log_dict({'trn_dictionary_loss': dictionary_loss, 'trn_commitment_loss': commitment_loss,
                      'trn_reconstruction_loss': reconstruction_loss, 'trn_total_loss': total_loss},
                       on_step = False, on_epoch = True, logger = True)

        return total_loss

    
    def validation_step(self, batch, batch_idx):
        """ doc
        """
        
        self.vqvae.use_quantizer = False if self.current_epoch < self.quantize_on else True

        x = batch.to(self.device)
        dec, z_quantized, dictionary_loss, commitment_loss = self(x)
        reconstruction_loss = F.mse_loss(dec, x, reduction = 'mean')
        total_loss = (reconstruction_loss + dictionary_loss + 
                      self.w_commitment * commitment_loss)
                
        self.log_dict({'val_dictionary_loss': dictionary_loss, 'val_commitment_loss': commitment_loss,
                       'val_reconstruction_loss': reconstruction_loss, 'val_total_loss': total_loss},
                       on_step = False, on_epoch = True, logger = True)
        
        return total_loss


    def test_step(self, batch, batch_idx):
        """ doc
        """
        
        x = batch.to(self.device)
        dec, z_quantized, dictionary_loss, commitment_loss = self(x)
        reconstruction_loss = F.mse_loss(dec, x, reduction = 'mean')
        total_loss = (reconstruction_loss + dictionary_loss + 
                      self.w_commitment * commitment_loss)
                
        self.log_dict({'test_dictionary_loss': dictionary_loss, 'test_commitment_loss': commitment_loss,
                       'test_reconstruction_loss': reconstruction_loss, 'test_total_loss': total_loss},
                       on_step = False, on_epoch = True, logger = True)
        
        return total_loss
        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
