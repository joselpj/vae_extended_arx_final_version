import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dim, latent_dim):
        
        super(Decoder, self).__init__()
        self.dec1 = nn.Linear(input_dim+latent_dim, layer_dim)
        self.dec2 = nn.Linear(layer_dim, layer_dim)
        self.out = nn.Linear(layer_dim, output_dim)

    def forward(self, z):
        x = F.elu(self.dec1(z))
        x = F.elu(self.dec2(x))
        return self.out(x)
