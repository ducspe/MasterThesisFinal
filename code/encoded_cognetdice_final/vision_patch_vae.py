import torch
from torch import from_numpy, cuda
import torch.nn as nn
import torch.nn.functional as F


use_gpu = cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()

        self.hidden_in = nn.Linear(input_dim, hidden_dim)
        self.latent_mu = nn.Linear(hidden_dim, z_dim)
        self.latent_var = nn.Linear(hidden_dim, z_dim)
        self.hidden_out = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

        self.to(device).double()

    def load_weights(self, weights_location):
        self.load_state_dict(torch.load(weights_location))

    def encode(self, x):
        hid_in = F.relu(self.hidden_in(x))
        z_mu = self.latent_mu(hid_in)
        z_var = self.latent_var(hid_in)

        return z_mu, z_var

    def decode(self, x):
        hid_out = F.relu(self.hidden_out(x))
        predicted = torch.sigmoid(self.out(hid_out))
        return predicted

    def forward(self, x):
        # Encode
        z_mu, z_var = self.encode(x)

        # Reparametrize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # Decode
        predicted = self.decode(x_sample)
        return predicted, z_mu, z_var

    def encode_numpy(self, x):
        if device == 'cuda':
            x = from_numpy(x).cuda()
        else:
            x = from_numpy(x).cpu()
        x = x.view(-1, 4 * 4)
        z_mu, z_var = self.encode(x)
        return z_mu, z_var