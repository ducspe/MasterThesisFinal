# Third party libraries:
import torch
from torch import from_numpy, cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# My libraries:
import training_data_generator as gnt


use_gpu = cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'


class SourceImagesDataset(Dataset):

    def __init__(self, filename):
        vision_patch_arrays = np.load(f'datafolder/{filename}')
        self.len = vision_patch_arrays.shape[0]
        self.x_data = vision_patch_arrays

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()

        self.hidden_in = nn.Linear(input_dim, hidden_dim)
        self.latent_mu = nn.Linear(hidden_dim, z_dim)
        self.latent_var = nn.Linear(hidden_dim, z_dim)
        self.hidden_out = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

        self.to(device).double()

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


def train(model, train_iterator):
    model.train()
    train_loss = 0

    for data in train_iterator:
        data = data.view(-1, 4 * 4)
        data = data.to(device).double()

        optimizer.zero_grad()
        x_sample, z_mu, z_var = model(data)

        reconstruction_loss = F.binary_cross_entropy(x_sample, data, reduction='sum')
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
        total_loss = reconstruction_loss + kl_loss

        total_loss.backward()
        train_loss += total_loss.item()

        optimizer.step()

    return train_loss


def find_latent_variable_range(model, all_patches):
    z_mu_list = []
    z_var_list = []
    for data in all_patches:
        data = data.view(-1, 4 * 4)
        data = data.to(device).double()

        _, z_mu, z_var = model(data)
        z_mu_list.append(z_mu.detach().cpu().numpy()[0][0])
        z_var_list.append(z_var.detach().cpu().numpy()[0][0])

    print("List of latent means: ", z_mu_list)
    print("Min of z_mu_list: ", min(z_mu_list))
    print("Max of z_mu list: ", max(z_mu_list))
    print("List of latent variances: ", z_var_list)
    print("Min of z_var_list: ", min(z_var_list))
    print("Max of z_var_list: ", max(z_var_list))


def perform_training(model, nr_epochs):
    for epoch in range(nr_epochs):
        train_loss = train(model, train_loader)
        train_loss /= len(train_dataset)

        print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}')

    torch.save(model.state_dict(), "./datafolder/vision_patch_vae_weights")


INPUT_DIM = 4 * 4
HIDDEN_DIM = 8
LATENT_DIM = 1

train_dataset = SourceImagesDataset("vision_patch_train_dataset.npy")
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)

model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters())

# Training:
# perform_training(model, nr_epochs=100)

#######################################################################################################################
obs_vae = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device).double()
obs_vae.load_state_dict(torch.load("./datafolder/vision_patch_vae_weights"))

# Sample an observation:
scene, patch = gnt.generate_test_scene_and_patch(off_x=0, off_y=3)
test_tensor = from_numpy(patch).to(device)
print("The original test patch tensor: ", test_tensor)
test_tensor_numpy = test_tensor.cpu().detach().numpy()
test_image = test_tensor_numpy[0][0][:, :]

# Reconstruction:
reconstructed_tensor, z_mu, z_var = obs_vae(test_tensor.view(-1, 4 * 4))
print("Reconstructed test patch tensor: ", reconstructed_tensor.view(4, 4))
print(f"z_mu {z_mu}")
print(f"z_var {z_var}")
reconstructed_img = reconstructed_tensor.view(4, 4).cpu().detach()

# Plot results:
fig1 = plt.figure()
plt.title("Original test image")
plt.imshow(test_image, cmap="coolwarm")

fig2 = plt.figure()
plt.title("Reconstructed test image")
plt.imshow(reconstructed_img, cmap="coolwarm")

plt.show()

# Find the latent variable range:
all_patches_dataset = SourceImagesDataset("vision_patch_train_dataset.npy")
all_patches_from_scene = DataLoader(dataset=all_patches_dataset, batch_size=1, shuffle=False, num_workers=0)
find_latent_variable_range(obs_vae, all_patches_from_scene)
