import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, from_numpy
from torch.utils.data import Dataset, DataLoader


device = 'cuda' if cuda.is_available() else 'cpu'

########################################################################################################################

class GraphDataset(Dataset):
    """The dataset consists of input x_data for the visual sense of the agent, and y_data for the next node in the controller graph"""
    def __init__(self, twod_data, next_node):
        self.x_data = from_numpy(twod_data)
        self.y_data = from_numpy(next_node)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y_data)

########################################################################################################################
# Define the learning model


class TwodGNet(nn.Module):
    def __init__(self, nr_hidden_nodes, nr_output_nodes, image_height, image_width):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width

        # Define the layers:
        self.hidden_layer = nn.Linear(image_height * image_width, nr_hidden_nodes)  # This layer accepts the flattened image tensor.
        self.out_layer = nn.Linear(nr_hidden_nodes, nr_output_nodes)

        # Set the training device to 'cuda' if available / 'cpu' otherwise
        print("The image model will be trained on device: ", device)
        self.to(device).double()

    def forward(self, x):
        x = x.to(device).double()
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden_layer(x))
        out = F.softmax(self.out_layer(x), dim=1)
        return out

    def classify(self, x):
        x_reshaped = np.zeros(shape=(1, x.shape[0], x.shape[1]))  # add one channel to signify we just have a gray scale image and not an RGB image
        x_reshaped[0] = x
        x_net = from_numpy(x_reshaped)  # Create the tensor
        out = self.forward(x_net)
        return out.detach().cpu().numpy()[0]

########################################################################################################################


def train(x_input, y_output, num_epochs, batch_size, network_model):
    # Use cross-entropy loss for multi-class classification:
    criterion = nn.CrossEntropyLoss()
    # Use stochastic gradient descent:
    optimizer = optim.SGD(network_model.parameters(), lr=0.01, momentum=0.5)

    # Load the training data:
    dataset = GraphDataset(np.array(x_input), np.array(y_output))
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(1, num_epochs + 1):
        for batch_count, (input_data, next_node_target) in enumerate(train_loader, 1):
            input_x, label_y = input_data.to(device), next_node_target.to(device)

            # Compute the predicted output by performing a forward pass through the network:
            y_hat = network_model(input_x)

            # Compute the loss:
            loss = criterion(y_hat, label_y)

            # Back-propagation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def save_node_model(network_model, filepath):
    torch.save(network_model.state_dict(), filepath)