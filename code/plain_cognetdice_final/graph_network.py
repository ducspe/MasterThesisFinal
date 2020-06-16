import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, from_numpy
from torch.utils.data import Dataset, DataLoader


device = 'cuda' if cuda.is_available() else 'cpu'

########################################################################################################################
# To be able to load the accumulated training data:


class GraphDataset(Dataset):
    """The dataset consists of input x_data for the location of the agent, and y_data for the next node"""
    def __init__(self, location_data, next_node):
        pre_x_data = [np.array([x]) for x in location_data]
        pre_x_data = np.array(pre_x_data)
        self.x_data = from_numpy(pre_x_data)
        self.y_data = from_numpy(next_node)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y_data)

########################################################################################################################
# Define the learning model


class GNet(nn.Module):
    def __init__(self, nr_hidden_nodes, nr_output_nodes):
        super().__init__()
        self.num_outputs = nr_output_nodes
        # Define the layers:
        self.l1 = nn.Linear(1, nr_hidden_nodes)  # The x coordinate along the moving axis gets mapped to a higher dimensional space
        self.l2 = nn.Linear(nr_hidden_nodes, nr_output_nodes)

        # Set the training device to 'cuda' if available / 'cpu' otherwise
        print("The model will be trained on device: ", device)
        self.to(device)


    def forward(self, x):
        x = x.to(device).float()
        h1 = F.relu(self.l1(x))
        out = F.softmax(self.l2(h1), dim=1)
        return out

    def classify(self, x):
        x_arr = x * np.ones(shape=(1, 1), dtype=float)
        x_net = from_numpy(x_arr)
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


def show_node_input_response(joint_policy, observation_interval, agent, node, num_divisions):

    network_model = joint_policy.local_policies[agent].nodes[node].classifier

    observation_samples = np.linspace(observation_interval[0], observation_interval[1], num_divisions)
    print("Sweep observations: ", observation_samples)

    next_nodes_likelihood = -1*np.ones(shape=(num_divisions, network_model.num_outputs), dtype=float)
    for index, obs in enumerate(observation_samples):
        next_nodes_likelihood[index] = network_model.classify(obs)

    print(f"Agent {agent}, node {node} has the following next nodes: {next_nodes_likelihood}")
    return observation_samples, next_nodes_likelihood




























