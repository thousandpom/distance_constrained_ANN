import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Computes the n-dimensional Euclidean distance between neuronal pairs
def compute_distance_matrix(
    input_size: int,
    hidden_size: int,
    output_size: int,
    n_spatial_dims: int,
    norm: int,
) -> torch.tensor:
    """Computes the Euclidean distance between nodes from all layers.

    Args:
        input_size (int): the number of nodes in the input layer
        hidden_size (int): the number of nodes in the hidden layer
        output_size (int): the number of nodes in the output layer
        n_spatial_dims (int, optional): defines the dimension of the point
        clouds in the neural network. Defaults to 3.

    Returns:
    torch.tensor: a distance matrix that contains the element-wise euclidean
    distance between all nodes in the network.
    """

    # Reproducibility
    #     torch.manual_seed(42)

    # Creates a n-dimensional point cloud and assign them with a position.
    n_nodes = input_size + hidden_size + output_size
    node_positions = torch.rand(n_nodes, n_spatial_dims)

    # Stores the computed element-wise distance in a matrix.
    dist = torch.cdist(node_positions, node_positions, p=norm)
    #     print(
    #         f"Nodes positions: {node_positions}, \n"
    #         f"Distance: {dist} \n"
    #         f"Distance shape: {dist.shape}")
    #     print(node_positions)
    return dist, node_positions


class BaselineModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_spatial_dims: int,
        norm: int,
        activation="relu",
    ):
        """A baseline RNN model class to implement n-bit memory task as
        described in Sussillo & Barak, Neural Computation, 2013.

        Args:
            input_size (int): Defines the number of nodes in the input layer,
            which is equal to the number of bits set in the configuration file.
            hidden_size (int): the number of nodes in the hidden layer.
            output_size (int): the number of nodes in the output layer;
            equals to the input_size.
            device (str, optional): Defaults to 'cpu'.
            activation (str, optional): Defines the activation used in the
            hidden layer. Defaults to 'relu'.
        """

        super(BaselineModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_spatial_dims = n_spatial_dims
        self.norm = norm

        # Defines the layers
        self.rnn = nn.RNN(
            input_size, hidden_size, nonlinearity=activation, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        time_steps = x.size(1)
        #         hidden = torch.zeros(batch_size, self.hidden_size).type_as(
        #             x.data
        #         )  # initialize hidden state
        hidden_list = torch.zeros(
            time_steps, batch_size, self.hidden_size
        ).type_as(x.data)
        output_list = torch.zeros(
            time_steps, batch_size, self.output_size
        ).type_as(x.data)

        hidden_list, hidden = self.rnn(x, hidden)
        #         print(output_list.shape)

        output_list = self.output_layer(hidden_list)
        #         print(output_list.shape)
        #         hidden_list = hidden_list.permute(1, 0, 2)
        #         output_list = output_list.permute(1, 0, 2)
        return hidden_list, output_list, hidden


class ConstrainedModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_spatial_dims,
        norm,
        activation="relu",
        dropout=0,
    ):
        """A distance-constrained version of the BaselineModel. All nodes in
        the network are assigned with a n-dimensional location, and the
        Euclidean distance between each neuronal pair is computed and used
        for regularizing the network's training loss.

        Args:
            Refer to the docstring in the BaselineModel class.
        """
        super(ConstrainedModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_spatial_dims = n_spatial_dims
        self.norm = norm

        distance_matrix_origin, nodes_matrix = compute_distance_matrix(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.n_spatial_dims,
            self.norm,
        )
        # Normalize distance matrix
        distance_matrix = distance_matrix_origin / torch.mean(
            distance_matrix_origin
        )

        hidden_pos = nodes_matrix[input_size : input_size + hidden_size]
        self.distance_matrix = nn.Parameter(distance_matrix)
        self.hidden_pos = nn.Parameter(hidden_pos)

        # Defines the layers
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            batch_first=True,
            nonlinearity=activation,
            dropout=dropout,
        )

        # Define the distance parameter for weights in each layer
        self.dist_ih = nn.Parameter(
            self.distance_matrix[
                0:input_size, input_size : input_size + hidden_size
            ].transpose(0, 1)
        )

        self.dist_hh = nn.Parameter(
            self.distance_matrix[
                input_size : input_size + hidden_size,
                input_size : input_size + hidden_size,
            ].transpose(0, 1)
        )

        self.dist_out = nn.Parameter(
            self.distance_matrix[
                input_size : input_size + hidden_size,
                input_size + hidden_size :,
            ].transpose(0, 1)
        )

        #         print(self.dist_ih, self.dist_hh, self.dist_out)

        self.layers_weight = [
            self.rnn.weight_ih_l0,
            self.rnn.weight_hh_l0,
            self.output_layer.weight,
        ]

        self.distances = [self.dist_ih, self.dist_hh, self.dist_out]

        for distance in self.distances:
            distance.requires_grad = False
        self.distance_matrix.requires_grad = False

    # print(f'Input_hidden layer weight shape is:
    # {self.rnn.weight_ih_l0.shape},\t its dist shape is
    # {self.dist_ih.shape}\n', f'Hidden_hidden layer weight shape is:
    # {self.rnn.weight_hh_l0.shape},\t its dist shape is
    # {self.dist_hh.shape}\n', f'Output layer weight shape is:
    # {self.output_layer.weight.shape},\t its dist shape is
    # {self.dist_out.shape}')

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        time_steps = x.size(1)
        #         print(hidden_states)# initialize hidden state
        hidden_list = torch.zeros(
            time_steps, batch_size, self.hidden_size
        ).type_as(x.data)
        output_list = torch.zeros(
            time_steps, batch_size, self.output_size
        ).type_as(x.data)

        hidden_list, hidden = self.rnn(
            x, hidden
        )  # output the state of hidden layers

        output_list = self.output_layer(hidden_list)
        return hidden_list, output_list, hidden

    def compute_weight_regularizer(self, norm: int):
        dist_reg = torch.tensor(0.0, dtype=float)
        for w, dist in zip(self.layers_weight, self.distances):
            dist_reg = dist_reg + (w * dist).norm(p=norm)
        return dist_reg

    def compute_lregularizer(self, norm: int):
        l_reg = torch.tensor(0.0, dtype=float)
        for w in self.layers_weight:
            l_reg = l_reg + w.norm(p=norm)
        return l_reg
