# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random

import utils

# Compute the euclidean distance between neuronal pairs


def compute_distance_matrix(
        input_size: int, hidden_size: int, output_size: int,
        n_spatial_dims: int, norm: int):

    # Create an n-dimensional point cloud matching the total number of nodes of
    # the neural net
    n_nodes = input_size + hidden_size + output_size
    node_positions = torch.rand(n_nodes, n_spatial_dims)

    # Compute distance matrix and store the euclidean distance in an
    # uppertriangular matrix
    dist = torch.triu(
        torch.cdist(node_positions, node_positions, p=norm), diagonal=1
    )
    # print(
    #     f"Nodes positions: {node_positions}, \n"
    #     f"Distance: {dist} \n"
    #     f"Distance shape: {dist.shape}"
    # )
    return dist, node_positions


# Fully connected (fc) neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_spatial_dims: int,
        norm: int
    ):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_spatial_dims = n_spatial_dims
        self.norm = norm

        # Hidden layers with distance assigned
        distance_matrix_origin, nodes_pos = compute_distance_matrix(
            self.input_size, self.hidden_size *
            2, self.output_size, self.n_spatial_dims, self.norm
        )
        # Normalize distance matrix to 1
        distance_matrix = distance_matrix_origin / torch.mean(
            distance_matrix_origin
        )
        self.distance_matrix = nn.Parameter(distance_matrix)

        # Define model layers and their corresponding distance matrix
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dist_fc1 = nn.Parameter(
            self.distance_matrix[
                0:input_size, input_size: input_size + hidden_size
            ].transpose(0, 1)
        )
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dist_fc2 = nn.Parameter(
            self.distance_matrix[
                input_size: input_size + hidden_size,
                input_size + hidden_size: input_size + hidden_size * 2,
            ].transpose(0, 1)
        )
        self.relu = nn.ReLU()

        # Output layer
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dist_fc3 = nn.Parameter(
            self.distance_matrix[
                input_size + hidden_size: input_size + hidden_size * 2,
                input_size + hidden_size * 2:,
            ].transpose(0, 1)
        )

        self.distances = [self.dist_fc1, self.dist_fc2, self.dist_fc3]
        for distance in self.distances:
            distance.requires_grad = False
        self.distance_matrix.requires_grad = False

        self.layers = [self.fc1, self.fc2, self.fc3]

        # print(
        #     f"Layer1: {self.dist_fc1}, \n "
        #     f"Layer2: {self.dist_fc2}, \n"
        #     f"Layer3: {self.dist_fc3}"
        # )

    def forward(self, x):
        """Pass the input tensors through each of our operation, and return the
        output logits.
        """
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out), dim=1)
        return out

    # Implement distance constraint as L2 norm in the regularizer.
    def compute_weight_regularizer(self, norm: int):
        l2_reg = torch.tensor(0.0, dtype=float)
        for layer, distance in zip(self.layers, self.distances):
            l2_reg = l2_reg + (layer.weight * distance).norm(p=norm)
        return l2_reg
    def compute_lregularizer(self, norm: int):
        l_reg = torch.tensor(0.0, dtype=float)
        for layer in self.layers:
            # l_reg = l_reg + 1/layer.weight.numel() * (layer.weight).norm(p=norm)
            l_reg = l_reg + (layer.weight).norm(p=norm)
        return l_reg


def get_data():
    # MNIST dataset
    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, transform=transforms.ToTensor()
    )

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


"""Train the network"""


def train_nn(
    model, train_loader, test_loader, alpha, beta, params, model_dir, writer, 
    perr
):

    # Check Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")
    # Define the criterion and optimizer
    learning_rate = params.learning_rate
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.to(device)
    
    # total_step = len(train_loader)

    ep_stop = 50
    d_loss = []
    conv_loss = 0.01

    init_w = []
    for layer in model.layers:
        init_w += [torch.flatten(layer.weight).cpu().detach().numpy()]
    init_w = np.concatenate(init_w)
    w_init = np.var(init_w)
    
    scl = 100
    prev_scl = 200

    delta = scl
    thrs = w_init / scl

    model.train()
    while delta > 5e-3:
        for epoch in range(params.n_epochs):
            pred_loss = []
            reg_loss = []
            for im, (images, labels) in enumerate(train_loader):

                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                loss_model = criterion(outputs, labels)

                dist_reg = model.compute_weight_regularizer(norm=params.norm)
                l_reg = model.compute_lregularizer(norm=params.norm)

                loss = loss_model + alpha * (
                    beta * dist_reg + (1 - beta) * l_reg
                )
                pred_loss.append(loss_model.item())
                reg_loss.append(alpha*(beta * dist_reg + (1 - beta) * l_reg).item())


                # Backprpagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for x in model.layers:
                    x.weight.data = x.weight * (torch.abs(x.weight) > thrs)

                writer.add_scalar(
                    "loss", loss.item(), im + epoch * len(train_loader)
                )
                # Tensorboard plot.
                # if im < 99 and epoch == 0:
                #     tensorboard_plot(model, im + epoch * len(train_loader))
                # if (im + 1) % 100 == 0:
                #     tensorboard_plot(model, im + epoch * len(train_loader))
                # if (im + 1) % 100 == 0:
                #     print(
                #         f"Epoch [{epoch + 1}/{params.n_epochs}], "
                #         f"Step [{im + 1}/{total_step}], "
                #         f"Loss: {loss.item()}\t({loss_model.item()}  "
                #         f"{loss_reg.item()})"
                #     )
            # Test the model's performance after each epoch
            accuracy = test_nn(model, test_loader, device, alpha, epoch, writer)
            conns = np.mean([np.sum(torch.abs(x.weight).cpu().detach().numpy() > thrs)/x.weight.numel()
                    for x in model.layers])
            print('\tNum conections {:.5f}'.format(
                    conns), end='\t'
            )
            print('Pred Loss {:.6f} - Reg. Loss {:.6f}'.format(np.mean(pred_loss), np.mean(reg_loss)), flush=True
            )
            writer.add_scalar(
                    "loss_pred", np.mean(pred_loss), epoch
                )
            writer.add_scalar(
                    "loss_reg", np.mean(reg_loss), epoch
                )
            writer.add_scalar(
                    "connections", conns, epoch
                )

            pred_loss = []
            reg_loss = []


            if len(d_loss) >= ep_stop:
                d_loss.pop(0)
            d_loss.append(loss.item())
            if (epoch + 1) % ep_stop == 0:
                acc = np.rint(accuracy)
                if acc < perr:
                    if (
                        np.abs(
                            np.mean(d_loss[: ep_stop // 2])
                            - np.mean(d_loss[ep_stop // 2:])
                        )
                        > conv_loss
                    ):
                        continue
                    else:
                        logger.info(
                            f"Current epoch is {epoch} and accuracy is {acc}. Failing threshold is {thrs}."
                            )
                        restore_path = os.path.join( 
                            model_dir, "checkpoints", f"last.pth"
                        )
                        model = NeuralNet(params.input_size, params.hidden_size,
                            params.output_size, params.n_spatial_dims, params.norm)
                        model.to(device)
                        optimizer = torch.optim.Adam(
                                model.parameters(), lr=learning_rate
                            )

                        ckpt = torch.load(restore_path)
                        model.load_state_dict(ckpt["state_dict"])
                        optimizer.load_state_dict(ckpt["optim_dict"])
                        loss = ckpt["loss"]
                        scl_last = ckpt["scl_last"]

                        delta = abs(prev_scl - scl) / 2
                        prev_scl = scl_last
                        scl = scl + delta
                        thrs = w_init / scl

                        model.train()

                        break
                else:
                    utils.save_checkpoints(
                        {
                            "epoch": epoch + 1,
                            "accuracy": accuracy,
                            "state_dict": model.state_dict(),
                            "optim_dict": optimizer.state_dict,
                            "loss": loss,
                            "scl_last": scl,
                            "threshold": thrs,
                            "init_w": init_w
                        },
                        checkpoint=os.path.join(
                            model_dir,
                            "checkpoints",
                            "last.pth"
                        )
                    )
                    logger.info(f"Checkpoint saved!")
                    
                    delta = abs(prev_scl - scl) / 2
                    prev_scl = scl
                    scl = scl - delta
                    thrs = w_init / scl
                    break 
    logger.info(
            f"Restarting from delta {delta} and scl {scl}, with threshold={thrs} (alpha={alpha}, beta={beta})."
        )   
    return


"""Test the model."""


def test_nn(model, test_loader, device, alpha, epoch, writer):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            "Accuracy of the network on the 10000 test images: {} %".format(
                accuracy
            ), end=''
        )
        writer.add_scalar("accuracy", accuracy, epoch)
    return accuracy


def eval_alpha(train_loader, test_loader, params, model_dir, restore_file=None):
    """Train and evaluate the model with different values of alpha.

    Args:
    model: (torch.nn.Module) the neural network
    train_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches
    training data
    test_loader:(DataLoader) a torch.utils.data.DataLoader object that fetches
    validation
    data optimizer: (torch.optim) optimizer for parameters of model
    model_dir:(string) directory containing config, weights and log
    restore_file: (string) optional- name of file to restore from
    """

    # Reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + ".pth.tar"
        )
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model)

    # Set values for alpha
    alpha = torch.arange(0.00001, 0.00011, 0.00001)
    # alpha = [0.02, 0.03, 0.04, 0.06, 0.07,0.08]
    # alpha = torch.FloatTensor(alpha)
    for a in alpha:
        beta = torch.FloatTensor([0.0, 1.0])
        for b in beta:
            logger.info(f"alpha = {a}, beta={b}")
            model = NeuralNet(params.input_size, params.hidden_size,
                            params.output_size, params.n_spatial_dims, params.norm)
            alpha_dir = os.path.join(
                model_dir,
                f"alpha_{round(a.item(),5)}_beta_{round(b.item(),3)}",
            )
            if not os.path.exists(alpha_dir):
                os.makedirs(alpha_dir)
                logger.info(f"Created directory {alpha_dir}")

            writer = SummaryWriter(log_dir=os.path.join(alpha_dir, "logs"))

            train_nn(model, train_loader, test_loader,
                    a, b, params, alpha_dir, writer, params.perr)
    logger.info("Experiments are completed!")


# def tensorboard_plot(model, step, threshold: float = 0.01):
#     x = []
#     for layer in model.layers:
#         x += [torch.flatten(layer.weight.data).numpy()]
#     x = np.concatenate(x)
#     writer.add_histogram("weight_dist", x, step)

#     w_above = np.mean(x > threshold)

#     writer.add_scalar("w_above_threshold", w_above, step)


parser = argparse.ArgumentParser(
    description="Two layer fully connected net with distance constraints."
)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--restore_file",
    default=None,
    help="Optional, name of the file in --model_dir containing \
                        weights to reload before training",
)

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    logger.info(f"Passed arguments {args}")

    # Initialize the training
    manualSeed = args.seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    json_path = os.path.join(os.getcwd(), "params.json")
    assert os.path.isfile(
        json_path
    ), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Define hyperparameters
    batch_size = params.batch_size
    input_size = params.input_size
    hidden_size = params.hidden_size
    output_size = params.output_size
    trial = params.trial

    # Define dataset
    train_loader, test_loader = get_data()

    model_dir = os.path.join(os.getcwd(), f'trial{trial}',
                             f"seed{manualSeed}trained_model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"Created directory {model_dir}")

    eval_alpha(
        train_loader,
        test_loader,
        params,
        model_dir,
        restore_file=args.restore_file,
    )
