import argparse
import os
import sys

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from generate_input import get_data
from model import ConstrainedModel
import utils

from torch.utils.tensorboard import SummaryWriter
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from time import time
from collections import defaultdict

def train_nn(
    model,
    train_loader,
    hps,
    model_dir,
    alpha,
    writer,
    beta=None,
):
    # Check Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    learning_rate = hps.learning_rate
    n_epochs = hps.n_epochs

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss stopping criteria
    ep_stop = 20
    d_loss = []
    conv_loss = 1e-3

    model.to(device)
    logger.info("Training starts!")
       
    model.train()
    init_wd = []
    for layer in model.layers_weight:
        init_wd += [torch.flatten(layer.data).cpu().numpy()]
    
    for epoch in range(n_epochs):
        for idx, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()

            inputs, targets = (
                Variable(inputs).to(device),
                Variable(targets).to(device),
            )

            # Forward pass
            hidden_states, outputs, _ = model(inputs)

            loss_model = criterion(outputs, targets)

            # Load the constraints to the network
            dist_reg = model.compute_weight_regularizer(norm=hps.norm)
            l_reg = model.compute_lregularizer(norm=hps.norm)

            loss = loss_model + alpha * (
                beta * dist_reg + (1 - beta) * l_reg
            )

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar(
                "training loss",
                loss.item(),
                idx + epoch * len(train_loader),
            )
           
        if epoch == 0:
            utils.save_checkpoints(
                    {
                        "state_dict": model.state_dict(),
                        "optim_dict": optimizer.state_dict(),
                        "loss": loss,
                        "inputs": inputs,
                        "outputs": outputs,
                        "targets": targets,
                        "hidden_states": hidden_states,
                        "init_wd": init_wd,
                    },
                    checkpoint=os.path.join(
                        model_dir, "checkpoints", "init.pth"
                    ),
                )
        if len(d_loss) >= ep_stop:
            d_loss.pop(0)
        d_loss.append(loss.item())

        if (epoch + 1) % ep_stop == 0:
            acc_avg, acc_max, acc_min, acc_std = utils.evaluate(
                outputs, targets
            )
            accuracies = [acc_avg, acc_max, acc_min, acc_std]
            if (
                np.abs(
                    np.mean(d_loss[: ep_stop // 2])
                    - np.mean(d_loss[ep_stop // 2 :])
                )
                > conv_loss
            ):
                continue
            else:
                logger.info(
                    f"Current epoch is {epoch} and accuracy is {acc_avg}."
                )
                utils.save_checkpoints(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optim_dict": optimizer.state_dict(),
                        "loss": loss,
                        "inputs": inputs,
                        "outputs": outputs,
                        "targets": targets,
                        "hidden_states": hidden_states,
                        "init_wd": init_wd,
                        "accuracy": accuracies,
                    },
                    checkpoint=os.path.join(
                        model_dir, "checkpoints", "last.pth"
                    ),
                )
                print("Checkpoint saved")
                break


def reg_train(hps, model_dir, alpha): 
    if isinstance(alpha, (int, float)):
        alpha = [alpha]
    alpha = (
        torch.arange(0.0, 0.009, 0.001)
        if alpha is None
        else np.array(alpha)
    )
    for a in alpha:
        beta = torch.FloatTensor([0.0, 1.0])
        for b in beta:
            logger.info(f"alpha = {a}, beta = {b}")
            # Initialize new dataset and model for each alpha value
            train_loader, test_loader = get_data(hps)
            model = ConstrainedModel(
                hps.n_bits,
                hps.hidden_size,
                hps.n_bits,
                hps.n_spatial_dims,
                hps.norm,
            )

            # Creates directories for each alpha and beta value
            alpha_dir = os.path.join(
                model_dir,
                f"alpha_{round(a.item(),5)}_beta_{round(b.item(),3)}",
            )
            writer = SummaryWriter(log_dir=os.path.join(alpha_dir, "logs"))
            if not os.path.exists(alpha_dir):
                os.makedirs(alpha_dir)
                logger.info(f"Created directory {alpha_dir}")

            # Initialize training
            train_nn(
                model,
                train_loader,
                hps,
                alpha_dir,
                a,
                writer,
                b
            )
    logger.info("Experiments are completed!")


parser = argparse.ArgumentParser(
    description="Process net with distance constraints and L1/L2 regularization."
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--alpha", type=float, default=None)

args = parser.parse_args()

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

json_path = os.path.join(os.getcwd(), "hps.json")
assert os.path.isfile(
    json_path
), "No json configuration file found at {}".format(json_path)
hps = utils.Params(json_path)

model_dir = os.path.join(
    os.getcwd(), f"trial{hps.trial}", f"seed{manualSeed}trained_model"
)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    logger.info(f"Created directory {model_dir}")

reg_train(
    hps,
    model_dir,
    alpha=args.alpha,
)
