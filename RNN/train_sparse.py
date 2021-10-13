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

# time_profiles = defaultdict(list)


# def time_interval(time_start):
#     now = time()
#     return now - time_start, now


def train_nn(
    model,
    train_loader,
    hps,
    model_dir,
    alpha,
    writer,
    mode,
    beta=None,
    base_file=None,
    perr=None
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

    if mode == "basic":
        model.train()
        model.to(device)
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
                loss = criterion(outputs, targets)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Keep weights that are larger than the threshold
                for x in model.layers_weight:
                    x.data = x * (torch.abs(x) > thrs)
                
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
                            "scl_last": scl,
                            "threshold": thrs,
                            "inputs": inputs,
                            "outputs": outputs,
                            "targets": targets,
                            "hidden_states": hidden_states,
                            "init_wd": init_wd,
                            "w_init": w_init,
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
                acc = np.rint(acc_avg)

                if (
                    np.abs(
                        np.mean(d_loss[: ep_stop // 2])
                        - np.mean(d_loss[ep_stop // 2 :])
                    )
                    > conv_loss
                ):
                    continue
                else:
                    if acc <= perr:
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
                                "accuracy": accuracies,
                            },
                            checkpoint=os.path.join(
                                model_dir, "checkpoints", "last.pth"
                            ),
                        )
                        logger.info(
                            f"Basic model is saved (Accuracy is {accuracies}!"
                            f"\n The most recent losses are:{d_loss}"
                        )
                        break
                    else:
                        print(
                            "Experiment ended! Model cannot achieve desired performance!"
                        )
    elif mode == "train":
        # Load the base model to start training with threshold
        if base_file == None:
            model.to(device)
            logger.info("Training starts!")
        else:
            logger.info(f"Resume training from {base_file}")
            base = torch.load(base_file)
            model.to(device)
            model.load_state_dict(base["state_dict"])
            optimizer.load_state_dict(base["optim_dict"])
            loss = base["loss"]

        # Set initial variables for weight thresholding
        init_wd = []
        for layer in model.layers_weight:
            init_wd += [torch.flatten(layer.data).cpu().numpy()]
        w_init = np.var(np.concatenate(init_wd))

        scl = hps.scl
        prev_scl = hps.scl*2

        delta = scl
        thrs = w_init / scl

        model.train()

        while delta > 5e-3:
            print(
                f"Restarting from Epoch 0 with delta {delta} and scl {scl}, with threshold={thrs} (alpha={alpha}, beta={beta})"
            )
            for epoch in range(n_epochs):
                pred_loss = []
                reg_loss = []
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
                    pred_loss.append(loss_model.item())
                    reg_loss.append(alpha*(beta * dist_reg + (1 - beta) * l_reg).item())

                    # Backpropagation and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Keep weights that are larger than the threshold
                    for x in model.layers_weight:
                        x.data = x * (torch.abs(x) > thrs)

                    writer.add_scalar(
                        "training loss",
                        loss.item(),
                        idx + epoch * len(train_loader),
                    )

                conns = np.mean([np.sum(torch.abs(x).cpu().detach().numpy() > thrs)/x.numel()
                    for x in model.layers_weight])
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
                if epoch == 0:
                    utils.save_checkpoints(
                            {
                                "state_dict": model.state_dict(),
                                "optim_dict": optimizer.state_dict(),
                                "loss": loss,
                                "scl_last": scl,
                                "threshold": thrs,
                                "inputs": inputs,
                                "outputs": outputs,
                                "targets": targets,
                                "hidden_states": hidden_states,
                                "init_wd": init_wd,
                                "w_init": w_init,
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
                    acc = np.rint(acc_avg)
                    if acc > perr:
                        if (
                            np.abs(
                                np.mean(d_loss[: ep_stop // 2])
                                - np.mean(d_loss[ep_stop // 2 :])
                            )
                            > conv_loss
                        ):
                            continue
                        else:
                            # Restore from the previous checkpoint and resume training with a smaller threshold 
                            logger.info(
                                f"Current epoch is {epoch} and accuracy is {acc}. Failing threshold is {thrs}."
                            )
                            restore_path = os.path.join(
                                model_dir, "checkpoints", f"last.pth"
                            )
                            model = ConstrainedModel(
                                hps.n_bits,
                                hps.hidden_size,
                                hps.n_bits,
                                hps.n_spatial_dims,
                                hps.norm,
                            )
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
                        # Save checkpoint of the latest model with the specified performance and continue training with a larger threshold
                        utils.save_checkpoints(
                            {
                                "epoch": epoch,
                                "state_dict": model.state_dict(),
                                "optim_dict": optimizer.state_dict(),
                                "loss": loss,
                                "scl_last": scl,
                                "threshold": thrs,
                                "inputs": inputs,
                                "outputs": outputs,
                                "targets": targets,
                                "hidden_states": hidden_states,
                                "init_wd": init_wd,
                                "w_init": w_init,
                                "accuracy": accuracies,
                            },
                            checkpoint=os.path.join(
                                model_dir, "checkpoints", "last.pth"
                            ),
                        )
                        print("Checkpoint saved")

                        delta = abs(prev_scl - scl) / 2
                        prev_scl = scl
                        scl = scl - delta
                        thrs = w_init / scl
                        break


def reg_train(hps, model_dir, alpha, mode="basic", base_file=None):
    if mode == "basic":
        alpha = 0.0
        logger.info(f"Start training without constraints.")
        # Initialize new dataset and model for each alpha value
        train_loader, _ = get_data(hps)
        model = ConstrainedModel(
            hps.n_bits,
            hps.hidden_size,
            hps.n_bits,
            hps.n_spatial_dims,
            hps.norm,
        )

        # Creates directory for the basic model
        alpha_dir = os.path.join(model_dir, f"alpha_0.0_beta_none")
        writer = SummaryWriter(log_dir=os.path.join(alpha_dir, "logs"))
        if not os.path.exists(alpha_dir):
            os.makedirs(alpha_dir)
            logger.info(f"Created directory {alpha_dir}")

        # Initialize training
        train_nn(model, train_loader, hps, alpha_dir, alpha, writer, mode)

    elif mode == "train":
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
                    mode,
                    b,
                    base_file,
                    hps.perr
                )
    logger.info("Experiments are completed!")


parser = argparse.ArgumentParser(
    description="Process net with distance constraints and L1/L2 regularization."
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--mode", type=str, default="basic")
parser.add_argument("--base_file", type=str, default=None)

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
    mode=args.mode,
    base_file=args.base_file,
)
