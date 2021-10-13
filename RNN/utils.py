import json
import logging
import os
import shutil

import torch
import matplotlib.pyplot as plt
import numpy as np


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by
        `params.dict['learning_rate']"""
        return self.__dict__


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoints(state, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If
    is_best==True, also saves checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such
        as epoch, optimzer state_dict
        is_best: (bool) True if it is the best model
        till now
        checkpoint: (string) folder where parameters are to be saved
    """
    folder = os.path.dirname(checkpoint)
    filepath = checkpoint  # os.path.join(checkpoint, "last.pth.tar")
    if not os.path.exists(folder):
        print(
            "Checkpoing directory does not exist! Making directory {}".format(
                folder
            )
        )
        os.makedirs(folder, exist_ok=True)
    else:
        print(f"Checkpoint directory {folder} exists!")
    torch.save(state, filepath)


def load_checkpoint(checkpoint, model):
    """Loads model parameters (state_dict) from file_path. If optimizer is
    provided, loads state_dict of optimzer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    return checkpoint


def plot_trial(inputs, outputs, targets, epoch, hps, model_dir):
    """Takes arguments from a trained RNN (either the baseline model or the
    constrained model) and creates trail plots for each bit.
    Figures will be saved in the "trained_model_*/trail_figure" directory.
    """

    n_bits = hps.n_bits
    vertical_spacing = 2.5

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 4, 3])

    directory = "trial_figure"
    model_dir = os.path.join(model_dir, directory)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory {model_dir}")

    for bit_idx in range(n_bits):
        ax.step(
            range(inputs.shape[1]),
            inputs[0, :, bit_idx].detach() + vertical_spacing * bit_idx,
            color="#9C3D17",
            label="Inputs",
            linewidth=2.5,
        )
        ax.plot(
            range(outputs.shape[1]),
            outputs[0, :, bit_idx].detach() + vertical_spacing * bit_idx,
            color="#00119E",
            label="Outputs",
            linewidth=3.5,
        )
        ax.plot(
            range(targets.shape[1]),
            targets[0, :, bit_idx].detach() + vertical_spacing * bit_idx,
            color="#E8C23A",
            label="Targets",
            linewidth=3.5,
        )

    ax.set_yticks([(bit_idx * vertical_spacing) for bit_idx in range(n_bits)])
    ax.set_yticklabels(
        ["Bit %d" % (n_bits - bit_idx) for bit_idx in range(n_bits)],
        fontweight="bold",
    )
    ax.set_title(f"Trials at Epoch_{epoch+1}", fontweight="bold")
    ax.set_xlabel("Time Step", fontweight="bold")

    fig.savefig(
        os.path.join(model_dir, f"Epoch_{epoch+1}.png"),
        bbox_inches="tight",
        dpi="figure",
    )
    plt.close(fig)


def evaluate(
    outputs: torch.tensor, targets: torch.tensor, threshold: float = 0.03
):
    """Computes the mean square error for a specific epoch.

    Args:
        outputs (torch.tensor): trial outputs predicted by the model.
        targets (torch.tensor): groundtrutch targets.
        threshold (float, optional): a hyperparameter that is used to evaluate
        the accuracy of the model, and this value indicates how much do the
        outputs and the targets differ. Defaults to 0.03.

    Returns:
    accuracy (numpy.list): a list contains the mean, maximum and minimum
    accuracy of a specific epoch.
    """
    acc = []
    for o, t in zip(outputs.unbind(), targets.unbind()):
        diff = torch.square(t - o).cpu().detach().numpy()
        acc.append(diff.sum())
    mse, acc_max, acc_min, acc_std = (
        np.mean(acc),
        np.max(acc),
        np.min(acc),
        np.std(acc),
    )
    return mse, acc_max, acc_min, acc_std
