import json
import logging
import os
import shutil

import torch


class Params():
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


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


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
    # else:
    #     logger.info(f"Checkpoint directory {folder} exists!")
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
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    return checkpoint
