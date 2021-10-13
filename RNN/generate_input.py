#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch.utils.data as data


class FlipFlopDataset(data.Dataset):
    def __init__(self, hps, numel=1024):
        self.numel = numel  # Large number for training loops
        self.batch_size = hps.batch_size
        self.n_time = hps.n_time
        self.n_bits = hps.n_bits
        self.p_flip = hps.p_flip

    def __len__(self):
        return self.numel

    def __getitem__(self, item):

        rng = np.random
        unsigned_inputs = rng.binomial(
            1, self.p_flip, [self.n_time, self.n_bits]
        )

        unsigned_inputs[0, :] = 1

        # Generate random signs {-1, +1}
        random_signs = 2 * rng.binomial(1, 0.5, [self.n_time, self.n_bits]) - 1

        # Apply random signs to input pulses
        inputs = np.multiply(unsigned_inputs, random_signs)
        output = np.zeros([self.n_time, self.n_bits])

        # Update inputs (zero-out random start holds) & compute output

        for bit_idx in range(self.n_bits):
            input_ = np.squeeze(inputs[:, bit_idx])
            t_flip = np.where(input_ != 0)
            for flip_idx in range(np.size(t_flip)):
                # Get the time of the next flip
                t_flip_i = t_flip[0][flip_idx]

                """Set the output to the sign of the flip for the
                remainder of the trial. Future flips will overwrite future
                output"""
                output[t_flip_i:, bit_idx] = inputs[t_flip_i, bit_idx]
        return inputs, output


def get_data(hps):
    # Note that in the FlipFlop task, all input signals are generated randomly,
    # and it is not necessary to have a test_dataset as every sample from the
    # dataset is randomly generated. 
    train_dataset = FlipFlopDataset(hps)
    test_dataset = FlipFlopDataset(hps)

    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=hps.batch_size
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=hps.batch_size
    )
    return train_loader, test_loader
