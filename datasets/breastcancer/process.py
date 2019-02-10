#!/usr/bin/env python3
import random
import pickle

import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    with open('breast-cancer-wisconsin.data') as f:
        d = [line.strip().split(',') for line in f.readlines()]
        d = [
            (sample[1:-1], [sample[-1]]) for sample in d  # strip sample no.s
        ]

        for i, (d_in, d_out) in enumerate(d):
            for j, v in enumerate(d_in):
                if v == '?':
                    d_in[j] = np.nan

        d_in = [si for si, so in d]
        d_out = [so for si, so in d]

        train_data_in, test_data_in, train_data_out, test_data_out = train_test_split(
            d_in, d_out, test_size=.2, random_state=42  # seed RNG
        )

    d = {
        'train': {
            'inputs': train_data_in,
            'outputs': train_data_out
        },
        'test': {
            'inputs': test_data_in,
            'outputs': test_data_out
        }
    }

    with open('breastcancer.dataset', 'wb') as f:
        pickle.dump(d, f)

    print("saved to breastcancer.dataset")
