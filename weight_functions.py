"""Functions to calculate sample weights."""
from __future__ import print_function
import numpy as np

def flatweight(y_true, sample):
    """Function to weight inverse to size of population in training sample"""

    spectraweights = np.load(sample+'TrueSpecWeight.npy')
    weights = np.zeros(len(y_true))
    for i in range(0, len(y_true)):
        energy = int(y_true[i]*10)
        if energy > 49:
            energy = 49
        weights[i] = 1000*spectraweights[energy]

    return weights
