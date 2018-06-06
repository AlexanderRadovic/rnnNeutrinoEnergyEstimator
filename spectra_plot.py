# -*- coding: utf-8 -*-
'''
Make True energy spectra, and inverse to reweight training of our RNN.
'''
from __future__ import print_function
import numpy as np
import argparse
np.random.seed(1337)  # for reproducibility

import plot_functions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, required=True)
    args = parser.parse_args()

    print('Loading data...')

    Y_raw = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/truthList.txt',
                          delimiter=',', dtype=str) #labels
    Y = np.zeros(len(Y_raw))
    for i in range(0, len(Y_raw)):
        Y[i] = float(Y_raw[i][0])

    indices = np.arange(Y.shape[0])
    #same random seed as training, should allow us to march the validation set
    np.random.shuffle(indices)

    Y_test_presel = Y[indices[int(Y.shape[0]*0.8):]]

    filterList = np.zeros(len(Y_test_presel))

    #Remove events too high to be of interest for oscillation physics
    for i in range(0, len(Y_test_presel)):
        filterList[i] = Y_test_presel[i] < 10.

    Y_test = np.compress(filterList, Y_test_presel)
    plot_functions.plotTrueSpectra(Y_test, args.sample)

if __name__ == "__main__":
    main()
