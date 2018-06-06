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

    y_raw = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/truthList.txt',
                          delimiter=',', dtype=str) #labels
    y = np.zeros(len(y_raw))
    for i in range(0, len(y_raw)):
        y[i] = float(y_raw[i][0])

    indices = np.arange(y.shape[0])
    #same random seed as training, should allow us to march the validation set
    np.random.shuffle(indices)

    y_test_presel = y[indices[int(y.shape[0]*0.8):]]

    filterList = np.zeros(len(y_test_presel))

    #Remove events too high to be of interest for oscillation physics
    for i in range(0, len(y_test_presel)):
        filterList[i] = y_test_presel[i] < 10.

    y_test = np.compress(filterList, y_test_presel)
    plot_functions.plot_true_spectra(y_test, args.sample)

if __name__ == "__main__":
    main()
