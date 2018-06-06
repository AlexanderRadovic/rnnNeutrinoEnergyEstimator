# -*- coding: utf-8 -*-
'''
Make simple plots and fits to benchmark performance of LSTM based
energy estimator against existing approaches.
'''
from __future__ import print_function
import numpy as np
import math
from numpy import mean, sqrt, square
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import time
import argparse
np.random.seed(1337)  # for reproducibility

import keras
from keras.models import load_model
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Embedding, RepeatVector, TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU, Input
from keras.datasets import imdb
import plot_functions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, required=True)
    args = parser.parse_args()

    print('Loading data...')

    #prong level information
    x = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/inputList.txt',
                      delimiter='*', dtype=str)

    #labels
    y_raw = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/truthList.txt',
                          delimiter=',', dtype=str)
    y = np.zeros(len(x))
    for i in range(0, len(x)):
        y[i] = float(y_raw[i][0])

    n_raw = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/'+args.sample+'List.txt',
                          delimiter=',', dtype=str)
    n = np.zeros(len(x))
    for i in range(0, len(x)):
        n[i] = float(n_raw[i][0])


    #calorimetric energy
    c = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/caleList.txt')
    #header information
    h = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/remainderList.txt')

    #dimensions of the prong level information
    number_of_variables = 29
    number_of_prongs = 5

    #reformat prong level input to be broken down by prong and variable.
    x_mva = np.zeros((len(x), len(x[0]), number_of_variables))
    for i in range(0, len(x)):
        for j in range(0, number_of_prongs):
            x_mva[i][j] = (x[i][j].split(","))

    indices = np.arange(x_mva.shape[0])
    #same random seed as training, should allow us to march the validation set
    np.random.shuffle(indices)

    x_test = x_mva[indices[int(x_mva.shape[0]*0.8):]]
    y_test_presel = y[indices[int(y.shape[0]*0.8):]]
    h_test = h[indices[int(h.shape[0]*0.8):]]
    c_test_presel = c[indices[int(c.shape[0]*0.8):]]
    n_test_presel = n[indices[int(n.shape[0]*0.8):]]
    h_test = np.reshape(h_test, (len(h_test), 1))

    filterList = np.zeros(len(y_test_presel))

    #Remove events too high to be of interest for oscillation physics
    for i in range(0, len(y_test_presel)):
        filterList[i] = y_test_presel[i] < 5.

    y_test = np.compress(filterList, y_test_presel)
    c_test = np.compress(filterList, c_test_presel)
    n_test = np.compress(filterList, n_test_presel)

    print('Build model...')

    #load and run pretrained LSTM estimator
    model = load_model(args.sample+'Best.hdf5')
    preds = model.predict([x_test, h_test], verbose=0)

    #makes reco-true residuals
    c_perf = (c_test-y_test)/(y_test)
    n_perf = (n_test-y_test)/(y_test)
    preds_presel = np.reshape(preds[0][:], (len(x_test)))
    preds = np.compress(filterList, preds_presel)
    x_perf = (preds-y_test)/(y_test)

    #fit a simple gaussian distribution
    (c_mu, c_sigma) = norm.fit(c_perf)
    (n_mu, n_sigma) = norm.fit(n_perf)
    (x_mu, x_sigma) = norm.fit(x_perf)

    #get the rms
    c_rms = sqrt(mean(square(c_perf)))
    n_rms = sqrt(mean(square(n_perf)))
    x_rms = sqrt(mean(square(x_perf)))

    c_m = (mean(c_perf))
    n_m = (mean(n_perf))
    x_m = (mean(x_perf))

    #print means and RMS for different estimators
    print('c Sigma', c_sigma)
    print('n Sigma', n_sigma)
    print('x Sigma', x_sigma)

    print('c Mu', c_mu)
    print('n Mu', n_mu)
    print('x Mu', x_mu)

    print('c Mean', c_m)
    print('n Mean', n_m)
    print('x Mean', x_m)

    print('c RMS', c_rms)
    print('n RMS', n_rms)
    print('x RMS', x_rms)

    plot_functions.plot_residual(x_perf, n_perf, c_perf, args.sample)

    #make 2d array of predicted energy vs. true energy, cut on less interesting regions of energy
    x_hist_rvt = np.zeros((50, 50))
    for i in range(0, len(y_test)):
        if (y_test[i] < 5.0) and (preds[i] < 5.0):
            x_bin = int(math.floor((y_test[i]/5)*50))
            y_bin = 49-int(math.floor((preds[i]/5)*50))
            x_hist_rvt[y_bin][x_bin] = x_hist_rvt[y_bin][x_bin]+1


    n_hist_rvt = np.zeros((50, 50))
    for i in range(0, len(y_test)):
        if (y_test[i] < 5.0) and (n_test[i] < 5.0):
            x_bin = int(math.floor((y_test[i]/5)*50))
            y_bin = 49-int(math.floor((n_test[i]/5)*50))
            n_hist_rvt[y_bin][x_bin] = n_hist_rvt[y_bin][x_bin]+1

    c_hist_rvt = np.zeros((50, 50))
    for i in range(0, len(y_test)):
        if (y_test[i] < 5.0) and (c_test[i] < 5.0):
            x_bin = int(math.floor((y_test[i]/5)*50))
            y_bin = 49-int(math.floor((c_test[i]/5)*50))
            c_hist_rvt[y_bin][x_bin] = c_hist_rvt[y_bin][x_bin]+1

    #make 2d array of (predicted energy-true)/true vs. true energy
    x_hist = np.zeros((50, 50))
    for i in range(0, len(y_test)):
        if (y_test[i] < 5.0) and (x_perf[i] < 0.5) and (x_perf[i] > -0.5):
            x_bin = int(math.floor((y_test[i]/5)*50))
            y_bin = 49-int(math.floor((x_perf[i]+0.5)*50))
            x_hist[y_bin][x_bin] = x_hist[y_bin][x_bin]+1

    n_hist = np.zeros((50, 50))
    for i in range(0, len(y_test)):
        if (y_test[i] < 5.0) and (n_perf[i] < 0.5) and (n_perf[i] > -0.5):
            x_bin = int(math.floor((y_test[i]/5)*50))
            y_bin = 49-int(math.floor((n_perf[i]+0.5)*50))
            n_hist[y_bin][x_bin] = n_hist[y_bin][x_bin]+1

    c_hist = np.zeros((50, 50))
    for i in range(0, len(y_test)):
        if (y_test[i] < 5.0) and (c_perf[i] < 0.5) and (c_perf[i] > -0.5):
            x_bin = int(math.floor((y_test[i]/5)*50))
            y_bin = 49-int(math.floor((c_perf[i]+0.5)*50))
            c_hist[y_bin][x_bin] = c_hist[y_bin][x_bin]+1

    plot_functions.plot2D_energy_response(x_hist, 'LSTM', 'Residual',
                                          'rnnEnergy_'+args.sample+'.png', False, 0, 5, -50, 50)
    plot_functions.plot2D_energy_response(n_hist, '3A', 'Residual',
                                          '3aEnergy_'+args.sample+'.png', False, 0, 5, -50, 50)
    plot_functions.plot2D_energy_response(c_hist, 'calE', 'Residual',
                                          'caleEnergy_'+args.sample+'.png', False, 0, 5, -50, 50)

    plot_functions.plot2D_energy_response(x_hist, 'LSTM', 'Residual',
                                          'rnnEnergy_norm_'+args.sample+'.png', True, 0, 5, -50, 50)
    plot_functions.plot2D_energy_response(n_hist, '3A', 'Residual',
                                          '3aEnergy_norm_'+args.sample+'.png', True, 0, 5, -50, 50)
    plot_functions.plot2D_energy_response(c_hist, 'calE', 'Residual',
                                          'caleEnergy_norm_'+args.sample+'.png', True, 0, 5, -50, 50)

    plot_functions.plot2D_energy_response(x_hist_rvt, 'LSTM', 'Reconstructed Energy',
                                          'rnnEnergy_'+args.sample+'_rvt_norm.png', True, 0, 5, 0, 5)
    plot_functions.plot2D_energy_response(n_hist_rvt, '3A', 'Reconstructed Energy',
                                          '3aEnergy_'+args.sample+'_rvt_norm.png', True, 0, 5, 0, 5)
    plot_functions.plot2D_energy_response(c_hist_rvt, 'calE', 'Reconstructed Energy',
                                          'caleEnergy_'+args.sample+'_rvt_norm.png', True, 0, 5, 0, 5)

if __name__ == "__main__":
    main()
