# -*- coding: utf-8 -*-
'''
Make simple plots and fits to benchmark performance of LSTM based energy estimator against existing approaches.
'''
from __future__ import print_function
import numpy as np
from numpy import mean, sqrt, square
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import time
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

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

print('Loading data...')
X = np.genfromtxt('/media/alexander/SAMSUNG/kerasFiles/numu_new/all/inputList.txt',delimiter='*',dtype='string') #prong level information
Y = np.genfromtxt('/media/alexander/SAMSUNG/kerasFiles/numu_new/all/truthList.txt') #labels
N = np.genfromtxt('/media/alexander/SAMSUNG/kerasFiles/numu_new/all/numu3aList.txt') #numu energy estimator
C = np.genfromtxt('/media/alexander/SAMSUNG/kerasFiles/numu_new/all/caleList.txt') #calorimetric energy
H = np.genfromtxt('/media/alexander/SAMSUNG/kerasFiles/numu_new/all/remainderList.txt') #header information


number_of_variables = 14
number_of_prongs = 5

#reformat prong level input to be broken down by prong and variable.
X_mva=np.zeros((len(X),len(X[0]),number_of_variables))
for i in range(0,len(X)):
    for j in range(0,number_of_prongs):
        X_mva[i][j]=(X[i][j].split(","))
        
print('X shape:', X.shape)
print('X mva shape:', X_mva.shape)
print('Y shape:', Y.shape)
print('H shape:', H.shape)

print(len(X), 'train sequences')
print(X[0], 'first entry, train')
print(X_mva[0], 'first entry, train')

print(len(Y), 'target sequences')
print(Y[0], 'first entry, train')

print(len(H), 'target sequences')
print(H[0], 'first entry, train')

#X = sequence.pad_sequences(X, maxlen=maxlen)
print(X[0], 'first entry, train')

indices = np.arange(X_mva.shape[0])
np.random.shuffle(indices)
X_test=X_mva[indices[int(X_mva.shape[0]*0.8):]];
Y_test=Y[indices[int(Y.shape[0]*0.8):]];
H_test=H[indices[int(H.shape[0]*0.8):]];
C_test=C[indices[int(C.shape[0]*0.8):]];
N_test=N[indices[int(N.shape[0]*0.8):]];


H_test = np.reshape(H_test, (len(H_test),1))

print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
print('H_test shape:', H_test.shape)
print('C_test shape:', C_test.shape)
print('N_test shape:', N_test.shape)

print(X_test[0], 'first entry, X')
print(Y_test[0], 'first entry, Y')
print(H_test[0], 'first entry, H')
print(C_test[0], 'first entry, C')
print(N_test[0], 'first entry, N')

print('Build model...')

#load and run pretrained LSTM estimator
model = load_model('my_model_numu.hdf5')
preds = model.predict([X_test,H_test], verbose=0)

#makes reco-true residuals
C_perf=(C_test-Y_test)/(Y_test)
N_perf=(N_test-Y_test)/(Y_test)
preds = np.reshape(preds, (len(X_test)))
X_perf=(preds-Y_test)/(Y_test)


#fit a simple gaussian distribution
(C_mu, C_sigma) = norm.fit(C_perf)
(N_mu, N_sigma) = norm.fit(N_perf)
(X_mu, X_sigma) = norm.fit(X_perf)

#get the rms
C_rms=sqrt(mean(square(C_perf)))
N_rms=sqrt(mean(square(N_perf)))
X_rms=sqrt(mean(square(X_perf)))

C_m=(mean(C_perf))
N_m=(mean(N_perf))
X_m=(mean(X_perf))

print('C Sigma', C_sigma)
print('N Sigma', N_sigma)
print('X Sigma', X_sigma)

print('C Mu', C_mu)
print('N Mu', N_mu)
print('X Mu', X_mu)

print('C Mean', C_m)
print('N Mean', N_m)
print('X Mean', X_m)

print('C RMS', C_rms)
print('N RMS', N_rms)
print('X RMS', X_rms)

bins = np.linspace(-1, 1, 100)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_title('')
ax.set_ylabel('Events')
ax.set_xlabel('(Reco E - True E)/True E')
plt.hist(C_perf, bins, color='b', alpha=0.9, histtype='step',lw=2,label='CalE')
plt.hist(N_perf, bins, color='g', alpha=0.9, histtype='step',lw=2,label='Biswa NuMu')
plt.hist(X_perf, bins, color='r', alpha=0.9, histtype='step',lw=2,label='LSTM')
ax.legend(loc='right',frameon=False)
plt.savefig('numuComparison.png',dpi = 1000)
