# -*- coding: utf-8 -*-
'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
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


max_features = 20000
maxlen = 10  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
X = np.genfromtxt('numu/inputList.txt',delimiter='*',dtype='string')
Y = np.genfromtxt('numu/truthList.txt')
N = np.genfromtxt('numu/numuList.txt')
C = np.genfromtxt('numu/caleList.txt')
H = np.genfromtxt('numu/remainderList.txt')

#H=np.concatenate((H, H))

number_of_variables = 12
number_of_prongs = 10

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

X_test=X_mva[int(X_mva.shape[0]*0.8):];
Y_test=Y[int(Y.shape[0]*0.8):];
H_test=H[int(H.shape[0]*0.8):];

C_test=C[int(C.shape[0]*0.8):];
N_test=N[int(N.shape[0]*0.8):];

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
print(C_test[0]-Y_test[0], 'first entry, C')

print('Build model...')

model = load_model('my_model.hdf5')
preds = model.predict([X_test,H_test], verbose=0)

C_perf=(C_test-Y_test)/(Y_test)
N_perf=(N_test-Y_test)/(Y_test)
#print(preds.shape)
preds = np.reshape(preds, (len(X_test)))
#print(preds.shape)
X_perf=(preds-Y_test)/(Y_test)


(C_mu, C_sigma) = norm.fit(C_perf)
(N_mu, N_sigma) = norm.fit(N_perf)
(X_mu, X_sigma) = norm.fit(X_perf)

C_rms=sqrt(mean(square(C_perf)))
N_rms=sqrt(mean(square(N_perf)))
X_rms=sqrt(mean(square(X_perf)))

print('C Sigma', C_sigma)
print('N Sigma', N_sigma)
print('X Sigma', X_sigma)

print('C RMS', C_rms)
print('N RMS', N_rms)
print('X RMS', X_rms)

bins = np.linspace(-1, 1, 100)
plt.hist(C_perf, bins, alpha=0.5)
plt.hist(N_perf, bins, alpha=0.5)
plt.hist(X_perf, bins, alpha=0.5)

# X_g = mlab.normpdf( bins, X_mu, X_sigma)
# X_gl = plt.plot(bins, X_g, 'r--', linewidth=2)
# C_g = mlab.normpdf( bins, C_mu, C_sigma)
# C_gl = plt.plot(bins, C_g, 'b--', linewidth=2)
# N_g = mlab.normpdf( bins, N_mu, N_sigma)
# N_gl = plt.plot(bins, N_g, 'g--', linewidth=2)


plt.show()


