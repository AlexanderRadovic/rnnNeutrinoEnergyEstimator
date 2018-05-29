# -*- coding: utf-8 -*-
'''
Make simple plots and fits to benchmark performance of LSTM based energy estimator against existing approaches.
'''
from __future__ import print_function
import numpy as np
import math
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
import plotFunctions


print('Loading data...')

X = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/numu/inputList.txt',delimiter='*',dtype='string') #prong level information

Y_raw = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/numu/truthList.txt',delimiter=',',dtype='string') #labels
Y=np.zeros(len(X))
for i in range(0,len(X)):
        Y[i]=float(Y_raw[i][0])

N_raw = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/numu/numuList.txt',delimiter=',',dtype='string') #numu energy estimator
N=np.zeros(len(X))
for i in range(0,len(X)):
        N[i]=float(N_raw[i][0])


C = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/numu/caleList.txt') #calorimetric energy
H = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/numu/remainderList.txt') #header information

#dimensions of the prong level information
number_of_variables = 29
number_of_prongs = 5

#reformat prong level input to be broken down by prong and variable.
X_mva=np.zeros((len(X),len(X[0]),number_of_variables))
for i in range(0,len(X)):
    for j in range(0,number_of_prongs):
        X_mva[i][j]=(X[i][j].split(","))
        
indices = np.arange(X_mva.shape[0])
#same random seed as training, should allow us to march the validation set
np.random.shuffle(indices)

X_test=X_mva[indices[int(X_mva.shape[0]*0.8):]];
Y_test_presel=Y[indices[int(Y.shape[0]*0.8):]];
H_test=H[indices[int(H.shape[0]*0.8):]];
C_test_presel=C[indices[int(C.shape[0]*0.8):]];
N_test_presel=N[indices[int(N.shape[0]*0.8):]];
H_test = np.reshape(H_test, (len(H_test),1))

filterList=np.zeros(len(Y_test_presel))

#Remove events too high to be of interest for oscillation physics
for i in range(0,len(Y_test_presel)):
    filterList[i]=Y_test_presel[i]<5.

Y_test=np.compress(filterList,Y_test_presel)
C_test=np.compress(filterList,C_test_presel)
N_test=np.compress(filterList,N_test_presel)

print('Build model...')

#load and run pretrained LSTM estimator
model = load_model('my_model_numu.hdf5')
preds = model.predict([X_test,H_test], verbose=0)

#makes reco-true residuals
C_perf=(C_test-Y_test)/(Y_test)
N_perf=(N_test-Y_test)/(Y_test)
preds_presel = np.reshape(preds[0][:], (len(X_test)))
preds=np.compress(filterList,preds_presel)
X_perf=(preds-Y_test)/(Y_test)


plotFunctions.plotTrueSpectra(Y_test, 'numu')

