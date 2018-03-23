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

#print means and RMS for different estimators
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

plotResidual(C_perf, N_perf, X_perf, 'numu')

#make 2d array of predicted energy vs. true energy, cut on less interesting regions of energy
X_hist_rvt=np.zeros((40,40))
for i in range(0,len(Y_test)):
    if((Y_test[i] < 5.0) and (preds[i]<5.0) and (Y_test[i] > 1.0) and (preds[i]>1.0)):
        x_bin=int(math.floor((Y_test[i]/5)*50))-10
        y_bin=49-int(math.floor((preds[i]/5)*50))
        X_hist_rvt[y_bin][x_bin]= X_hist_rvt[y_bin][x_bin]+1


N_hist_rvt=np.zeros((40,40))
for i in range(0,len(Y_test)):
    if((Y_test[i] < 5.0) and (N_test[i]<5.0) and (Y_test[i] > 1.0) and (N_test[i]>1.0)):
        x_bin=int(math.floor((Y_test[i]/5)*50))-10
        y_bin=49-int(math.floor((N_test[i]/5)*50))
        N_hist_rvt[y_bin][x_bin]= N_hist_rvt[y_bin][x_bin]+1

C_hist_rvt=np.zeros((40,40))
for i in range(0,len(Y_test)):
    if((Y_test[i] < 5.0) and (C_test[i]<5.0) and (Y_test[i] > 1.0) and (C_test[i]>1.0)):
        x_bin=int(math.floor((Y_test[i]/5)*50))-10
        y_bin=49-int(math.floor((C_test[i]/5)*50))
        C_hist_rvt[y_bin][x_bin]= C_hist_rvt[y_bin][x_bin]+1

        
#make 2d array of (predicted energy-true)/true vs. true energy
X_hist=np.zeros((50,40))
for i in range(0,len(Y_test)):
    if((Y_test[i] < 5.0) and (X_perf[i] < 0.5) and (X_perf[i] > -0.5) and (Y_test[i] > 1.0)):
        x_bin=int(math.floor((Y_test[i]/5)*50))-10
        y_bin=49-int(math.floor((X_perf[i]+0.5)*50))
        X_hist[y_bin][x_bin]= X_hist[y_bin][x_bin]+1
        
N_hist=np.zeros((50,40))
for i in range(0,len(Y_test)):
    if((Y_test[i] < 5.0) and (N_perf[i] < 0.5) and (N_perf[i] > -0.5) and (Y_test[i] > 1.0)):
        x_bin=int(math.floor((Y_test[i]/5)*50))-10
        y_bin=49-int(math.floor((N_perf[i]+0.5)*50))
        N_hist[y_bin][x_bin]= N_hist[y_bin][x_bin]+1
        
C_hist=np.zeros((50,40))
for i in range(0,len(Y_test)):
    if((Y_test[i] < 5.0) and (C_perf[i] < 0.5) and (C_perf[i] > -0.5) and (Y_test[i] > 1.0)):
        x_bin=int(math.floor((Y_test[i]/5)*50))-10
        y_bin=49-int(math.floor((C_perf[i]+0.5)*50))
        C_hist[y_bin][x_bin]= C_hist[y_bin][x_bin]+1


plot2DEnergyResponse(X_hist, 'LSTM', 'Residual', 'rnnEnergy_numu.pdf', False)
plot2DEnergyResponse(N_hist, '3A', 'Residual', '3aEnergy_numu.pdf', False)
plot2DEnergyResponse(C_hist, 'calE', 'Residual', 'caleEnergy_numu.pdf', False)

plot2DEnergyResponse(X_hist, 'LSTM', 'Residual', 'rnnEnergy_numu.pdf', True)
plot2DEnergyResponse(N_hist, '3A', 'Residual', '3aEnergy_numu.pdf', True)
plot2DEnergyResponse(C_hist, 'calE', 'Residual', 'caleEnergy_numu.pdf', True)

plot2DEnergyResponse(X_hist, 'LSTM', 'Residual', 'rnnEnergy_numu_norm.pdf', True)
plot2DEnergyResponse(N_hist, '3A', 'Residual', '3aEnergy_numu_norm.pdf', True)
plot2DEnergyResponse(C_hist, 'calE', 'Residual', 'caleEnergy_numu_norm.pdf', True)

plot2DEnergyResponse(X_hist_rvt, 'LSTM', 'Reconstructed Energy', 'rnnEnergy_numu_rvt_norm.pdf', True)
plot2DEnergyResponse(N_hist_rvt, '3A', 'Reconstructed Energy', '3aEnergy_numu_rvt_norm.pdf', True)
plot2DEnergyResponse(C_hist_rvt, 'calE', 'Reconstructed Energy', 'caleEnergy_numu_rvt_norm.pdf', True)

