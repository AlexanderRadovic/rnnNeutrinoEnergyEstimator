# -*- coding: utf-8 -*-
'''
An attempt to build a LSTM based network for neutrino energy information. 
Combining reconstructed object level informaton with event level information.
The network is also designed to be multitarget, with the aim of also predicting
other useful kinematic information.
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(1337)  # for reproducibility

import keras
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Embedding, RepeatVector, TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU, Input
from keras.regularizers import l1, l2
from keras.datasets import imdb
from keras.callbacks import LearningRateScheduler


import weightFunctions

#define a learning rate strategy
def learning_rate_plan(epoch):
    learningRate=0.001
    if epoch<21:
        learningRate=learningRate*1
    elif epoch>=21 and epoch<41:
         learningRate=learningRate*0.5
    elif epoch>=41:
         learningRate=learningRate*0.25

    return learningRate

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

batch_size = 128

print('Loading data...')
X = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/numu/inputList.txt',delimiter='*',dtype=str) #the prong level info
Y_raw = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/numu/truthList.txt',delimiter=',',dtype=str) #the labels
H = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/numu/remainderList.txt') #the "header" with event level information

#dimensions of the prong level information
number_of_variables = 29
number_of_prongs = 5

#reformat prong level input to be broken down by prong and variable.
X_mva=np.zeros((len(X),len(X[0]),number_of_variables))
for i in range(0,len(X)):
    for j in range(0,number_of_prongs):
        X_mva[i][j]=(X[i][j].split(","))

print('Y_raw shape:', Y_raw.shape)

#extract target information
Y=np.zeros(len(X))
Y_lept=np.zeros(len(X))
Y_had=np.zeros(len(X))
Y_hadfrac=np.zeros(len(X))

for i in range(0,len(X)):
        Y[i]=float(Y_raw[i][0])
        Y_lept[i]=float(Y_raw[i][1])
        Y_had[i]=float(Y_raw[i][2])
        Y_hadfrac[i]=float(Y_raw[i][3])

#check dimensions and sample inputs match expectations        
print('X shape:', X.shape)
print('X mva shape:', X_mva.shape)
print('Y_raw shape:', Y_raw.shape)
print('Y shape:', Y.shape)
print('H shape:', H.shape)

print(len(X), 'train sequences')
print(X[0], 'first entry, train')
print(X_mva[0], 'first entry, train')

print(len(Y), 'target sequences')
print(Y[0], 'first entry, train')

print(len(H), 'target sequences')
print(H[0], 'first entry, train')

print(X[0], 'first entry, train')

#split into train and test samples, use indices to keep various arrays in sync
indices = np.arange(X_mva.shape[0])
np.random.shuffle(indices)

X_test=X_mva[indices[int(X_mva.shape[0]*0.8):]];
Y_test=Y[indices[int(Y.shape[0]*0.8):]];
Y_lept_test=Y_lept[indices[int(Y.shape[0]*0.8):]];
Y_had_test=Y_had[indices[int(Y.shape[0]*0.8):]];
Y_hadfrac_test=Y_hadfrac[indices[int(Y.shape[0]*0.8):]];

H_test=H[indices[int(H.shape[0]*0.8):]];

X_train=X_mva[indices[:int(X_mva.shape[0]*0.8)]];
Y_train=Y[indices[:int(Y.shape[0]*0.8)]];
Y_lept_train=Y_lept[indices[:int(Y.shape[0]*0.8)]];
Y_had_train=Y_had[indices[:int(Y.shape[0]*0.8)]];
Y_hadfrac_train=Y_hadfrac[indices[:int(Y.shape[0]*0.8)]];
H_train=H[indices[:int(H.shape[0]*0.8)]];

#reshape the header so that it's 2d rather than 1d, which is what keras expects
H_train = np.reshape(H_train, (len(H_train),1))
H_test = np.reshape(H_test, (len(H_test),1))

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)
print('H_train shape:', H_train.shape)
print('H_test shape:', H_test.shape)

print(X_train[0], 'first entry, train')
print(Y_train[0], 'first entry, train')
print(H_train[0], 'first entry, train')

print('Build model...')

#Calculate weights for the sample
W_train=flatweight(Y_train)
W_test=flatweight(Y_test)


#define a model with multiple inputs, one based on prongs the other using our event level header
main_input = Input(shape=X_train[0].shape, dtype='float', name='main_input')
aux_input = Input(shape=H_train[0].shape, dtype='float', name='aux_input')

#push the prong level information through a LSTM or some other form of RNN

main_branch=LSTM(16)(main_input)#,W_regularizer = l1(0.001),U_regularizer = l1(0.001),b_regularizer = l1(0.001))(main_input)

#merge the prong and header level information
x = keras.layers.merge([main_branch, aux_input], mode='concat')

#use combined output to make our energy estimate
main_output = Dense(1, activation='linear', name='main_output')(x)
lepton_output = Dense(1, activation='linear', name='lepton_output')(x)
hadron_output = Dense(1, activation='linear', name='hadron_output')(x)
hadfrac_output = Dense(1, activation='linear', name='hadfrac_output')(x)

model = Model(input=[main_input, aux_input], output=[main_output,lepton_output,hadron_output,hadfrac_output])

#mse as a regression task, rmsprop is meant to be be a good pick for LSTMs
rmsprop = optimizers.RMSprop(lr=0.001)
model.compile(loss='mean_absolute_percentage_error',
              optimizer=rmsprop)#,loss_weights={'main_output': 1., 'lepton_output': 0.33, 'hadron_output': 0.33, 'hadfrac_output': 0.33})

print('Train...')

#start the training, set number of epochs
start_time = time.time()
epochs = 10
lossPlan=keras.callbacks.LearningRateScheduler(learning_rate_plan)
resultLog=model.fit([X_train,H_train], [Y_train,Y_lept_train,Y_had_train,Y_hadfrac_train], sample_weight=W_train, batch_size=batch_size, nb_epoch=epochs,
                    validation_data=([X_test,H_test], [Y_test,Y_lept_test,Y_had_test,Y_hadfrac_test], W_test),callbacks=[lossPlan])

average_time_per_epoch = (time.time() - start_time) / epochs    

#take a look at final performance on some example events from the test set
for i in range(10):
    ind = np.random.randint(0, len(X_test))
    rowX, rowH, rowy = X_test[np.array([ind])], H_test[np.array([ind])], Y_test[np.array([ind])]
    preds = model.predict([rowX,rowH], verbose=0)
    print('predict size:', preds[0].shape)
    print('q size:', rowX.shape)
    print('t size:', rowy.shape)
    q = rowX[0]
    h = rowH[0]
    correct = rowy[0]
    guess = preds[0][0]
    print('Q', q)
    print('H', h)
    print('T', correct)
    #print('G', guess)
    print(colors.ok + '☑' + colors.close if (abs((correct-guess)/correct) < 0.05) else colors.fail + '☒' + colors.close, guess)
    print('---')


# Compare models' accuracy, loss and elapsed time per epoch.
print(average_time_per_epoch,'average time per epoch')

# plt.style.use('ggplot')
# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_title('Loss')
# ax.set_ylabel('Validation Loss')
# ax.set_xlabel('Epochs')
# ax.plot(resultLog.epoch, resultLog.history['val_loss'],'r')
# ax.plot(resultLog.epoch, resultLog.history['loss'],'b')
# plt.tight_layout()
# plt.show()

np_loss_history_val = np.array(resultLog.history['val_loss'])
np_loss_history_train = np.array(resultLog.history['loss'])
np_epoch = np.array(resultLog.epoch)

np.save('trainloss_numu.npy',np_loss_history_train)
np.save('valloss_numu.npy',np_loss_history_val)
np.save('epoch_numu.npy',np_epoch)

print(resultLog.history)

#save the model definition + weights
model.save('my_model_numu.hdf5')
#save just the weights
model.save_weights('test_numu.hdf5') 

