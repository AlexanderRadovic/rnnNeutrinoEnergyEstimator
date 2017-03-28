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
import matplotlib.pyplot as plt
import time
np.random.seed(1337)  # for reproducibility

import keras
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Embedding, RepeatVector, TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU, Input
from keras.datasets import imdb

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


batch_size = 32

print('Loading data...')
X = np.genfromtxt('numu/inputList.txt',delimiter='*',dtype='string') #the prong level info
Y = np.genfromtxt('numu/truthList.txt') #the labels
H = np.genfromtxt('numu/remainderList.txt') #the "header" with event level information

#dimensions of the prong level information
number_of_variables = 12
number_of_prongs = 10

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

print(X[0], 'first entry, train')

#split into train and test samples
X_test=X_mva[int(X_mva.shape[0]*0.8):];
Y_test=Y[int(Y.shape[0]*0.8):];
H_test=H[int(H.shape[0]*0.8):];

X_train=X_mva[:int(X_mva.shape[0]*0.8)];
Y_train=Y[:int(Y.shape[0]*0.8)];
H_train=H[:int(H.shape[0]*0.8)];

#reshape the header so that it's 2d rather than 1d, which is what keras expects
#Y_train = np.reshape(Y_train, len(Y_train))
H_train = np.reshape(H_train, (len(H_train),1))
#Y_train = np.reshape(Y_train, len(Y_train))
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

print(H_train[0].shape)
print(Y_train[0].shape)
print(X_train[0].shape)

print('Build model...')

#define a model with multiple inputs, one based on prongs the other using our event level header
main_input = Input(shape=X_train[0].shape, dtype='float', name='main_input')
aux_input = Input(shape=H_train[0].shape, dtype='float', name='aux_input')

#push the prong level information through a LSTM
main_branch=LSTM(16)(main_input)

#merge the prong and header level information
x = keras.layers.merge([main_branch, aux_input], mode='concat')

#use combined output to make our energy estimate
main_output = Dense(1, activation='linear', name='main_output')(x)

model = Model(input=[main_input, aux_input], output=main_output)

#mse as a regression task, rmsprop is meant to be be a good pick of LSTMs
model.compile(loss='mse',
              optimizer='rmsprop')

print('Train...')

#start the training, set number of epochs
start_time = time.time()
epochs = 3
resultLog=model.fit([X_train,H_train], Y_train, batch_size=batch_size, nb_epoch=epochs,
          validation_data=([X_test,H_test], Y_test))

average_time_per_epoch = (time.time() - start_time) / epochs    

#take a look at final performance on some example events from the test set
for i in range(10):
    ind = np.random.randint(0, len(X_test))
    rowX, rowH, rowy = X_test[np.array([ind])], H_test[np.array([ind])], Y_test[np.array([ind])]
    preds = model.predict([rowX,rowH], verbose=0)
    print('predict size:', preds.shape)
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

print(resultLog.history)

#save the model definition + weights
model.save('my_model.hdf5')
#save just the weightsx
model.save_weights('test.hdf5') 

