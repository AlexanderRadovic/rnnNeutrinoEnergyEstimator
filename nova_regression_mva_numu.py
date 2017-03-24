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

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, RepeatVector, TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


max_features = 20000
maxlen = 10  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
X = np.genfromtxt('inputList.txt',delimiter='*',dtype='string')
Y = np.genfromtxt('truthList.txt')
number_of_variabes = 11
number_of_prongs = 10

X_mva=np.zeros((len(X),len(X[0]),number_of_variables))

for i in range(0,len(X)):
    for j in range(0,number_of_prongs):
        X_mva[i][j]=(X[i][j].split(","))
        
print('X shape:', X.shape)
print('X mva shape:', X_mva.shape)
print('Y shape:', Y.shape)

print(len(X), 'train sequences')
print(X[0], 'first entry, train')
print(X_mva[0], 'first entry, train')

print(len(Y), 'target sequences')
print(Y[0], 'first entry, train')



#X = sequence.pad_sequences(X, maxlen=maxlen)
print(X[0], 'first entry, train')

X_test=X_mva[int(X_mva.shape[0]*0.8):];
Y_test=Y[int(Y.shape[0]*0.8):];

X_train=X_mva[:int(X_mva.shape[0]*0.8)];
Y_train=Y[:int(Y.shape[0]*0.8)];

Y_train = np.reshape(Y_train, len(Y_train))
Y_test = np.reshape(Y_test, len(Y_test))

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


print(X_train[0], 'first entry, train')
print(Y_train[0], 'first entry, train')

print('Build model...')
model = Sequential()
model.add(TimeDistributed(Dense(12), input_shape=X_train[0].shape))
model.add(TimeDistributed(Dense(12)))
model.add(TimeDistributed(Dense(6)))
model.add(LSTM(128, dropout_W=0.1, dropout_U=0.1))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('linear'))

# try using different optimizers and different optimizer configs
model.compile(loss='mse',#loss='binary_crossentropy',
              optimizer='rmsprop')

print('Train...')
start_time = time.time()
epochs = 4
#for iteration in range(1, epochs+1):
#    print()
#    print('-' * 50)
#    print('Iteration', iteration)
resultLog=model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
          validation_data=(X_test, Y_test))
average_time_per_epoch = (time.time() - start_time) / epochs    
# Select 10 samples from the validation set at random so we can visualize errors
for i in range(10):
    ind = np.random.randint(0, len(X_test))
    rowX, rowy = X_test[np.array([ind])], Y_test[np.array([ind])]
    preds = model.predict(rowX, verbose=0)
    print('predict size:', preds.shape)
    print('q size:', rowX.shape)
    print('t size:', rowy.shape)
    q = rowX[0]
    correct = rowy[0]
    guess = preds[0][0]
    print('Q', q)
    print('T', correct)
    #print('G', guess)
    print(colors.ok + '☑' + colors.close if (abs((correct-guess)/correct) < 0.05) else colors.fail + '☒' + colors.close, guess)
    print('---')


# Compare models' accuracy, loss and elapsed time per epoch.

print(average_time_per_epoch,'average time per epoch')

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title('Loss')
ax.set_ylabel('Validation Loss')
ax.set_xlabel('Epochs')
ax.plot(resultLog.epoch, resultLog.history['val_loss'],'r')
ax.plot(resultLog.epoch, resultLog.history['loss'],'b')
plt.tight_layout()
plt.show()

print(resultLog.history)
model.save_weights('test.hdf5') 

