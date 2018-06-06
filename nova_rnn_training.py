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
import argparse
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
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


import weight_functions

def learning_rate_plan(epoch):
    '''define a learning rate strategy'''
    learning_rate = 0.001
    if epoch < 21:
        learning_rate = learning_rate*1
    elif epoch >= 21 and epoch < 41:
        learning_rate = learning_rate*0.5
    elif epoch >= 41 and epoch < 61:
        learning_rate = learning_rate*0.25
    elif epoch >= 61:
        learning_rate = learning_rate*0.125
    return learning_rate

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, required=True)
    parser.add_argument('--weight', type=bool, required=True)
    args = parser.parse_args()

    batch_size = 1024

    print('Loading data...')
    #the prong level info
    x = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/inputList.txt',
                      delimiter='*', dtype='string')
    #the labels
    y_raw = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/truthList.txt',
                          delimiter=',', dtype='string')
    #the "header" with event level information
    h = np.genfromtxt('/mnt/kerasFiles/miniprod4FD/'+args.sample+'/remainderList.txt')

    #dimensions of the prong level information
    number_of_variables = 29
    number_of_prongs = 5

    #reformat prong level input to be broken down by prong and variable.
    x_mva = np.zeros((len(x), len(x[0]), number_of_variables))
    for i in range(0, len(x)):
        for j in range(0, number_of_prongs):
            x_mva[i][j] = (x[i][j].split(","))

    #extract target information
    y = np.zeros(len(x))
    y_lept = np.zeros(len(x))

    for i in range(0, len(x)):
        y[i] = float(y_raw[i][0])
        y_lept[i] = float(y_raw[i][1])

    #check dimensions and sample inputs match expectations
    print('x shape:', x.shape)
    print('x mva shape:', x_mva.shape)
    print('y_raw shape:', y_raw.shape)
    print('y shape:', y.shape)
    print('h shape:', h.shape)

    #split into train and test samples, use indices to keep various arrays in sync
    indices = np.arange(x_mva.shape[0])
    np.random.shuffle(indices)

    x_test = x_mva[indices[int(x_mva.shape[0]*0.8):]]
    y_test = y[indices[int(y.shape[0]*0.8):]]
    y_lept_test = y_lept[indices[int(y.shape[0]*0.8):]]

    h_test = h[indices[int(h.shape[0]*0.8):]]

    x_train = x_mva[indices[:int(x_mva.shape[0]*0.8)]]
    y_train = y[indices[:int(y.shape[0]*0.8)]]
    y_lept_train = y_lept[indices[:int(y.shape[0]*0.8)]]
    h_train = h[indices[:int(h.shape[0]*0.8)]]

    #reshape the header so that it's 2d rather than 1d, which is what keras expects
    h_train = np.reshape(h_train, (len(h_train), 1))
    h_test = np.reshape(h_test, (len(h_test), 1))

    print('Build model...')

    #Calculate weights for the sample
    W_train = weight_functions.flatweight(y_train, args.sample)
    W_test = weight_functions.flatweight(y_test, args.sample)

    #define a model with multiple inputs, one based on prongs the other using our event level header
    main_input = Input(shape=x_train[0].shape, dtype='float', name='main_input')
    aux_input = Input(shape=h_train[0].shape, dtype='float', name='aux_input')

    #push the prong level information through a LSTM or some other form of RNN

    main_branch = LSTM(16)(main_input)

    #merge the prong and header level information
    x = keras.layers.concatenate([main_branch, aux_input])

    #use combined output to make our energy estimate
    main_output = Dense(1, activation='linear', name='main_output')(x)
    lepton_output = Dense(1, activation='linear', name='lepton_output')(x)

    model = Model(inputs=[main_input, aux_input], outputs=[main_output, lepton_output])

    #mse as a regression task, rmsprop is meant to be be a good pick for LSTMs
    rmsprop = optimizers.RMSprop(lr=0.001)
    model.compile(loss='mean_absolute_percentage_error',
                  optimizer=rmsprop)

    print('Train...')

    #start the training, set number of epochs
    start_time = time.time()
    nepochs = 200
    #loss_plan=keras.callbacks.LearningRateScheduler(learning_rate_plan)
    loss_plan = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=10, cooldown=0, verbose=1)

    if args.weight:
        print('Reweight loss to flatten energy response vs. true energy.')
        log_model = keras.callbacks.ModelCheckpoint(args.sample+'Best.hdf5',
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True)
        result_log = model.fit([x_train, h_train], [y_train, y_lept_train],
                               sample_weight=[W_train, W_train],
                               batch_size=batch_size, epochs=nepochs,
                               validation_data=
                               ([x_test, h_test], [y_test, y_lept_test], [W_test, W_test]),
                               callbacks=[loss_plan, log_model])
    else:
        log_model = keras.callbacks.ModelCheckpoint(args.sample+'BestNoWeight.hdf5',
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True)
        result_log = model.fit([x_train, h_train], [y_train, y_lept_train],
                               batch_size=batch_size,
                               epochs=nepochs,
                               validation_data=
                               ([x_test, h_test], [y_test, y_lept_test]),
                               callbacks=[loss_plan, log_model])

    average_time_per_epoch = (time.time() - start_time) / nepochs

    #take a look at final performance on some example events from the test set
    for i in range(10):
        ind = np.random.randint(0, len(x_test))
        row_x, row_h, row_y = x_test[np.array([ind])], h_test[np.array([ind])], y_test[np.array([ind])]
        preds = model.predict([row_x, row_h], verbose=0)
        print('predict size:', preds[0].shape)
        print('q size:', row_x.shape)
        print('t size:', row_y.shape)
        q = row_x[0]
        h = row_h[0]
        correct = row_y[0]
        guess = preds[0][0]
        print('Q', q)
        print('H', h)
        print('T', correct)
        #print('G', guess)
        print(colors.ok + 'Y' + colors.close if (abs((correct-guess)/correct) < 0.05) else
              colors.fail + 'N' + colors.close, guess)
        print('---')


    # Compare models' accuracy, loss and elapsed time per epoch.
    print(average_time_per_epoch, 'average time per epoch')

    # plt.style.use('ggplot')
    # fig, ax = plt.subplots(figsize=(6,6))
    # ax.set_title('Loss')
    # ax.set_ylabel('Validation Loss')
    # ax.set_xlabel('Epochs')
    # ax.plot(result_log.epoch, result_log.history['val_loss'],'r')
    # ax.plot(result_log.epoch, result_log.history['loss'],'b')
    # plt.tight_layout()
    # plt.show()

    np_loss_history_val = np.array(result_log.history['val_loss'])
    np_loss_history_train = np.array(result_log.history['loss'])
    np_epoch = np.array(result_log.epoch)

    np.save('trainloss_numu.npy', np_loss_history_train)
    np.save('valloss_numu.npy', np_loss_history_val)
    np.save('epoch_numu.npy', np_epoch)

    print(result_log.history)

    #save the model definition + weights
    #model.save('my_model_numu.hdf5')
    #save just the weights
    #model.save_weights('test_numu.hdf5')

if __name__ == "__main__":
    main()
