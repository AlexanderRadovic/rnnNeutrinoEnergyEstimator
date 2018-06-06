'''Place to store alternative loss functions'''

from __future__ import absolute_import

from keras import backend as K

def mean_absolute_percentage_error_clipped(y_true, y_pred):
    '''Simple alteration where I can play with clipping
    the loss.'''
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))

    diff = K.clip(diff, 0., 1.)

    return 100. * K.mean(diff, axis=-1)
