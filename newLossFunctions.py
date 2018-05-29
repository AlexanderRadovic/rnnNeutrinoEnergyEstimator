#Place to store alternative loss functions
from __future__ import absolute_import
import six
from keras import backend as K

#First attempt at reweighting loss as a function of true energy, to correct for our highly biased input spectra
def flattened_response_for_estimator(y_true, y_pred):
    diff = K.exp(-y_true)*K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)
