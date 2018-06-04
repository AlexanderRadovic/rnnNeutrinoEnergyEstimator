#Place to store alternative loss functions
#Currently not used, as I'm trying sample weights instead
from __future__ import absolute_import
import six
from keras import backend as K

def weightfromspectra(energy):
    spectraweights=np.load('numuTrueSpecWeight.npy')
    return 1000*spectraweights[int(energy*10)]

    
#Function to weight inverse to size of population in training sample
def flatweight(y_true):
    weights=y_true
    print(K.shape(y_true))
    print(K.eval(y_true))
    for i in range(0, len(y_true)):
        weights[i]=weightfromspectra(y_true[i])
    
    return K.variable(weights)
    
#First attempt at reweighting loss as a function of true energy, to correct for our highly biased input spectra
def flattened_response_for_estimator(y_true, y_pred):
    
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    
    flatrdiff=flatweight(y_true)*diff

    return 100. * K.mean(diff, axis=-1)



