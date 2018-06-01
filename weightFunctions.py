
#Functions to calculate sample weights.

#Function to pull weight from reference array
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
    
    return weights
