'''
Incredibly simple macro for plotting a graph of the model.
'''

import keras
from keras.models import load_model
from keras.utils.visualize_util import plot

model = load_model('my_model.hdf5')
plot(model, to_file='model.png', show_shapes=True)
