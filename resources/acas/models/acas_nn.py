import keras
from keras.layers import Activation,Dense
from keras.models import Sequential
import numpy as np


def parse_acas_nnet(filename, h5name):
    f = open('{}.nnet'.format(filename), 'r')
    """
    Read comment lines
    """
    line = f.readline()
    while line[0] == '/':
        line = f.readline()
    """
    Read number of layers, input size and layers' sizes
    """
    line = line.strip()
    line = line[0:-1]
    data = line.split(',')
    num_layers = int(data[0])
    input_size = int(data[1])
    line = f.readline()
    line = line.strip()
    line = line[0:-1]
    layer_sizes = line.split(',')
    for i in range(len(layer_sizes)):
        layer_sizes[i] = int(layer_sizes[i])
    """
    Define a weights matrix of four dimensions:
    1. layer
    2. weight_matrix if 0 or bias vector if 1
    3. node in the previous layer
    4. node in the layer
    """
    weights = []
    for i in range(num_layers):
        shape = (layer_sizes[i], layer_sizes[i + 1])
        matrix = np.empty(shape=shape, dtype=np.float32)
        bias = np.empty(shape=layer_sizes[i + 1], dtype=np.float32)
        weights.append((matrix, bias))
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    """
    Read the weights
    """
    layer = 0
    w_switch = 0
    i = 0
    j = 0
    for line in f:
        if i >= layer_sizes[layer + 1]:

            if w_switch == 0:
                w_switch = 1
            else:
                w_switch = 0
                layer += 1
            i = 0
            j = 0

        line = line.strip()
        line = line[0:-1]
        data = line.split(',')
        for d in data:
            if w_switch == 0:
                weights[layer][w_switch][j][i] = float(d)
                j += 1
            else:
                weights[layer][w_switch][i] = float(d)
        j = 0
        i += 1
    """
    Construct the keras model
    """
    model = Sequential()
    model.add(Dense(layer_sizes[1], activation='relu', input_shape=(input_size,)))
    model.layers[-1].set_weights(weights[0])
    for i in range(1, num_layers-1):
        model.add(Dense(layer_sizes[i + 1], activation='relu'))
        model.layers[-1].set_weights(weights[i])
    model.add(Dense(layer_sizes[num_layers], activation='linear'))
    model.layers[-1].set_weights(weights[num_layers-1])
    model.save('{}.h5'.format(h5name))


for i in range(5):
    for j in range(9):
        filename = "ACASXU_run2a_{}_{}_batch_2000".format(i+1,j+1)
        h5name = "acas_{}_{}".format(i+1,j+1)
        parse_acas_nnet(filename, h5name)
