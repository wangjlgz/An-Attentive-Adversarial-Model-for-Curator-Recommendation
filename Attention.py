import numpy as np
import pandas as pd


from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Lambda
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
import tensorflow as tf


def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start : end]
        if dimension == 1:
            return x[:, start : end]
        if dimension == 2:
            return x[:, :, start : end]
        if dimension == 3:
            return x[:, :, :, start : end]
        if dimension == 4:
            return x[:, :, :, :, start : end]
    return Lambda(func)

def crop_nostart(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[ : end]
        if dimension == 1:
            return x[:, : end]
        if dimension == 2:
            return x[:, :,  : end]
        if dimension == 3:
            return x[:, :, :,  : end]
        if dimension == 4:
            return x[:, :, :, :,  : end]
    return Lambda(func)

def cropnoend(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start : ]
        if dimension == 1:
            return x[:, start : ]
        if dimension == 2:
            return x[:, :, start : ]
        if dimension == 3:
            return x[:, :, :, start : ]
        if dimension == 4:
            return x[:, :, :, :, start : ]
    return Lambda(func)


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()
    def build(self, input_shape):
        #print input_shape
        assert len(input_shape) == 3
        self.W = K.variable(self.init((int(input_shape[-1]/2), self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))

        self.W2 = K.variable(self.init((int(input_shape[-1]/2), self.attention_dim)))
        
        self.trainable_weights = [self.W, self.b, self.u,self.W2]
        self.cutting = int(input_shape[-1]/2)
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        x1 = crop(2, 0, self.cutting)(x)
        x2 = crop(2, self.cutting, 2*self.cutting)(x)


        uit1 = K.dot(x1, self.W)
        uit2 = K.dot(x2, self.W2)
        uit = K.tanh(K.bias_add(uit1+uit2, self.b))

        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x1 * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(input_shape[-1]/2))