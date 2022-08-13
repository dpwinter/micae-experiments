import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.layers import Input, Lambda
from keras.models import Model

from keras.datasets import mnist
from keras.utils import to_categorical

def gen_data(n_enc, batch_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_splits = n_enc**(1/2)
    nrows, ncols = int(x_train.shape[-2]//n_splits), int(x_train.shape[-1]//n_splits)
    train_data = preprocess(x_train, y_train, nrows, ncols, batch_size)
    test_data = preprocess(x_test, y_test, nrows, ncols, batch_size)
    return train_data, test_data

def preprocess(x_data, y_data, nrows, ncols, batch_size):
    x_data = x_data.reshape(x_data.shape[0], 28, 28, 1)
    x_data = x_data.astype('float32') / 255.
    y_data = to_categorical(y_data)
    x_data_split = np.array([split(x, nrows, ncols) for x in x_data], dtype='float32')
    data = tf.data.Dataset.from_tensor_slices((x_data_split, x_data, y_data))
    # data = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    data = data.shuffle(buffer_size=1024).batch(batch_size)
    return data

class Reshaper(Model):
    # Helper Network to shape arbitrary input to output along sampling dimension
    def __init__(self, input_shape, output_shape):
        model_in  = Input(shape=input_shape)
        model_out = K.concatenate( (K.variable([-1], dtype='int32'), output_shape) ) # -1 (arbitrary samples)
        shaper = Lambda( lambda x: K.reshape(x, model_out) )(model_in)
        super(Reshaper, self).__init__(inputs=model_in, outputs=shaper)

def split(im, nrows, ncols):
    # split 'im' (w*h*c) array into n equal parts.
    width, height = im.shape[:-1]
    im = im.reshape(height//nrows, nrows, -1, ncols)  # split in n 2d arrays along cols
    im = im.swapaxes(1, 2)     # restore order: zig-zag
    im = im.reshape(-1, nrows, ncols, 1) # x 2d arrays with new dims + channel
    return im
