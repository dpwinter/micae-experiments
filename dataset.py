from keras.datasets import mnist
from keras.utils import Sequence
import numpy as np
import math

# Load & preprocess data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

class BatchGenerator(Sequence):
    # Generate ([x], x_merged) for mini-batches

    def __init__(self, x_set, n_x, batch_size):
        self.xs = [x_set for _ in range(n_x)]
        self.batch_size = batch_size
        self.on_epoch_end() # shuffle inputs

    def __len__(self):
        return math.ceil(len(self.xs[0])/self.batch_size)

    def __getitem__(self, idx):
        xs = []
        for x in self.xs:
            batch = x[idx*self.batch_size : (idx+1)*self.batch_size]
            a = np.array(batch)
            xs.append(np.array(batch))
        ys = np.concatenate(xs, axis=1)
        return xs, ys

    def on_epoch_end(self):
        self.xs = [np.random.permutation(x) for x in self.xs]

class BatchGenerator2(Sequence):
    # Generate (x1,x2,x3,x4), x_tot for mini batches (x1-4 are quarters of x_tot)

    def __init__(self, x_set, batch_size):
        self.batch_size = batch_size
        self.xs = x_set
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.xs)/self.batch_size)

    def split(self, arr, nrows, ncols):
        r, h = arr.shape[:-1]
        return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols, 1))

    def __getitem__(self, idx):
        batch = self.xs[idx*self.batch_size : (idx+1)*self.batch_size]
        x1s, x2s, x3s, x4s = [], [], [], []
        for x in batch:
            x1,x2,x3,x4 = self.split(x, 14, 14)
            # x1,x2,x3,x4 = self.split(x, 16, 16)
            x1s.append(x1)
            x2s.append(x2)
            x3s.append(x3)
            x4s.append(x4)
        xs = [np.array(x1s), np.array(x2s), np.array(x3s), np.array(x4s)]
        return xs, batch

    def on_epoch_end(self):
        self.xs = np.random.permutation(self.xs)

class BatchGenerator3(Sequence):
    # Generate ([x], x_merged) for mini-batches by using all same images for x1,x2,x3.

    def __init__(self, x_set, n_x, batch_size):
        self.xs = [x_set for _ in range(n_x)]
        self.batch_size = batch_size
        self.on_epoch_end() # shuffle inputs

    def __len__(self):
        return math.ceil(len(self.xs[0])/self.batch_size)

    def __getitem__(self, idx):
        xs = []
        for x in self.xs:
            batch = x[idx*self.batch_size : (idx+1)*self.batch_size]
            a = np.array(batch)
            xs.append(np.array(batch))
        ys = np.concatenate(xs, axis=1)
        return xs, ys

    def on_epoch_end(self):
        np.random.shuffle(self.xs)
