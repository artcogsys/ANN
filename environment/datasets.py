import numpy as np
import random
import chainer.datasets as datasets

def get_random_timeseries():

    # We define the inputs as randomly
    # generated timeseries. That is, each input time point comprises two random numbers between
    # zero and one. Then, we define the targets as a function of the inputs. That is, each output
    # time point is zero if the sum of the previous input time point is less than one or one if
    # the sum of the previous input time point is greater than one.

    X = {}
    T = {}
    for phase in ['training', 'validation']:
        X[phase] = [[random.random(), random.random()] for _ in xrange(1000)]
        T[phase] = [0] + [0 if sum(i) < 1.0 else 1 for i in X[phase]][:-1]
        X[phase] = np.array(X[phase], 'float32')
        T[phase] = np.array(T[phase], 'int32')

    nin = X['training'].shape[1]
    nout = np.max(T['training']) + 1

    return [X, T, nin, nout]

def get_mnist():

    # get train and validation data as TupleDatasets
    train, validation = datasets.get_mnist()

    X = {}
    T = {}
    X['training'] = train._datasets[0].astype('float32')
    T['training'] = train._datasets[1].astype('int32')
    X['validation'] = validation._datasets[0].astype('float32')
    T['validation'] = validation._datasets[1].astype('int32')

    nin = X['training'].shape[1]
    nout = (np.max(T['training']) + 1)

    return [X, T, nin, nout]