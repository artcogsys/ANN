import numpy as np
import random
import chainer.datasets as datasets

def get_supervised_feedforward_classification_data():

    # We define the inputs as randomly generated examples. Each input comprises
    # two random numbers between zero and one. Then, we define the targets as a function of the
    # inputs. That is, each output is zero if the sum of the inputs is less than one
    # or one if the sum of the previous inputs is greater than one.

    X = {}
    T = {}
    for phase in ['training', 'validation']:
        X[phase] = [[random.random(), random.random()] for _ in xrange(1000)]
        T[phase] = [0 if sum(i) < 1.0 else 1 for i in X[phase]]
        X[phase] = np.array(X[phase], 'float32')
        T[phase] = np.array(T[phase], 'int32')

    nin = X['training'].shape[1]
    nout = np.max(T['training']) + 1

    return [X, T, nin, nout]

def get_supervised_feedforward_regression_data():

    # We define the inputs as randomly generated timeseries. That is, each input comprises
    # two random numbers between zero and one. Then, we define the targets as a function of the
    # inputs. That is, the first output is the sum of the inputs and the the second output
    # is the difference of the inputs

    X = {}
    T = {}
    for phase in ['training', 'validation']:
        X[phase] = [[random.random(), random.random()] for _ in xrange(1000)]
        T[phase] = [[np.sum(i), np.prod(i)] for i in X[phase]]
        X[phase] = np.array(X[phase], 'float32')
        T[phase] = np.array(T[phase], 'float32')

    nin = X['training'].shape[1]
    nout = np.max(T['training']) + 1

    return [X, T, nin, nout]


def get_supervised_recurrent_classification_data():

    # We define the inputs as randomly generated timeseries. That is, each input time point comprises
    # two random numbers between zero and one. Then, we define the targets as a function of the previous
    # inputs. That is, each output is zero if the sum of the previous inputs is less than one
    # or one if the sum of the previous inputs is greater than one.

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


def get_supervised_recurrent_regression_data():

    # We define the inputs as randomly generated timeseries. That is, each input time point comprises
    # two random numbers between zero and one. Then, we define the targets as a function of the previous
    # inputs. That is, the first output is the sum of the previous time points and the the second output
    # is the difference of the previous time points

    X = {}
    T = {}
    for phase in ['training', 'validation']:
        X[phase] = [[random.random(), random.random()] for _ in xrange(1000)]
        T[phase] = [[1,0]] + [[np.sum(i), np.prod(i)] for i in X[phase]][:-1]
        X[phase] = np.array(X[phase], 'float32')
        T[phase] = np.array(T[phase], 'float32')

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