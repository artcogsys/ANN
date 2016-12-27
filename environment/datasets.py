import numpy as np
import random
import chainer.datasets as datasets

#####
## Base classes

class SupervisedData(object):

    def __init__(self, X, T, batch_size=1, shuffle=False):
        """

        :param X: input data
        :param T: target data
        :param batch_size: number of examples per batch
        :param permute: whether or not to shuffle examples

        Note:
        - with batch_size=1 and shuffle=False we have online learning
        - recurrent neural networks require shuffle=False

        """

        self.X = X
        self.T = T

        if shuffle:
            self.perm = np.random.permutation(np.arange(len(X)))
        else:
            self.perm = np.arange(len(X))

        self.batch_size = batch_size

        self.steps = len(X) // batch_size

        self.step = 0

        self.nexamples = self.X.shape[0]

    def __iter__(self):
        return self  # simplest iterator creation

    def next(self):

        if self.step == self.steps:
            self.step = 0
            raise StopIteration

        x = [self.X[self.perm[(seq * self.steps + self.step) % len(self.X)]] for seq in xrange(self.batch_size)]
        t = [self.T[self.perm[(seq * self.steps + self.step) % len(self.T)]] for seq in xrange(self.batch_size)]

        self.step += 1

        return x, t

class UnsupervisedData(object):

    def __init__(self, X, batch_size=1, shuffle=False):

        self.X = X

        if shuffle:
            self.perm = np.random.permutation(np.arange(len(X)))
        else:
            self.perm = np.arange(len(X))

        self.batch_size = batch_size

        self.steps = len(X) // batch_size

        self.step = 0

        self.nexamples = self.X.shape[0]

    def __iter__(self):
        return self  # simplest iterator creation

    def next(self):

        if self.step == self.steps:
            self.step = 0
            raise StopIteration

        x = [self.X[self.perm[(seq * self.steps + self.step) % len(self.X)]] for seq in xrange(self.batch_size)]

        self.step += 1

        return x


#####
## Supervised datasets

class SupervisedFeedforwardClassificationData(SupervisedData):

    def __init__(self, batch_size=1):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [0 if sum(i) < 1.0 else 1 for i in X]
        X = np.array(X, 'float32')
        T = np.array(T, 'int32')

        self.nin = X.shape[1]
        self.nout = np.max(T) + 1

        super(SupervisedFeedforwardClassificationData, self).__init__(X, T, batch_size, shuffle=False)


class SupervisedFeedforwardRegressionData(SupervisedData):

    def __init__(self, batch_size=1):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [[np.sum(i), np.prod(i)] for i in X]
        X = np.array(X, 'float32')
        T = np.array(T, 'float32')

        self.nin = X.shape[1]
        self.nout = T.shape[1]

        super(SupervisedFeedforwardRegressionData, self).__init__(X, T, batch_size, shuffle=False)


class SupervisedRecurrentClassificationData(SupervisedData):

    def __init__(self, batch_size=1):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [0] + [0 if sum(i) < 1.0 else 1 for i in X][:-1]
        X = np.array(X, 'float32')
        T = np.array(T, 'int32')

        self.nin = X.shape[1]
        self.nout = np.max(T) + 1

        super(SupervisedRecurrentClassificationData, self).__init__(X, T, batch_size, shuffle=False)


class SupervisedRecurrentRegressionData(SupervisedData):

    def __init__(self, batch_size=1):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [[1, 0]] + [[np.sum(i), np.prod(i)] for i in X][:-1]
        X = np.array(X, 'float32')
        T = np.array(T, 'float32')

        self.nin = X.shape[1]
        self.nout = T.shape[1]

        super(SupervisedRecurrentRegressionData, self).__init__(X, T, batch_size, shuffle=False)


class MNISTData(SupervisedData):

    def __init__(self, validation=False, convolutional=True, batch_size=1):

        if validation:
            data = datasets.get_mnist()[1]
        else:
            data = datasets.get_mnist()[0]

        X = data._datasets[0].astype('float32')
        T = data._datasets[1].astype('int32')

        if convolutional:
            X = np.reshape(X,np.concatenate([[X.shape[0]], [1], [28, 28]]))
            self.nin = [1, 28, 28]
        else:
            self.nin = X.shape[1]

        self.nout = (np.max(T) + 1)

        super(MNISTData, self).__init__(X, T, batch_size, shuffle=False)