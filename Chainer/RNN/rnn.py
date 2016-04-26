from chainer import Chain, ChainList, Variable, serializers
from chainer import optimizers
from itertools import product
from termcolor import colored
from time import time
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

class Regressor(Chain):
    """Regressor class links a neural network predictor to a loss function."""

    def __call__(self, t, x):
        self.y = self.predictor(x)
        self.loss = self.lossfun(self.y, t)

        return self.loss

    def __init__(self, predictor, lossfun):
        self.y = None
        self.loss = None
        self.lossfun = lossfun

        super(Regressor, self).__init__(predictor=predictor)

class Predictor(ChainList):
    """"
    Predictor class implements a neural network as given by a list of layers.

    Input:
    x : input data
    internal : Returns the internal activations as well (default: False)

    """

    def __call__(self, x, internal = False):

        if internal:
            activations = []
            for l in self:
                x = l(x)
                activations.append(x)

            return activations.pop(), activations

        else:
            for l in self:
                x = l(x)

            return x

    def __init__(self, *args):
        super(Predictor, self).__init__(*args)

    def reset_state(self):

        for l in self:
            if hasattr(l, 'reset_state') != 0:
                l.reset_state()


class ANN(object):
    """Artificial neural network class."""

    def __init__(self, GPU, optimizer, predictor = optimizers.Adam(), lossfun = F.mean_squared_error, hook=None):

        self.GPU = GPU
        self.log = {}
        self.log['duration'] = np.empty(0)
        self.log['loss'] = np.empty(0)

        self.optimizer = optimizer
        self.regressor = Regressor(predictor,lossfun)

        if self.GPU:
            self.regressor = self.regressor.to_gpu()

        self.optimizer.setup(self.regressor)

        if hook is not None:
            optimizer.add_hook(hook)

    def train(self, T, X, epochs, n, callback=None):
        """
        Train an artificial neural network

        T is a numpy.ndarray of shape (N, L, P)
        N is the number of sequences (e.g. number of runs).
        L is the length of sequences (e.g. length of runs).
        P is the number of output variables.

        X is a numpy.ndarray of shape (N, L, P)
        N is the number of sequences (e.g. number of runs).
        L is the length of sequences (e.g. length of runs).
        P is the number of input variables.

        epochs is the number of training epochs
        n is the minibatch size
        """

        self.log['duration'] = np.concatenate((self.log['duration'], np.empty(epochs)))
        self.log['loss'] = np.concatenate((self.log['loss'], np.empty(epochs)))

        for epoch in xrange(self.optimizer.epoch, self.optimizer.epoch + epochs):

            self.log['duration'][epoch] = time()

            iterations = 0

            if n < T.shape[0]:
                seed = np.random.randint(0, 4294967296)

                np.random.seed(seed)
                np.random.shuffle(T)
                np.random.seed(seed)
                np.random.shuffle(X)

            for iteration in xrange(0, T.shape[0], n):

                loss = Variable(np.zeros((), 'float32'))

                if self.GPU:
                    loss.to_gpu()

                self.regressor.predictor.reset_state()

                for step in xrange(T.shape[1]):

                    t = Variable(T[iteration: iteration + n, step])
                    x = Variable(X[iteration: iteration + n, step])

                    if self.GPU:
                        t.to_gpu()
                        x.to_gpu()

                    loss += self.regressor(t, x)

                    if (step + 1) % 10 == 0 or (step + 1) == T.shape[1]:
                        self.optimizer.zero_grads()
                        loss.backward()
                        loss.unchain_backward()
                        self.optimizer.update()

                iterations += 1

            self.log['loss'][epoch] = loss.data

            self.log['duration'][epoch] = time() - self.log['duration'][epoch]

            self.optimizer.new_epoch()

            if callback is None:

                x = np.arange(0, self.optimizer.epoch)

                plt.subplot(121)
                plt.plot(x, self.log['duration'][:self.optimizer.epoch], 'k')
                plt.xlabel('epoch')
                plt.ylabel('duration')
                plt.subplot(122)
                plt.plot(x, self.log['loss'][:self.optimizer.epoch], 'r')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.tight_layout()

                display.clear_output(wait=True)
                display.display(plt.gcf())

                print 'epoch: {0}'.format(epoch)
                print 'duration: {0}'.format(self.log['duration'][epoch])
                print 'loss: {0}'.format(self.log['loss'][epoch])

                # plt.show()
                plt.close()

            else:
                callback(self, T, X, epochs, n)

    def test(self, X, internal = False):
        """
        Test an artificial neural network

        X is a numpy.ndarray of shape (N, L, P)
        N is the number of sequences (e.g. number of runs).
        L is the length of sequences (e.g. length of runs).
        P is the number of input variables.

        internal : returns all the internal activations as well (default: False)

        """

        for iteration in xrange(X.shape[0]):

            self.regressor.predictor.reset_state()

            for step in xrange(X.shape[1]):

                x = Variable(X[iteration: iteration + 1, step, :])

                if self.GPU:
                    x.to_gpu()

                if internal:
                    [y,activations] = self.regressor.predictor(x,True)
                    print activations.shape
                else:
                    y = self.regressor.predictor(x)

                if self.GPU:
                    y.to_cpu()

                if iteration == step == 0:
                    Y = np.empty((X.shape[0], X.shape[1], y.data.shape[1]), 'float32')

                Y[iteration: iteration + 1, step,:] = y.data

        return Y

    def load(self, prefix):
        self.log = np.load('{0}_log.npy'.format(prefix))[()]

        serializers.load_npz('{0}_optimizer'.format(prefix), self.optimizer)
        serializers.load_npz('{0}_regressor'.format(prefix), self.regressor)

    def save(self, prefix):
        np.save('{0}_log'.format(prefix), self.log)
        serializers.save_npz('{0}_ANN'.format(prefix), self.optimizer)
        serializers.save_npz('{0}_regressor'.format(prefix), self.regressor)

