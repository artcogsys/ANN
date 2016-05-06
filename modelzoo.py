from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np

class MLP(Chain):
    """
    Multilayer perceptron with continuous output
    """

    def __init__(self, ninput, nhidden, noutput):
        super(MLP, self).__init__(
            l1=L.Linear(ninput, nhidden),
            l2=L.Linear(nhidden, noutput) #, initialW=np.zeros((noutput, nhidden), dtype=np.float32))
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y  = self.l2(h1)
        return y

    def reset(self):
        pass

class RNN(Chain):
    """
    Recurrent neural network
    """

    def __init__(self, ninput, nhidden, noutput):
        super(RNN, self).__init__(
            l1=L.LSTM(ninput, nhidden),
            l2=L.Linear(nhidden, noutput)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y  = self.l2(h1)
        return y

    def reset(self):
        self.l1.reset_state()