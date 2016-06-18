from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np

# class MLP(Chain):
#     """
#     Multilayer perceptron with continuous output
#     """
#
#     def __init__(self, ninput, nhidden, noutput):
#         super(MLP, self).__init__(
#             l1=L.Linear(ninput, nhidden, wscale=np.sqrt(2)),
#             bn1=L.BatchNormalization(nhidden),
#             l2=L.Linear(nhidden, noutput, initialW=np.zeros((noutput, nhidden), dtype=np.float32)))
#             bn2=L.BatchNormalization(noutput)
#         )
#
#     def __call__(self, x):
#         h = F.relu(self.bn1(self.l1(x)))
#         y = self.bn2(self.l2(h))
#         return y

class MLP(Chain):
    """
    Multilayer perceptron with continuous output
    """

    def __init__(self, ninput, nhidden, noutput):
        super(MLP, self).__init__(
            l1=L.Linear(ninput, nhidden, wscale=np.sqrt(2)),
            l2=L.Linear(nhidden, noutput, initialW=np.zeros((noutput, nhidden), dtype=np.float32)))


    def __call__(self, x):
        h = F.relu(self.l1(x))
        y = self.l2(h)
        return y

# class CNN(Chain):
#     """
#     Convolutional neural network
#     """
#
#     def __init__(self, ninput, nframes, nhidden, noutput):
#         super(CNN, self).__init__(
#             # dependence between filter size and padding; output still 20x20 due to padding
#             l1=L.Convolution2D(nframes, nhidden, 3, 1, 1),
#             bn1=L.BatchNormalization(nhidden),
#             l2=L.Linear(ninput[0]*ninput[1]*nhidden, noutput, initialW=np.zeros((noutput, ninput[0]*ninput[1]*nhidden), dtype=np.float32)),
#             bn2=L.BatchNormalization(noutput)
#         )
#
#     def __call__(self, x):
#         h = F.relu(self.bn1(self.l1(x)))
#         y = self.bn2(self.l2(h))
#         return y

class CNN(Chain):
    """
    Convolutional neural network
    """

    def __init__(self, ninput, nframes, nhidden, noutput):
        super(CNN, self).__init__(
            # dependence between filter size and padding; output still 20x20 due to padding
            l1=L.Convolution2D(nframes, nhidden, 3, 1, 1),
            l2=L.Linear(ninput[0]*ninput[1]*nhidden, noutput, initialW=np.zeros((noutput, ninput[0]*ninput[1]*nhidden), dtype=np.float32)))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        y = self.l2(h)
        return y


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

class CRNN(Chain):
    """
    Convolutional recurrent neural network
    """

    def __init__(self, ninput, nhidden, noutput):
        super(CNN, self).__init__(
            l1=L.Convolution2D(1, nhidden, 3, 1, 1),
            l2=L.LSTM(ninput[0]*ninput[1]*nhidden, nhidden),
            l3=L.Linear(nhidden, noutput, initialW=np.zeros((noutput, nhidden), dtype=np.float32))
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y  = self.l3(h2)
        return y

    def reset(self):
        self.l2.reset_state()