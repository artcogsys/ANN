from chainer import Chain
import chainer.functions as F
import chainer.links as L

#####
## RNN

class RNN(Chain):
    """
    Recurrent neural network
    """

    def __init__(self, ninput, nhidden, noutput):
        super(RNN, self).__init__(
            l1=L.LSTM(ninput, nhidden),
            l2=L.Linear(nhidden, noutput)
        )

        self.h = {}

        self.type = 'recurrent'

    def __call__(self, x):
        self.h[0] = self.l1(x)
        y  = self.l2(self.h[0])
        return y

    def reset_state(self):
        self.l1.reset_state()
