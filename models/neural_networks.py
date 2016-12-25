from chainer import Chain, ChainList
import chainer.links as L
import chainer.functions as F

#####
## Deep Neural Network

class DeepNeuralNetwork(ChainList):
    """
    Fully connected deep neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units
    """

    def __init__(self, ninput, nhidden, noutput, nlayer=2, actfun=F.relu):
        """

        :param ninput: number of inputs
        :param nhidden: number of hidden units
        :param noutput: number of outputs
        :param nlayer: number of weight matrices (2; standard MLP)
        :param actfun: used activation function (ReLU)
        """

        links = ChainList()
        if nlayer == 1:
            links.add_link(L.Linear(ninput, noutput))
        else:
            links.add_link(L.Linear(ninput, nhidden))
            for i in range(nlayer - 2):
                links.add_link(L.Linear(nhidden, nhidden))
            links.add_link(L.Linear(nhidden, noutput))

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nlayer = nlayer
        self.actfun = actfun

        self.h = {}

        self.type = 'feedforward'

        super(DeepNeuralNetwork, self).__init__(links)

    def __call__(self, x):

        if self.nlayer == 1:
            y = self[0][0](x)
        else:
            self.h[0] = self.actfun(self[0][0](x))
            for i in range(1,self.nlayer-1):
                self.h[i] = self.actfun(self[0][i-1](self.h[i-1]))
            y = self[0][-1](self.h[self.nlayer-2])

        return y


    def reset_state(self):
        # allows generic handling of stateful and stateless networks
        pass


#####
## Recurrent Neural Network

class RecurrentNeuralNetwork(ChainList):
    """
    Recurrent neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units
    """

    def __init__(self, ninput, nhidden, noutput, nlayer=2, link=L.LSTM, actfun=F.identity):
        """

        :param ninput: number of inputs
        :param nhidden: number of hidden units
        :param noutput: number of outputs
        :param nlayer: number of weight matrices (2; standard RNN with one hidden layer)
        :param link: used recurrent link (LSTM)
        :param actfun: used activation function (identity)

        """

        links = ChainList()
        if nlayer == 1:
            links.add_link(L.Linear(ninput, noutput))
        else:
            links.add_link(link(ninput, nhidden))
            for i in range(nlayer - 2):
                links.add_link(link(nhidden, nhidden))
            links.add_link(L.Linear(nhidden, noutput))

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nlayer = nlayer
        self.actfun = actfun

        self.h = {}

        self.type = 'recurrent'

        super(RecurrentNeuralNetwork, self).__init__(links)

    def __call__(self, x):

        if self.nlayer == 1:
            y = self[0][0](x)
        else:
            self.h[0] = self.actfun(self[0][0](x))
            for i in range(1,self.nlayer-1):
                self.h[i] = self.actfun(self[0][i-1](self.h[i-1]))
            y = self[0][-1](self.h[self.nlayer-2])

        return y


    def reset_state(self):
        for i in range(self.nlayer - 1):
            self[0][i].reset_state()
