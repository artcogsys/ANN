from chainer import Chain, ChainList
import chainer.functions as F
import chainer.links as L

#####
## MLP

class MLP(Chain):
    """
    Multilayer perceptron consisting of one hidden layer
    """

    def __init__(self, ninput, nhidden, noutput):
        """
        :param ninput: number of inputs
        :param nhidden: number of hidden units
        :param noutput: number of outputs
        """
        super(MLP, self).__init__(
            l1=L.Linear(ninput, nhidden),
            l2=L.Linear(nhidden, noutput)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y


#####
## DNN

class DNN(ChainList):
    """
    Fully connected deep neural network consisting of nlayer layers (weight matrices) with each nhidden units
    """

    def __init__(self, ninput, nhidden, noutput, nlayer=2, actfun=F.relu):
        """

        :param ninput: number of inputs
        :param nhidden: number of hidden units
        :param noutput: number of outputs
        :param nlayer: number of weight matrices
        :param actfun: used activation function
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

        super(DNN, self).__init__(links)

    def __call__(self, x):

        h = x
        for i in range(self.nlayer-1):
            h = self.actfun(self[0][i](h))
        y = self[0][-1](h)
        return y
