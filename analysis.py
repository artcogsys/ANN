import numpy as np
import matplotlib.pyplot as plt
from chainer import Variable, cuda

class Analysis(object):

    def __init__(self, model, fname=None, gpu=-1):

        self.fname = fname
        self.model = model

        self.xp = np if gpu == -1 else cuda.cupy


    def supervised_analysis(self, X, T):

        self.model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = True
        self.model.predictor.train = False

        Y = []
        for step in xrange(X.shape[0]):

            x = Variable(self.xp.asarray(X[step][None]), True)
            Y.append(self.model.predict(x))

            if step == 0:
                H = [[self.model.predictor.h[i].data[0]] for i in xrange(len(self.model.predictor.h))]
            else:
                _ = [H[i].append(self.model.predictor.h[i].data[0]) for i in xrange(len(self.model.predictor.h))]

        H = [self.xp.asarray(H[i]) for i in xrange(len(H))]
        Y = np.squeeze(self.xp.asarray(Y))

        #  create scatterplot
        self.scatterplot(T, Y)

    def scatterplot(self,T, Y, idx=0):
        """
        scatterplot for targets T versus predictions Y

        :param T: T x M array
        :param Y: T x M array
        :param idx: output index
        """

        if idx==0 and T.ndim==1:
            t = T
        else:
            t = T[:, idx]

        y = Y[:, idx]

        plt.clf()
        plt.scatter(t, y)
        plt.axis('equal')
        plt.grid(True)
        plt.title('R = ' + str(np.corrcoef(t, y)[0, 1]))

        if self.fname:
            plt.savefig(self.fname + '_scatterplot.png')
        else:
            plt.show()


    def weight_matrix(self, W):
        """
        Plot weight matrix

        :param fname: file name
        :param W: N x M weight matrix
        """

        plt.clf()
        plt.pcolor(W)
        plt.title('Weight matrix')

        if self.fname:
            plt.savefig(self.fname + '_weight_matrix.png')
        else:
            plt.show()

    def functional_connectivity(self,data):
        """
        Plot functional connectivity matrix (full correlation)


        # perform an analysis on the optimal model
        z = [validation_data.X]
        [z.append(H[i]) for i in range(len(H))]
        z.append(Y)
        ana.functional_connectivity(z)


        :param data: list containing T x Mi timeseries data
        """

        x = np.hstack(data)
        M = np.corrcoef(x.transpose())

        plt.clf()
        plt.pcolor(M)
        plt.title('Functional connectivity')

        if self.fname:
            plt.savefig(self.fname + '_functional_connectivity.png')
        else:
            plt.show()


    # def predict(self, data):
    #     """
    #     Predict outcome for a supervised dataset
    #
    #     :param data:
    #     :return: Y, H ; predictions and hidden states
    #     """
    #
    #     self.model.predictor.reset_state()
    #
    #     # check if we are in train or test mode (e.g. for dropout)
    #     self.model.predictor.test = True
    #     self.model.predictor.train = False
    #
    #     Y = []
    #     T = []
    #     for _x, _t in data:
    #
    #         x = Variable(self.xp.asarray(_x), True)
    #         Y.append(self.model.predict(x))
    #         T.append(_t)
    #
    #         if data.step == 1:
    #             H = [[self.model.predictor.h[i].data[0]] for i in xrange(len(self.model.predictor.h))]
    #         else:
    #             _ = [H[i].append(self.model.predictor.h[i].data[0]) for i in xrange(len(self.model.predictor.h))]
    #
    #     # check if hidden state order agrees with Y/T order!
    #     H = [self.xp.asarray(H[i]) for i in xrange(len(H))]
    #     Y = np.squeeze(self.xp.asarray(Y))
    #     T = np.squeeze(self.xp.asarray(T))
    #
    #     # deal with minibatches
    #     if data.batch_size != 1:
    #
    #         shape = Y.shape
    #         Y = np.reshape(Y, [np.prod(shape[0:-1]), shape[-1]])
    #         shape = T.shape
    #         T = np.reshape(T, [np.prod(shape[0:-1]), shape[-1]])
    #
    #     # check if sequential order of recurrent models is maintained!
    #     return Y, T, H