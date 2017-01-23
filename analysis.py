import numpy as np
import matplotlib.pyplot as plt
from chainer import Variable, cuda
import scipy.stats as ss

class Analysis(object):

    def __init__(self, model, fname=None, gpu=-1):

        self.fname = fname
        self.model = model

        self.xp = np if gpu == -1 else cuda.cupy

    def regression_analysis(self, X, T):

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

        [nexamples, nregressors] = Y.shape

        plt.clf()

        plt.subplot(121)
        for i in range(nregressors):
            rgb = np.tile(np.random.rand(3),[nexamples,1])
            plt.scatter(T[:, i], Y[:, i], c=rgb)
            plt.hold('on')
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('Observed value')
        plt.ylabel('Predicted value')
        plt.title('Scatterplot')

        plt.subplot(122)
        R = np.zeros([nregressors,1])
        for i in range(nregressors):
            R[i] = ss.pearsonr(np.squeeze(T[:,i]),np.squeeze(Y[:,i]))[0]
        plt.hist(R, np.min([nregressors, 50]), normed=1, facecolor='black')
        plt.grid(True)
        plt.xlabel('Pearson correlation')
        plt.title('Histogram of Pearson correlations')

        if self.fname:
            plt.savefig(self.fname + '_regression_analysis.png')
        else:
            plt.show()

    def classification_analysis(self, X, T):

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

        [nexamples, nregressors] = Y.shape

        plt.clf()

#        agreement = np.argmax(Y,axis=1) == T
        # compute count matrix
        count_mat = np.zeros([nregressors, nregressors])
        conf_mat = np.zeros([nregressors, nregressors])
        for i in range(nregressors):

            # get predictions for trials with real class equal to i
            clf = np.argmax(Y[T==i],axis=1)
            for j in range(nregressors):
                count_mat[i,j] = np.sum(clf == j)
            conf_mat[i] = count_mat[i]/np.sum(count_mat[i])

        # print accuracy
        clf = np.argmax(Y, axis=1)
        print 'accuracy: {0}'.format(np.mean(clf==T))

        plt.subplot(121)
        plt.imshow(count_mat)
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.xticks(np.arange(nregressors))
        plt.gca().set_xticklabels([str(item) for item in 1+np.arange(nregressors)])
        plt.yticks(np.arange(nregressors))
        plt.gca().set_yticklabels([str(item) for item in 1+np.arange(nregressors)])
        plt.colorbar()
        plt.title('Count matrix')

        plt.subplot(122)
        plt.imshow(conf_mat)
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.xticks(np.arange(nregressors))
        plt.gca().set_xticklabels([str(item) for item in 1 + np.arange(nregressors)])
        plt.yticks(np.arange(nregressors))
        plt.gca().set_yticklabels([str(item) for item in 1 + np.arange(nregressors)])
        plt.colorbar()
        plt.title('Confusion matrix')

        if self.fname:
            plt.savefig(self.fname + '_classification_analysis.png')
        else:
            plt.show()

    def accuracy(self, supervised_data):
        """
        Return overall accuracy, calculated in batches (much faster).
        
        :param supervised_data: SupervisedData object
        """
        self.model.predictor.reset_state()
        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = True
        self.model.predictor.train = False
        Y = []
        T = []
        for data in supervised_data:
            x = Variable(self.xp.asarray(data[0]), True)
            Y.append(np.argmax(self.model.predict(x),axis=1))
            T.append(data[1])
        Y = np.squeeze(np.asarray(Y))
        T = np.squeeze(np.asarray(T))
        acc = np.mean(Y==T)      
        return acc
    
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
