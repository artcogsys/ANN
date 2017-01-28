from chainer import Variable, cuda, serializers
import numpy as np
import pickle
import time
import tqdm
import matplotlib.pyplot as plt


class SupervisedLearner(object):

    def __init__(self, optimizer, gpu=-1):

        self.model = optimizer.target

        self.optimizer = optimizer

        self.log = {}
        self.log[('training', 'loss')] = []
        self.log[('training', 'throughput')] = []
        self.log[('validation', 'loss')] = []
        self.log[('validation', 'throughput')] = []

        self.xp = np if gpu==-1 else cuda.cupy

    def optimize(self, training_data, validation_data=None, epochs=50):
        """

        :param training_data: Required training data set
        :param validation_data: Optional validation data set; optimize returns best model
                according to validation or last model it was trained on
        :param epochs: number of training epochs
        :return:
        """

        # keep track of minimal validation loss
        min_loss = float('nan')

        for epoch in tqdm.tqdm(xrange(self.optimizer.epoch, self.optimizer.epoch + epochs)):

            then = time.time()
            loss = self.train(training_data)
            now = time.time()
            throughput = training_data.nexamples / (now - then)

            self.log[('training', 'loss')].append(loss)
            self.log[('training', 'throughput')].append(throughput)

            # testing in batch mode is much faster
            if validation_data:
                then = time.time()
                loss = self.test(validation_data)
                now = time.time()
                throughput = validation_data.nexamples / (now - then)
            else:
                loss = float('nan')
                throughput = float('nan')

            self.log[('validation', 'loss')].append(loss)
            self.log[('validation', 'throughput')].append(throughput)

            # store optimal model
            if np.isnan(min_loss):
                optimal_model = self.optimizer.target.copy()
                min_loss = self.log[('validation', 'loss')][-1]
            else:
                if self.log[('validation', 'loss')][-1] < min_loss:
                    optimal_model = self.optimizer.target.copy()
                    min_loss = self.log[('validation', 'loss')][-1]

            self.optimizer.new_epoch()

        # model is set to the optimal model according to validation loss
        # or to last model in case no validation set is used
        self.model = optimal_model

    def load(self, fname):

        with open('{}_log'.format(fname), 'rb') as f:
            self.log = pickle.load(f)

        serializers.load_npz('{}_optimizer'.format(fname), self.optimizer)
        serializers.load_npz('{}_model'.format(fname), self.model)


    def save(self, fname):

        with open('{}_log'.format(fname), 'wb') as f:
            pickle.dump(self.log, f, -1)

        serializers.save_npz('{}_optimizer'.format(fname), self.optimizer)
        serializers.save_npz('{}_model'.format(fname), self.model)


    def report(self, fname=None):

        plt.clf()
        plt.subplot(121)
        plt.plot(self.log[('training', 'loss')], 'r', label='training')
        plt.plot(self.log[('validation', 'loss')], 'g', label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(122)
        plt.plot(self.log[('training', 'throughput')], 'r', label='training')
        plt.plot(self.log[('validation', 'throughput')], 'g', label='validation')
        plt.xlabel('epoch')
        plt.ylabel('throughput')
        plt.legend()

        if fname:
            plt.savefig(fname)
        else:
            plt.show()


class FeedforwardLearner(SupervisedLearner):

    def train(self, data):

        self.model.predictor.reset_state()

        cumloss = self.xp.zeros((), 'float32')

        loss = Variable(self.xp.zeros((), 'float32'))

        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = False
        self.model.predictor.train = True

        for _x, _t in data:

            x = Variable(self.xp.asarray(_x))
            t = Variable(self.xp.asarray(_t))

            loss = self.model(x, t)

            self.optimizer.zero_grads()
            loss.backward()
            self.optimizer.update()

        return float(loss.data)


    def test(self, data):

        loss = Variable(self.xp.zeros((), 'float32'), True)

        model = self.model

        model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        model.predictor.test = True
        model.predictor.train = False

        for _x, _t in data:
            x = Variable(self.xp.asarray(_x), True)
            t = Variable(self.xp.asarray(_t), True)

            loss += model(x, t)

        return float(loss.data / data.nbatches)


class RecurrentLearner(SupervisedLearner):

    def __init__(self, optimizer, gpu=-1, cutoff=None):
        """

        :param optimizer: Optimizer to run
        :param gpu: Run on GPU or not (-1)
        :param cutoff: cutoff length for truncated backpropagation (None=no cutoff)
        """

        super(RecurrentLearner, self).__init__(optimizer, gpu)

        self.cutoff = cutoff


    def train(self, data):

        if not self.cutoff:
            cutoff = data.nbatches
        else:
            cutoff = self.cutoff

        self.model.predictor.reset_state()

        cumloss = self.xp.zeros((), 'float32')

        loss = Variable(self.xp.zeros((), 'float32'))

        # check if we are in train or test mode (used e.g. for dropout)
        self.model.predictor.test = False
        self.model.predictor.train = True

        for _x, _t in data:

            x = Variable(self.xp.asarray(_x))
            t = Variable(self.xp.asarray(_t))

            loss += self.model(x, t)

            # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
            if data.step % cutoff == 0 or data.step == data.nbatches:

                self.optimizer.zero_grads()
                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()

                cumloss += loss.data
                loss = Variable(self.xp.zeros((), 'float32'))

        return float(cumloss / data.nbatches)


    def test(self, data):

        loss = Variable(self.xp.zeros((), 'float32'), True)

        model = self.model.copy()

        model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        model.predictor.test = True
        model.predictor.train = False

        for _x, _t in data:
            x = Variable(self.xp.asarray(_x), True)
            t = Variable(self.xp.asarray(_t), True)

            loss += model(x, t)

        return float(loss.data / data.nbatches)