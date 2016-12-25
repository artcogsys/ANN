from chainer import Variable, cuda, serializers
import numpy as np
import pickle
import time
import tqdm

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


    def optimize(self, X, T, epochs=50, batch_size=32, cutoff=10):

        # keep track of minimal validation loss
        min_loss = float('nan')

        for epoch in tqdm.tqdm(xrange(self.optimizer.epoch, self.optimizer.epoch + epochs)):

            then = time.time()
            loss = self.train(X['training'], T['training'], batch_size, cutoff)
            now = time.time()
            throughput = T['training'].shape[0] / (now - then)

            self.log[('training', 'loss')].append(loss)
            self.log[('training', 'throughput')].append(throughput)

            if 'validation' in T:
                then = time.time()
                loss = self.test(X['validation'], T['validation'], batch_size)
                now = time.time()
                throughput = T['validation'].shape[0] / (now - then)
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

    def load(self, postfix, prefix):
        with open('{}log{}'.format(prefix, postfix), 'rb') as f:
            self.log = pickle.load(f)

        serializers.load_npz('{}optimizer{}'.format(prefix, postfix), self.optimizer)
        serializers.load_npz('{}model{}'.format(prefix, postfix), self.model)


    def save(self, postfix, prefix):
        with open('{}log{}'.format(prefix, postfix), 'wb') as f:
            pickle.dump(self.log, f, -1)

        serializers.save_npz('{}optimizer{}'.format(prefix, postfix), self.optimizer)
        serializers.save_npz('{}model{}'.format(prefix, postfix), self.model)


    def train(self, X, T, batch_size, cutoff):

        # required?
        self.model.predictor.reset_state()

        cumloss = self.xp.zeros((), 'float32')

        loss = Variable(self.xp.zeros((), 'float32'))

        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = False
        self.model.predictor.train = True

        steps = len(X) // batch_size

        if self.model.predictor.type == 'feedforward':

            # processing of random batches
            perm = np.random.permutation(np.arange(len(X)))
            for step in xrange(steps):

                x = Variable(self.xp.asarray([X[perm[(seq * steps + step) % len(X)]] for seq in xrange(batch_size)]))
                t = Variable(self.xp.asarray([T[perm[(seq * steps + step) % len(T)]] for seq in xrange(batch_size)]))

                loss = self.model(x, t)

                self.optimizer.zero_grads()
                loss.backward()
                self.optimizer.update()

            return float(loss.data)

        elif self.model.predictor.type == 'recurrent':

            # processing of sequences
            for step in xrange(steps):

                # uncertain about batch mode here
                x = Variable(self.xp.asarray([X[(seq * steps + step) % len(X)] for seq in xrange(batch_size)]))
                t = Variable(self.xp.asarray([T[(seq * steps + step) % len(T)] for seq in xrange(batch_size)]))

                loss += self.model(x, t)

                if (step + 1) % cutoff == 0 or (step + 1) == steps:
                    self.optimizer.zero_grads()
                    loss.backward()
                    loss.unchain_backward()
                    self.optimizer.update()

                    cumloss += loss.data
                    loss = Variable(self.xp.zeros((), 'float32'))

            return float(cumloss / steps)

        else:
            raise ValueError('unknown type')


    def test(self, X, T, batch_size):

        loss = Variable(self.xp.zeros((), 'float32'), True)

        if self.model.predictor.type == 'feedforward':
            model = self.model
        elif self.model.predictor.type == 'recurrent':
            model = self.model.copy()
        else:
            raise ValueError('unknown type')

        model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        model.predictor.test = True
        model.predictor.train = False

        steps = len(X) // batch_size

        if self.model.predictor.type == 'feedforward':

            # processing of random batches
            perm = np.random.permutation(np.arange(len(X)))
            for step in xrange(steps):
                x = Variable(self.xp.asarray([X[perm[(seq * steps + step) % len(X)]] for seq in xrange(batch_size)]),
                             True)
                t = Variable(self.xp.asarray([T[perm[(seq * steps + step) % len(T)]] for seq in xrange(batch_size)]),
                             True)

                loss += model(x, t)

        elif self.model.predictor.type == 'recurrent':

            # processing of sequences
            for step in xrange(steps):
                x = Variable(self.xp.asarray([X[(seq * steps + step) % len(X)] for seq in xrange(batch_size)]), True)
                t = Variable(self.xp.asarray([T[(seq * steps + step) % len(T)] for seq in xrange(batch_size)]), True)

                loss += model(x, t)

        return float(loss.data / steps)

    def predict(self, X):

        self.model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = True
        self.model.predictor.train = False

        Y = []
        for step in xrange(X.shape[0]):

            x = Variable(self.xp.asarray(X[step][None]), True)
            Y.append(self.model.predictor(x).data)

            if step == 0:
                H = [[self.model.predictor.h[i].data[0]] for i in xrange(len(self.model.predictor.h))]
            else:
                _ = [H[i].append(self.model.predictor.h[i].data[0]) for i in xrange(len(self.model.predictor.h))]

        H = [self.xp.asarray(H[i]) for i in xrange(len(H))]
        Y = np.squeeze(self.xp.asarray(Y))

        return Y, H