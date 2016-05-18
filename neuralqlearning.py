import numpy as np
from chainer import cuda, Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import copy
import modelzoo

# CAN WE REMOVE BUFFER AND ASSUME ITS THE LAST ENTRY IN THE CIRCULAR BUFFER?
# CAN WE CREATE A SIZE 1 BUFFER FOR TABULARQ AND REMOVE learn dependence on obs,act,reward?

class CircularBuffer(object):
    """
    Circular buffer used to store replay memory
    """

    def __init__(self, nbuffer, nvars, dtype):
        """
        Create circular buffer which contains nbuffer items of length nvariables and type dtype
        """

        # size of the buffer
        self.nbuffer = nbuffer

        # nr of variables for each item
        self.nvars = nvars

        # data type of the buffer
        self.dtype = dtype

        # Initialize buffer
        self.buffer = np.zeros([self.nbuffer, self.nvars], dtype=self.dtype)

        # Index of the next element in the buffer that can be overwritten
        self.bufidx = 0

        # Flags whether the whole buffer has been filled
        self.full = False

    def put(self, item):
        """
        Store new item in buffer
        :param item
        """

        # Store experience
        self.buffer[self.bufidx, :] = item

        # Increase buffer counter
        self.bufidx += 1

        # Set buffer to filled
        if self.bufidx >= self.nbuffer:
            self.full = True

        # Index of circular memory
        self.bufidx = self.bufidx % self.nbuffer

    def get(self, n):
        """
        Get nth element in the buffer

        :param n: array of numbers
        :return:
        """

        if not self.full:

            return self.buffer[n]

        else:

            idx = map(lambda x: (x + self.bufidx) % self.nbuffer, n)

            return self.buffer[idx]

class QLearner(object):
    """
    Base class for any Q learning
    """

    def __init__(self, ninput, noutput, **kwargs):

        # Number of input variables
        self.ninput = ninput

        # Number of actions
        self.noutput = noutput

        # (maximal) number of observation frames to consider
        self.nframes = kwargs.get('nframes', 3)

        # discounting factor
        self.gamma = kwargs.get('gamma', 0.99)

        # verbose mode for debugging
        self.verbose = True

    def reset(self):
        pass

class TabularQLearner(QLearner):
    """
    Tabular Q learning
    """

    def __init__(self, observations, noutput, **kwargs):
        super(TabularQLearner, self).__init__(np.nan, noutput, **kwargs)
        """
        :param observations: list of possible observations
        :param noutput: number of possible actions
        """

        # Convert observations to nitems x nvariables
        observations = map(lambda x: tuple([item for sublist in x for item in sublist]), observations)

        # Number of input variables
        self.ninput = len(observations[0])/self.nframes

        # Number of table entries
        self.nentries = len(observations)

        # Dictionary which maps immutable observations to table entries
        self.dictionary = dict(zip(observations,np.arange(self.nentries)))

        # Q Table
        self.QTable = np.zeros([self.nentries, self.noutput], dtype=np.float32)

        # Working memory buffer to maintain last nframes observations
        self.buffer = np.empty([self.nframes, self.ninput], dtype='float32')
        self.buffer[:] = np.nan

        # Learning rate
        self.eta = 10**-3

    def learn(self, action, obs2, reward):

        if not np.isnan(self.buffer).any():

            # an experience is a vector representation of nframes observations
            experience = self.buffer.reshape(1, self.buffer.size)

            q1idx = self.dictionary[tuple(experience.tolist()[0])]
            Q1 = self.QTable[q1idx,:]

            experience2 = np.vstack([self.buffer[1:], obs2]).reshape(1, self.buffer.size)

            # Compute q values based on next obs
            q2idx = self.dictionary[tuple(experience2.tolist()[0])]
            Q2 = self.QTable[q2idx,:]

            # Get actions that produce maximal q value
            maxQ2 = np.max(Q2)

            # Compute temporal difference error
            TD_error = (reward + self.gamma * maxQ2 - self.QTable[q1idx,action])

            # Update table
            self.QTable[q1idx,action] += self.eta * TD_error

            return TD_error**2

        else:
            return np.nan

    def act(self, obs, epsilon=0.1):
        """"
        Perform epsilon-greedy action.

        Input:
        epsilon: Probability of random action
        """

        # Update buffer
        self.buffer = np.vstack([self.buffer[1:], obs])

        if np.random.rand() < epsilon:

            action = np.random.randint(self.noutput)

            # if self.verbose:
            #     print 'random action: {0}'.format(action)

        else:

            experience = self.buffer.reshape(1, self.buffer.size)

            entry = self.dictionary[tuple(experience.tolist()[0])]

            action = np.argmax(self.QTable[entry,:])

            if self.verbose:
                print 'greedy action: {0}; experience {1}'.format(action, experience)

        return action

class DQN(QLearner):
    """
    Implementation of the DQN model
    """

    def __init__(self, ninput, noutput, **kwargs):
        super(DQN, self).__init__(ninput, noutput, **kwargs)

        # size of the replay buffer D
        self.nbuffer = kwargs.get('nbuffer', 10 ** 3)

        # number of experiences to replay (batch size)
        self.nreplay = kwargs.get('nreplay', 32)

        # update frequency of the target model
        self.update_freq = kwargs.get('update_freq', 10 ** 2)

        # initialize replay memory
        self.obs = CircularBuffer(self.nbuffer, self.ninput, np.float32)
        self.action = CircularBuffer(self.nbuffer, 1, np.uint8)
        self.reward = CircularBuffer(self.nbuffer, 1, np.float32)
        self.done = CircularBuffer(self.nbuffer, 1, np.bool)

        # keep track of number of training iterations
        self.trainiter = 0

        # define model
        self.nhidden = kwargs.get('nhidden',10)
        self.model = kwargs.get('model',modelzoo.MLP)
        self.model = self.model(self.ninput*self.nframes, self.nhidden, self.noutput)

        # target model is copy of defined model
        self.target_model = copy.deepcopy(self.model)

        # SGD optimizer
        self.optimizer = optimizers.Adam(alpha=0.0001, beta1=0.5)
        self.optimizer.setup(self.model)

        # working memory buffer to maintain last nframes observations
        self.buffer = np.zeros([self.nframes, self.ninput], dtype='float32')

    def addBuffer(self, _obs, _action, _reward, _done):
        """
        Store new experience in replay buffer.

        Input:
        obs    : current obs
        action : current action
        reward : received reward
        done : whether or not the episode is done

        """

        # Store experience
        self.obs.put(_obs)
        self.action.put(_action)
        self.reward.put(_reward)
        self.done.put(_done)

    def getBuffer(self):
        """
        Get nreplay items of length self.nframes from buffer.
        """

        # Select random examples in the buffer
        idx = np.random.randint(self.nframes - 1, self.nbuffer, (self.nreplay, 1))

        # get all frames to build multiple frame observation
        fidx = map(lambda x: x - np.arange(0, self.nframes), idx)
        obs = self.obs.get(fidx).reshape(self.nreplay, self.ninput * self.nframes)

        # same for observation at next point in time
        fidx2 = map(lambda x: x + 1, fidx)
        obs2 = self.obs.get(fidx2).reshape(self.nreplay, self.ninput * self.nframes)

        return obs, self.action.get(idx), self.reward.get(idx), obs2, self.done.get(idx)

    def learn(self):
        """
        Replay experience (batch) and perform backpropagation.

        Output:
        loss : TD error loss
        """

        obs,act,reward,obs2,done = self.getBuffer()

        # Target model update
        if self.trainiter % self.update_freq == 0:

            self.target_model = copy.deepcopy(self.model)

            if self.verbose:
                print 'updating target model'

        # Gradient-based update
        self.optimizer.zero_grads()
        loss = self.forward(obs, act, reward, obs2, done)
        loss.backward()
        self.optimizer.update()

        self.trainiter += 1

        return loss.data


    def forward(self, obs, action, reward, obs2, done):
        """
        Compute loss after forward sweep

        Input:
        obs  : nbatch x nframes x nvariables (should become nbatch x nframes x npixels x npixels)
        action : nbatch,
        reward : nbatch x 1
        obs2 : next obs; nbatch x nframes x nvariables
        done   : nbatch x 1 ; flags end of an episode

        obs is assumed continuous (float32), action is assumed uint8, reward is assumed continuous (float32)

        """

        # Compute q values based on current obs
        Q1 = self.model(Variable(obs))

        # Compute q values based on next obs
        Q2 = self.target_model(Variable(obs2))

        # Get actions that produce maximal q value
        maxQ2 = np.max(Q2.data,1)

        # Compute target q values
        target = np.copy(Q1.data)
        for i in xrange(self.nreplay):

            if not done[i]:
                target[i, action[i]] = reward[i] + self.gamma * maxQ2[i]
            else:
                target[i, action[i]] = reward[i]

        # Compute temporal difference error
        td_error = Variable(target) - Q1

        # Perform TD-error clipping
        td_tmp = td_error.data + 1000.0 * (abs(td_error.data) <= 1)  # Avoid zero division
        td_clip = td_error * (abs(td_error.data) <= 1) + td_error/abs(td_tmp) * (abs(td_error.data) > 1)

        # Compute MSE of the error against zero
        zero_val = Variable(np.zeros((self.nreplay, self.noutput), dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zero_val)

        return loss

    def act(self, obs, epsilon=0.1):
        """"
        Perform epsilon-greedy action. Acting is base on nframes observations

        Input:
        epsilon: Probability of random action
        """

        # Update buffer
        self.buffer = np.vstack([self.buffer[1:], obs])

        if np.random.rand() < epsilon:

            action = np.random.randint(self.noutput)

            # if self.verbose:
            #     print 'random action: {0}'.format(action)

        else:

            experience = self.buffer.reshape(1, self.buffer.size)

            Q = self.model(Variable(experience)).data
            action = np.argmax(Q)

            if self.verbose:
                print 'greedy action: {0}; experience {1}'.format(action, experience)

        return action

    def reset(self):
        self.model.reset()