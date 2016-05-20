import numpy as np
from chainer import cuda, Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import copy
import modelzoo
from circularbuffer import CircularBuffer

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
        self.nframes = kwargs.get('nframes', 2)

        # By default the buffer has the same size as the number of frames to store
        self.nbuffer = kwargs.get('nbuffer', self.nframes)
        assert(self.nbuffer >= self.nframes)

        # discounting factor
        self.gamma = kwargs.get('gamma', 0.99)

        # verbose mode for debugging
        self.verbose = True

        # initialize replay memory (not the most efficient but definitely the cleanest way)
        self.obs = CircularBuffer(self.nbuffer, self.ninput, np.float32)
        self.action = CircularBuffer(self.nbuffer, 1, np.uint8)
        self.reward = CircularBuffer(self.nbuffer, 1, np.float32)
        self.obs2 = CircularBuffer(self.nbuffer, self.ninput, np.float32)
        self.done = CircularBuffer(self.nbuffer, 1, np.bool)

    def addBuffer(self, _obs, _action, _reward, _obs2, _done):
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
        self.obs2.put(_obs2)
        self.done.put(_done)

    def getBuffer(self, _nreplay=1):
        """
        Get nreplay random experiences from the buffer
        """

        # Select random examples in the buffer
        # shape must be nreplay x 1
        idx = np.random.permutation(np.arange(self.nframes - 1, self.nbuffer))[0:_nreplay]

        # get all frames to build multiple frame observation
        # shape must be nreplay x nframes
        fidx = np.array(map(lambda x: x - np.arange(self.nframes-1,-1,-1), idx))

        return self.obs.get(fidx), self.action.get(idx), self.reward.get(idx), self.obs2.get(fidx), self.done.get(idx)

    def act(self, obs, epsilon=0.1):
        """"
        Perform epsilon-greedy action.

        Input:
        epsilon: Probability of random action
        """

        if np.random.rand() < epsilon:

            action = np.random.randint(self.noutput)

            # if self.verbose:
            #     print 'random action: {0}'.format(action)

        else:

            action = self.greedy_action(obs)

            if self.verbose:
                print 'greedy action: {0}; experience {1}'.format(action, self.getBuffer()[0].flatten())

        return action

    def reset(self):
        pass

class TabularQLearner(QLearner):
    """
    Tabular Q learning
    """

    def __init__(self, ninput, noutput, observations, **kwargs):
        super(TabularQLearner, self).__init__(ninput, noutput, **kwargs)
        """
        :param observations: list of possible observations
        :param noutput: number of possible actions
        """

        # Number of table entries
        self.nentries = len(observations)

        # Dictionary which maps immutable observations to table entries
        self.dictionary = dict(zip(observations,np.arange(self.nentries)))

        # Q Table
        self.QTable = np.zeros([self.nentries, self.noutput], dtype=np.float32)

        # Learning rate
        self.eta = 10**-3

    def tableIndex(self, obs):
        """
        Return table index of the observation; observation is a numpy array of nframes x nvariables
        :param obs:
        :return: table index
        """

        experience = obs.flatten()
        return self.dictionary[tuple(experience.tolist())]

    def learn(self):

        obs, action, reward, obs2, done = self.getBuffer()

        if not np.isnan(obs).any():

            # Compute q values based on current obs
            q1idx = self.tableIndex(obs)
            Q1 = self.QTable[q1idx,:]

            # Compute q values based on next obs
            Q2 = self.QTable[self.tableIndex(obs2),:]

            # Get actions that produce maximal q value
            maxQ2 = np.max(Q2)

            # Compute temporal difference error
            if not done:
                TD_error = reward + self.gamma * maxQ2 - self.QTable[q1idx,action]
            else:
                TD_error = reward - self.QTable[q1idx,action]

            # Update table
            self.QTable[q1idx,action] += self.eta * TD_error

            return TD_error**2

        else:
            return np.nan

    def greedy_action(self, obs):
        """
        Greedy action, returns best action according to the Q learner

        :param obs:
        :return: action
        """

        obs = np.vstack([self.obs.get(np.arange(1, self.nframes)), obs])

        if not np.isnan(obs).any():

            entry = self.tableIndex(obs)

            action = np.argmax(self.QTable[entry, :])

        else:

            action = np.random.randint(self.noutput)

        return action

class DQN(QLearner):
    """
    Implementation of the DQN model
    """

    def __init__(self, ninput, noutput, **kwargs):
        super(DQN, self).__init__(ninput, noutput, **kwargs)

        # number of experiences to replay (batch size)
        self.nreplay = kwargs.get('nreplay', np.min([self.nbuffer-self.nframes+1, 32]))

        # define number of hidden units
        self.nhidden = kwargs.get('nhidden',10)

        # define neural network
        self.model = kwargs.get('model',modelzoo.MLP)
        self.model = self.model(self.ninput*self.nframes, self.nhidden, self.noutput)

        # update frequency of the target model
        self.update_freq = kwargs.get('update_freq', 10 ** 2)

        # keep track of number of training iterations
        self.trainiter = 0

        # target model is copy of defined model
        self.target_model = copy.deepcopy(self.model)

        # SGD optimizer
        # self.optimizer = optimizers.Adam(alpha=0.0001, beta1=0.5)
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
        self.optimizer.setup(self.model)

    def learn(self):
        """
        Replay experience (batch) and perform backpropagation.

        Output:
        loss : TD error loss
        """

        obs,act,reward,obs2,done = self.getBuffer(self.nreplay)

        if not np.isnan(obs).any():

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

        else:

            return np.nan


    def forward(self, obs, action, reward, obs2, done):
        """
        Compute loss after forward sweep
        """

        # Compute q values based on current obs
        Q1 = self.model(Variable(obs.reshape([self.nreplay,self.nframes,self.ninput])))

        # Compute q values based on next obs
        Q2 = self.target_model(Variable(obs2.reshape([self.nreplay,self.nframes,self.ninput])))

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

    def greedy_action(self, obs):
        """
        Greedy action, returns best action according to the Q learner

        :param obs:
        :return: action
        """

        # get the nframes last observations
        obs = np.vstack([self.obs.get(np.arange(self.nbuffer - self.nframes + 1, self.nbuffer)), obs])

        if not np.isnan(obs).any():

            Q = self.model(Variable(obs.reshape([1,self.nframes,self.ninput]))).data
            action = np.argmax(Q)

        else:

            action = np.random.randint(self.noutput)

        return action
