import numpy as np
from chainer import cuda, Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import copy
import modelzoo
from circularbuffer import CircularBuffer
import matplotlib.pyplot as plt


# def unique_rows(data):
#     sorted_data =  data[np.lexsort(data.T),:]
#     return np.append([True],np.any(np.diff(sorted_data,axis=0),1))

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
        self.verbose = kwargs.get('verbose', True)

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

        idx = np.random.permutation(np.arange(self.nframes - 1, self.obs.buffer_size()))[0:_nreplay]

        # get all frames to build multiple frame observation
        # shape is nreplay x nframes
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

            if self.verbose:
                print 'random action: {0}'.format(action)

        else:

            action = self.greedy_action(obs)

            if self.verbose:
                print 'greedy action: {0}'.format(action)

        return action

    def reset(self):
        pass

    def getExperience(self,obs):
        """
        get experience defined as last nframes-1 observations and the final observation

        :param obs:
        :return: experience
        """

        # index of the last element in the buffer
        idx = self.obs.idx

        if idx >= self.nframes + 2:
            return np.vstack([self.obs.get(np.arange(idx - self.nframes + 2, idx+1)), obs])
        else:
            return np.nan

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

        experience = self.getExperience(obs)

        if self.verbose:
            print 'input observation: {0}'.format(experience.flatten())

        if not np.isnan(experience).any():

            entry = self.tableIndex(experience)

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
        self.nhidden = kwargs.get('nhidden',20)

        # define neural network
        self.model = kwargs.get('model',modelzoo.MLP)
        self.model = self.model(self.ninput*self.nframes, self.nhidden, self.noutput)

        # target model is copy of defined model
        self.target_model = copy.deepcopy(self.model)

        # update rate of target model: target_model = tau * model + (1 - tau) * target_model
        self.tau = 0.01

        # SGD optimizer
        # self.optimizer = optimizers.Adam(alpha=0.0001, beta1=0.5)
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.model)

    def learn(self):
        """
        Replay experience (batch) and perform backpropagation.

        Output:
        loss : TD error loss
        """

        obs,act,reward,obs2,done = self.getBuffer(self.nreplay)

        if not np.isnan(obs).any() and not obs.size == 0:

            # Soft updating of target model
            model_params = dict(self.model.namedparams())
            target_model_params = dict(self.target_model.namedparams())
            for i in target_model_params:
                target_model_params[i].data = self.tau * model_params[i].data + (1 - self.tau) * target_model_params[i].data

            # Gradient-based update
            self.optimizer.zero_grads()
            loss = self.forward(obs, act, reward, obs2, done)
            loss.backward()
            self.optimizer.update()

            return loss.data

        else:

            return np.nan


    def forward(self, obs, action, reward, obs2, done):
        """
        Compute loss after forward sweep
        """

        # Compute q values based on current obs
        s = Variable(obs) # obs.reshape([self.nreplay,self.nframes,self.ninput]))
        Q1 = self.model(s)

        # Compute q values based on next obs
        s2 = Variable(obs2) # obs2.reshape([self.nreplay,self.nframes,self.ninput]))
        Q2 = self.target_model(s2)

        # Get actions that produce maximal q value
        maxQ2 = np.max(Q2.data,1)

        # Compute target q values
        target = np.copy(Q1.data)

        for i in xrange(obs.shape[0]):

            # NOTE: DQN_AGENT_NATURE uses the sign of the reward; not the reward itself as in standard Q learning!
            # Can be problematic for certain environments that e.g. only have positive rewards
            if not done[i]:
                target[i, action[i]] = np.sign(reward[i]) + self.gamma * maxQ2[i]
            else:
                target[i, action[i]] = np.sign(reward[i])

        # Compute temporal difference error
        td_error = Variable(target) - Q1

        # Perform TD-error clipping
        td_tmp = td_error.data + 1000.0 * (abs(td_error.data) <= 1)  # Avoid zero division
        td_clip = td_error * (abs(td_error.data) <= 1) + td_error/abs(td_tmp) * (abs(td_error.data) > 1)

        # Compute MSE of the error against zero
        zero_val = Variable(np.zeros((obs.shape[0], self.noutput), dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zero_val)

        return loss

    def greedy_action(self, obs):
        """
        Greedy action, returns best action according to the Q learner

        :param obs:
        :return: action
        """

        # get the nframes last observations
        experience = getExperience(obs)

        if self.verbose:
            print 'input observation: {0}'.format(experience.flatten())

        if not np.isnan(experience).any():

            Q = self.model(Variable(experience.reshape([1,self.nframes,self.ninput]))).data
            action = np.argmax(Q)

        else:

            action = np.random.randint(self.noutput)

        return action
