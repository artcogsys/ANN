import numpy as np
from chainer import cuda, Chain, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import copy
import modelzoo
from circularbuffer import CircularBuffer
import matplotlib.pyplot as plt

###
# Base class for Q learning

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

        # Select random examples in the buffer
        if self.obs.full:
            nitems = self.nbuffer
        else:
            nitems = self.obs.offset+1

        idx = np.random.permutation(np.arange(self.nframes - 1, nitems))[0:_nreplay]

        # get all frames to build multiple frame observation
        # shape is nreplay x nframes
        fidx = np.array(map(lambda x: x - np.arange(self.nframes-1,-1,-1), idx))

        return self.obs.get(fidx), self.action.get(idx), self.reward.get(idx), self.obs2.get(fidx), self.done.get(idx)

    def act(self, obs, epsilon=0.1):
        """"
        Perform epsilon-greedy action.

        Input:
        epsilon: Probability of random action
        :return: action and Q value (either chainer variable or numpy array)
        """

        Q = self.getQ(obs)

        if np.random.rand() < epsilon:

            action = np.random.randint(self.noutput)

            if self.verbose:
                print 'random action: {0}'.format(action)

        else:

            if Q is None:
                action = np.random.randint(self.noutput)
            elif type(Q) is Variable:
                action = np.argmax(Q.data)
            else:
                action = np.argmax(Q)

            # analyseer actie lijst; makes sense? Waarom werkt het wel bij nframes=2 voor tabularq???
            if self.verbose:
                print 'greedy action: {0}'.format(action)

        return action, Q

    def reset(self):
        pass

###
# Tabular Q learning

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

    def getQ(self, obs):
        """
        Get Q value for all actions

        :param obs:
        :return: Q
        """

        obs = np.vstack([self.obs.get(np.arange(1, self.nframes)), obs])

        if self.verbose:
            print 'input observation: {0}'.format(obs.flatten())

        if not np.isnan(obs).any():

            entry = self.tableIndex(obs)
            return self.QTable[entry, :]

        else:
            return None

    def save(self):
        """
        Save table; not yet implemented
        """

        pass

###
# deep Q learning

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
        self.model_target = copy.deepcopy(self.model)

        # update rate of model target: model_target = tau * model + (1 - tau) * model_target
        self.tau = 10**-2

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

        if not np.isnan(obs).any():

            # Soft updating of target model
            model_params = dict(self.model.namedparams())
            model_target_params = dict(self.model_target.namedparams())
            for i in model_target_params:
                model_target_params[i].data = self.tau * model_params[i].data + (1 - self.tau) * model_target_params[i].data

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
        Q2 = self.model_target(s2)

        # Get actions that produce maximal q value
        maxQ2 = np.max(Q2.data,1)

        # Compute target q values
        target = np.copy(Q1.data)

        for i in xrange(obs.shape[0]):

            # NOTE: DQN_AGENT_NATURE uses the sign of the reward; not the reward itself as in standard Q learning!
            # Can be problematic for certain environments that e.g. only have positive rewards
            # NOTE 2: IF WE USE THIS IN TABULARQLEARNING IT ALSO FAILS; I.E. WHAT ARE THE CONSTRAINTS TO MAKE THIS WORK?
            # if not done[i]:
            #     target[i, action[i]] = np.sign(reward[i]) + self.gamma * maxQ2[i]
            # else:
            #     target[i, action[i]] = np.sign(reward[i])

            if not done[i]:
                target[i, action[i]] = reward[i] + self.gamma * maxQ2[i]
            else:
                target[i, action[i]] = reward[i]

        # Compute temporal difference error
        td_error = Variable(target) - Q1

        # Perform TD-error clipping
        # It seems to be essential to remove this!
        # td_tmp = td_error.data + 1000.0 * (abs(td_error.data) <= 1)  # Avoid zero division
        # td_clip = td_error * (abs(td_error.data) <= 1) + td_error/abs(td_tmp) * (abs(td_error.data) > 1)

        # Compute MSE of the error against zero
        zero_val = Variable(np.zeros((obs.shape[0], self.noutput), dtype=np.float32))
        loss = F.mean_squared_error(td_error, zero_val)

        return loss

    def getQ(self, obs):
        """
        Get Q value for all actions

        :param obs:
        :return: Q
        """

        if self.obs.offset is None:
            return None

        # get the nframes last observations
        if self.obs.full:
            nitems = self.nbuffer
        else:
            nitems = self.obs.offset + 1

        obs = np.vstack([self.obs.get(np.arange(nitems - self.nframes + 1, nitems)), obs])

        if self.verbose:
            print 'input observation: {0}'.format(obs.flatten())

        if not np.isnan(obs).any():
            return self.model(Variable(obs.reshape([1,self.nframes,self.ninput]))).data
        else:
            return None

    def save(self):
        """
        Save networks
        """

        serializers.save_npz('model.model', self.model)
        serializers.save_npz('model_target.model', self.model_target)

###
# Deep recurrent Q learning

class DRQN(QLearner):
    """
    Implementation of the DRQN model:
    VANILLA IMPLEMENTATION WHICH LEARNS DIRECTLY ON THE CURRENT STATE
    DRQN MIGHT REQUIRE ALL DQN TRICKS IN ORDER TO MAKE IT WORK BETTER...
    """

    def __init__(self, ninput, noutput, **kwargs):
        super(DRQN, self).__init__(ninput, noutput, **kwargs)

        # define number of hidden units
        self.nhidden = kwargs.get('nhidden',20)

        # define neural network
        self.model = kwargs.get('model', modelzoo.RNN)
        self.model = self.model(self.ninput, self.nhidden, self.noutput)

         # counter to determine truncated backprop
        self.count = 0

        # maintain loss
        self.loss = 0

        # SGD optimizer
        # self.optimizer = optimizers.Adam(alpha=0.0001, beta1=0.5)
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.model)

    def learn(self, Q1, Q2, action, reward, done):
        """
        Replay experience (batch) and perform backpropagation.

        Output:
        loss : TD error loss
        """

        # Get actions that produce maximal q value
        maxQ2 = np.max(Q2.data, 1)

        # Compute target q values
        target = np.copy(Q1.data)
        if not done:
            target[0,action] = reward + self.gamma * maxQ2
        else:
            target[0,action] = reward

        self.loss += F.mean_squared_error(Variable(target), Q1)
        self.count += 1
        if self.count % 10 == 0:
            self.optimizer.zero_grads()
            self.loss.backward()
            self.loss.unchain_backward()
            self.optimizer.update()

        return self.loss.data


    def getQ(self, obs):
        """
        Get Q value for all actions

        :param obs:
        :return: Q
        """

        if self.verbose:
            print 'input observation: {0}'.format(obs)

        if not np.isnan(obs).any():
            return self.model(Variable(obs))
        else:
            return None

    def save(self):
        """
        Save networks
        """

        serializers.save_npz('model.model', self.model)

    def reset(self):
        """
        Reset LSTM state
        """

        self.model.reset()
