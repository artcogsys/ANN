import numpy as np
from chainer import cuda, Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import copy
import modelzoo

class NeuralQAgent(object):
    """
    Base class for neural-network based Q learning
    """

    def __init__(self, ninput, noutput, **kwargs):

        self.ninput = ninput
        self.noutput = noutput

        # size of the replay buffer D
        self.nbuffer = kwargs.get('nbuffer', 10**3)

        # number of experiences to replay (batch size)
        self.nreplay = kwargs.get('nreplay',32)

        # (maximal) number of frames to consider
        self.nframes = kwargs.get('nframes', 2)

        # discounting factor
        self.gamma = kwargs.get('gamma', 0.99)

        # update frequency of the target model
        self.update_freq = kwargs.get('update_freq',10**2)

        # initialize replay memory
        self.obs = np.zeros([self.nbuffer, self.ninput], dtype=np.float32)
        self.action = np.zeros([self.nbuffer, 1], dtype=np.uint8)
        self.reward = np.zeros([self.nbuffer, 1], dtype=np.float32)
        self.done = np.zeros([self.nbuffer, 1], dtype=np.bool)

        # current index of the last element in the buffer
        self.bufidx = -1

        # keep track of number of training iterations
        self.trainiter = 0

        # verbose mode for debugging
        self.verbose = True

    def addBuffer(self, obs, action, reward, done):
        """
        Store new experience in replay buffer.

        Input:
        obs    : current obs
        action : current action
        reward : received reward
        done : whether or not the episode is done

        """

        # Increase buffer counter
        self.bufidx += 1

        # Index of circular memory
        self.bufidx = self.bufidx % self.nbuffer

        # Store experience
        self.obs[self.bufidx, :] = obs
        self.action[self.bufidx] = action
        self.reward[self.bufidx] = reward
        self.done[self.bufidx] = done


class DQN(NeuralQAgent):

    def __init__(self, ninput, noutput, **kwargs):
        super(DQN, self).__init__(ninput, noutput, **kwargs)

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

    def getBuffer(self):
        """
        Get nreplay items from buffer. Agent-specific.
        """

        # Select random examples in the buffer
        idx = np.random.randint(self.nframes-1, self.nbuffer-1, (self.nreplay, 1))

        # get all frames to build multiple frame observation
        fidx = map(lambda x: x - np.arange(0, self.nframes), idx)
        obs = self.obs[fidx].reshape(self.nreplay, self.ninput * self.nframes)

        # same for observation at next point in time
        fidx2 = map(lambda x: x + 1,fidx)
        obs2 = self.obs[fidx].reshape(self.nreplay, self.ninput * self.nframes)

        return obs, self.action[idx], self.reward[idx], obs2, self.done[idx]

    def experienceReplay(self):
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
        Perform epsilon-greedy action.

        Input:
        epsilon: Probability of random action
        """

        # Update buffer
        self.buffer = np.vstack([self.buffer[1:], obs])

        if np.random.rand() < epsilon:

            action = np.random.randint(self.noutput)

            if self.verbose:
                print 'random action: {0}'.format(action)

        else:

            experience = self.buffer.reshape(1, self.buffer.size)

            Q = self.model(Variable(experience)).data
            action = np.argmax(Q)

            if self.verbose:
                print 'greedy action: {0}; experience {1}'.format(action, experience)

        return action

    def reset(self):
        self.model.reset()