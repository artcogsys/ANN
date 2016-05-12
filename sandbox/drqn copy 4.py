import numpy as np
from chainer import cuda, Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import copy
import modelzoo
from collections import deque

# Question is whether deque is the best option. Impossible to process minibatches efficiently. Think about how data comes in
# and what should be solved for. Use standard RNN sequence to think about this.
# Maybe store everything a numpy arrays; properly handle buffer updating
# cant we just provide a huge sequence and do truncated backprop???

class DRQN(object):
    """
    Implementation of deep recurrent q network

    Hausknecht, M., Stone, P., 2015. Deep Recurrent Q-Learning for Partially Observable MDPs. arXiv Prepr. arXiv1507.06527.

    DRQN takes a RNN as underlying model. It should have the ability to deal with fully connected inputs (RQN) or convolutional inputs (DRQN)

    Data is stored as one long sequence.

    To do:
    - store separate episodes
    - put back in acting based on RNN


    """

    def __init__(self, ninput, noutput, **kwargs):

        self.ninput = ninput
        self.noutput = noutput

        # size of the replay buffer D (number of random length episodes to store)
        self.nbuffer = kwargs.get('nbuffer',10**3)

        # maximal length of each episode
        self.nframes = kwargs.get('nframes', 3)

        # number of episodes to replay (batch size)
        self.nreplay = kwargs.get('nreplay', 32)

        # discounting factor
        self.gamma = kwargs.get('gamma',0.99)

        # update frequency of the target model
        self.update_freq = kwargs.get('update_freq',10**2)

        # define model
        self.nhidden = kwargs.get('nhidden',100)
        self.model = kwargs.get('model',modelzoo.RNN)
        self.model = self.model(self.ninput, self.nhidden, self.noutput)

        # target model is copy of defined model
        self.target_model = copy.deepcopy(self.model)

        # define SGD optimizer
        self.optimizer = optimizers.Adam(alpha=0.00001, beta1=0.5)
        self.optimizer.setup(self.model)

        # initialize replay memory
        self.obs    = deque(maxlen=self.nbuffer)
        self.action = deque(maxlen=self.nbuffer)
        self.reward = deque(maxlen=self.nbuffer)

        # probability of random policy
        self.epsilon = 0.1

        # index of current episode
        self.bufidx = None

        # index of current frame in current episode
        self.frameidx = 0

    def addBuffer(self, obs, action, reward, done):
        """
        Store new experience in replay buffer.

        Input:
        obs    : current obs
        action : current action
        reward : received reward

        """

        # handling of first episode
        if not self.bufidx:

            self.obs.append(np.zeros([self.nframes, self.ninput], dtype=np.float32))
            self.action.append(np.zeros([self.nframes, 1], dtype=np.uint8))
            self.reward.append(np.zeros([self.nframes, 1], dtype=np.float32))

            self.bufidx = 0

        elif done:

            # redefine size of previous episode
            self.obs[-1] = self.obs[-1][0:self.frameidx,:]
            self.action[-1] = self.action[-1][0:self.frameidx,:]
            self.reward[-1] = self.reward[-1][0:self.frameidx, :]

            # define new episode
            self.obs.append(np.nan([self.nframes, self.ninput]), dtype=np.float32)
            self.action.append(np.nan([self.nframes, 1]), dtype=np.uint8)
            self.reward.append(np.nan([self.nframes, 1]), dtype=np.float32)

            # increase episodeidx
            self.bufidx += 1

            # reset frameidx
            self.frameidx=0

        # add new experience
        self.obs[-1][self.frameidx,:] = obs
        self.action[-1][self.frameidx] = action
        self.reward[-1][self.frameidx] = reward

        # increase frameidx
        self.frameidx += 1

    def getBuffer(self):
        """
        Get nreplay episodes from buffer
        """

        # generate nreplay episode indices
        idx = np.random.randint(0, self.bufidx+1, [self.nreplay, 1])

        return self.obs[idx], self.action[idx], self.reward[idx]

    def experienceReplay(self):
        """
        Replay experience (batch) and perform backpropagation.

        Output:
        loss : TD error loss

        """

        # Select random episodes in the buffer
        obs,actions,rewards = self.getBuffer()

        # Target model update (is this really a deep copy?)
        if self.time % self.update_freq == 0:
            self.target_model = copy.deepcopy(self.model)

        # Gradient-based update
        loss = Variable(np.zeros((), 'float32'))

        # Reset current model and target model
        self.model.reset()
        self.target_model.reset()

        # take nframes-1 steps to process observations and compute Q1
        eidx = self.nframes-1
        for i in xrange(eidx):
            Q1 = self.model(Variable(B['obs'][:, i]))
            self.target_model(Variable(B['obs'][:, i]))

        # get selected action, reward and done state
        actions = B['action'][:, eidx - 1]
        rewards = B['reward'][:, eidx - 1]
        dones = B['done'][:, eidx - 1]

        # compute Q2 based on next observation
        Q2 = self.target_model(Variable(B['obs'][:, eidx]))

        # Get actions that produce maximal q value
        maxQ2 = np.max(Q2.data, 1)

        # Compute target q values
        target = np.copy(Q1.data)
        for i in xrange(self.nreplay):
            if not dones[i]:
                target[i, actions[i]] = rewards[i] + self.gamma * maxQ2[i]
            else:
                target[i, actions[i]] = rewards[i]

        # Compute temporal difference error
        td_error = Variable(target) - Q1

        # Perform TD-error clipping
        td_tmp = td_error.data + 1000.0 * (abs(td_error.data) <= 1)  # Avoid zero division
        td_clip = td_error * (abs(td_error.data) <= 1) + td_error / abs(td_tmp) * (abs(td_error.data) > 1)

        # Compute MSE of the error against zero
        zero_val = Variable(np.zeros((self.nreplay, self.noutput), dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zero_val)

        self.optimizer.zero_grads()
        loss.backward()
        self.optimizer.update()

        return loss.data


    def act(self, obs):

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.noutput)
            qvalue = np.nan
        else:
            Q = self.model(Variable(obs)).data
            action = np.argmax(Q)
            qvalue = np.max(Q)

        return action, qvalue

    def reset(self):
        """
        Reset state of neural network
        """
        self.model.reset()
