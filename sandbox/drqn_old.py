import numpy as np
from chainer import cuda, Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import copy
import modelzoo

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

        # size of the replay buffer D
        self.nbuffer = kwargs.get('nbuffer',10**4)

        # number of episodes to replay (batch size)
        self.nreplay = kwargs.get('nreplay', 256)

        # length of each episode
        self.nframes = kwargs.get('nframes', 10)

        # discounting factor
        self.gamma = kwargs.get('gamma',0.99)

        # update frequency of the target model
        self.update_freq = kwargs.get('update_freq',10**2)

        # define model
        self.nhidden = kwargs.get('nhidden',10)
        self.model = kwargs.get('model',modelzoo.RNN)
        self.model = self.model(self.ninput, self.nhidden, self.noutput)

        # target model is copy of defined model
        self.target_model = copy.deepcopy(self.model)

        # define SGD optimizer
        self.optimizer = optimizers.Adam(alpha=0.00001, beta1=0.5)
        self.optimizer.setup(self.model)

        # initialize replay memory
        self.D = self.createBuffer(self.nbuffer)

        # Iteration number
        self.time = 0

        # probability of random policy
        self.epsilon = 0.1

        # buffer index
        self.bufidx = 0

    def createBuffer(self,n):
        """
        Create a replay buffer of size n
        """

        B = {
            'obs'   : np.zeros((n, self.ninput), dtype=np.float32),
            'action': np.zeros(n, dtype=np.uint8),
            'reward': np.zeros((n, 1), dtype=np.float32),
            'obs2'  : np.zeros((n, self.ninput), dtype=np.float32),
            'done'  : np.zeros((n, 1), dtype=np.bool)
        }

        return B

    def addBuffer(self, obs, action, reward, obs2, done):
        """
        Store new experience in replay buffer.

        Input:
        time : the iteration number
        obs  : current obs
        action : current action
        reward : received reward
        obs2 : next obs
        done   : flags end of an episode

        """

        idx = self.bufidx % self.nbuffer

        self.D['obs'][idx]    = obs
        self.D['action'][idx] = action
        self.D['reward'][idx] = reward
        self.D['obs2'][idx]   = obs2
        self.D['done'][idx]   = done

        self.bufidx += 1

    def getBuffer(self):
        """
        Get nreplay episodes of length nframes from buffer
        """

        n = self.nreplay
        m = self.nframes

        B = {
            'obs'   : np.zeros([n, m, self.ninput], dtype=np.float32),
            'action': np.zeros([n, m], dtype=np.uint8),
            'reward': np.zeros([n, m], dtype=np.float32),
            'obs2'  : np.zeros([n, m, self.ninput], dtype=np.float32),
            'done'  : np.zeros([n, m], dtype=np.bool)
        }

        # index of the first element
        sidx = self.bufidx % self.nbuffer

        for i in xrange(n):

            # define episode
            idx = np.random.randint(0, self.nbuffer - self.nframes)
            episode = np.arange(sidx+idx,sidx+(idx+self.nframes)) % self.nbuffer

            B['obs'][i]    = self.D['obs'][episode].reshape([1, m, self.ninput])
            B['action'][i] = self.D['action'][episode].reshape([1, m])
            B['reward'][i] = self.D['reward'][episode].reshape([1, m])
            B['obs2'][i]   = self.D['obs2'][episode].reshape([1, m, self.ninput])
            B['done'][i]   = self.D['done'][episode].reshape([1, m])

        return B


    def experienceReplay(self):
        """
        Replay experience (batch) and perform backpropagation.

        Output:
        loss : TD error loss

        """

        # Select random episodes in the buffer
        B = self.getBuffer()

        # Target model update
        if self.time % self.update_freq == 0:
            self.target_model = copy.deepcopy(self.model)

        self.model.reset()

        # target model takes one step
        self.target_model.reset()
        self.target_model(Variable(B['obs'][:, 0]))

        self.optimizer.zero_grads()

        loss = 0
        for i in xrange(self.nframes):

            loss += self.forward(B['obs'][:, i], B['action'][:, i], B['reward'][:, i], B['obs2'][:, i], B['done'][:, i])

        loss.backward()
        self.optimizer.update()

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

    def act(self, obs):

        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.noutput)
        else:
            Q = self.model(Variable(obs)).data
            action = np.argmax(Q)

        return action

    def reset(self):
        """
        Reset state of neural network
        """
        self.model.reset()
