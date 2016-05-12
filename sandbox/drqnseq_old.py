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

        # size of the replay buffer D
        self.nbuffer = kwargs.get('nbuffer',10**3)

        # maximal number of frames before truncation
        self.nframes = kwargs.get('nframes', 5)

        # discounting factor
        self.gamma = kwargs.get('gamma',0.99)

        # define model
        self.nhidden = kwargs.get('nhidden',10)
        self.model = kwargs.get('model',modelzoo.RNN)
        self.model = self.model(self.ninput, self.nhidden, self.noutput)

        # define SGD optimizer
        self.optimizer = optimizers.Adam(alpha=0.0001, beta1=0.5)
        self.optimizer.setup(self.model)

        # initialize replay memory
        self.obs    = np.zeros([self.nbuffer, self.ninput], dtype=np.float32)
        self.action = np.zeros([self.nbuffer, 1], dtype=np.uint8)
        self.reward = np.zeros([self.nbuffer, 1], dtype=np.float32)
        self.done   = np.zeros([self.nbuffer, 1], dtype=np.bool)

        # probability of random policy
        self.epsilon = 0.1

        # current index of the last element in the buffer
        self.bufidx = 0


    def addBuffer(self, obs, action, reward, done):
        """
        Store new experience in replay buffer.

        Input:
        obs    : current obs
        action : current action
        reward : received reward
        done : whether or not the episode is done

        """

        self.obs[self.bufidx,:] = obs
        self.action[self.bufidx] = action
        self.reward[self.bufidx] = reward
        self.done[self.bufidx] = done

        self.bufidx += 1
        self.bufidx = self.bufidx % self.nbuffer

    def experienceReplay(self):
        """
        Replay experience (batch) and perform backpropagation.

        Output:
        loss : TD error loss

        """

        # Reset model
        self.reset()

        Q1 = None

        loss = 0
        for i in xrange(self.nbuffer):

            # Compute Q1
            Q2 = self.model(Variable(self.obs[i,:].reshape([1,self.ninput])))

            if i>0:

                # Get actions that produce maximal q value
                maxQ2 = np.max(Q2.data, 1)

                # Compute target q values
                target = np.copy(Q1.data)
                if not self.done[i]:
                    target[0,self.action[i]] = self.reward[i] + self.gamma * maxQ2
                else:
                    target[0,self.action[i]] = self.reward[i]

                # Compute temporal difference error
                td_error = Variable(target) - Q1

                # Perform TD-error clipping
                td_tmp = td_error.data + 1000.0 * (abs(td_error.data) <= 1)  # Avoid zero division
                td_clip = td_error * (abs(td_error.data) <= 1) + td_error / abs(td_tmp) * (abs(td_error.data) > 1)

                # Compute MSE of the error against zero
                zero_val = Variable(np.zeros([1, self.noutput], dtype=np.float32))
                loss += F.mean_squared_error(Variable(target), Q1)

                if i % self.nframes == 0 or i == self.nbuffer-1:

                    self.optimizer.zero_grads()
                    loss.backward()
                    loss.unchain_backward()
                    self.optimizer.update()

            Q1 = Q2

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
