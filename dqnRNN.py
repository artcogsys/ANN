import numpy as np
from chainer import cuda, Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import copy
import modelzoo

class DQN(object):
    """
    Performs neural network based Q learning

    Implement standard MLP to predict Q value for arbitrary continuous inputs. Inspired by http://localhost:8888/edit/github/DQN-chainer/dqn_agent_nature.py

    To do:
    Put online and separate out DQN code from DQNAgent and environment
    Make it work on multiple input frames
    Enable GPU

    Next models:
    RNN-based DRL : replace deep network with standard LSTM?
    NTM-based DRL
    Allow continuous actions
    Can we do planning via deep dream?
    """

    def __init__(self, ninput, noutput, **kwargs):

        self.ninput = ninput
        self.noutput = noutput

        # number of input frames to consider
        self.nframes = kwargs.get('nframes',1)

        # size of the replay buffer
        self.nbuffer = kwargs.get('nbuffer',10**4)

        # number of experiences to replay (batch size)
        self.nreplay = kwargs.get('nreplay',32)

        # discounting factor
        self.gamma = kwargs.get('gamma',0.99)

        # number of iterations for initial exploration
        self.nexplore = kwargs.get('nexplore',10**2)

        # update frequency of the target model
        self.update_freq = kwargs.get('update_freq',10**2)

        # number of hidden units
        self.nhidden = kwargs.get('nhidden',10)

        self.model = kwargs.get('model',modelzoo.MLP)
        self.model = self.model(self.ninput, self.nhidden, self.noutput)

        self.target_model = copy.deepcopy(self.model)

        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.99, eps=0.0001)
        self.optimizer.setup(self.model)

        # initialize replay memory
        self.memory = self.createBuffer(self.nbuffer)

    def createBuffer(self,n):
        """
        Create a replay buffer of size n
        """

        B = {
            'obs'  : np.zeros((n, self.nframes, self.ninput), dtype=np.float32),
            'action' : np.zeros(n, dtype=np.uint8),
            'reward' : np.zeros((n, 1), dtype=np.float32),
            'obs2' : np.zeros((n, self.nframes, self.ninput), dtype=np.float32),
            'done'   : np.zeros((n, 1), dtype=np.bool)
        }

        return B

    def getBuffer(self,idx):
        """
        Get items from buffer indicated by indices
        """

        buf_size = idx.size

        B = self.createBuffer(buf_size)

        for i in xrange(buf_size):
            B['obs'][i]  = np.asarray(self.memory['obs'][idx[i]], dtype=np.float32)
            B['action'][i] = self.memory['action'][idx[i]]
            B['reward'][i] = self.memory['reward'][idx[i]]
            B['obs2'][i] = np.array(self.memory['obs2'][idx[i]], dtype=np.float32)
            B['done'][i]   = self.memory['done'][idx[i]]

        return B

    def storeExperience(self, time, obs, action, reward, obs2, done):
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

        idx = (time - 1) % self.nbuffer # time counter is relative to obs2

        self.memory['action'][idx] = action
        self.memory['reward'][idx] = reward
        self.memory['done'][idx] = done

        if self.nframes == 1:

            self.memory['obs'][idx]  = obs
            self.memory['obs2'][idx] = obs2

        else:

            # We should iterate over nframes and then add experience in previous iterations
            # How to deal with fixed length time windows? Filler state?
            # How does the MLP with input data of these dimensions? Same for RNN

            for frame in xrange(self.nframes):

                self.memory['obs'][idx - frame, self.nframes - frame - 1] = obs
                self.memory['obs2'][idx - frame, self.nframes - frame - 1] = obs2

    def experienceReplay(self, time):
        """
        Replay experience (batch) and perform backpropagation.

        Input:
        time : iteration number

        Output:
        loss : TD error loss

        """

        if time < self.nbuffer:
            idx = np.random.randint(0, time, (self.nreplay, 1))
        else:
            idx = np.random.randint(0, self.nbuffer, (self.nreplay, 1))

        B = self.getBuffer(idx)

#                 s_replay = cuda.to_gpu(s_replay)
#                 s_dash_replay = cuda.to_gpu(s_dash_replay)

        loss = Variable(np.zeros((), 'float32'))

        # if self.GPU:
        #     loss.to_gpu()

        self.model.reset()

        for frame in xrange(self.nframes):

            # if self.GPU:
            #     t.to_gpu()
            #     x.to_gpu()

            loss += self.forward(B['obs'][:,frame], B['action'], B['reward'], B['obs2'][:,frame], B['done'])

            # learning phase
            if time >= self.nexplore and ((frame + 1) % 10 == 0 or (frame + 1) == self.nframes):
                self.optimizer.zero_grads()
                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()

        # Target model update
        if time >= self.nexplore and np.mod(time, self.update_freq) == 0:
            self.target_model = copy.deepcopy(self.model)

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

#         td = Variable(cuda.to_gpu(target)) - Q  # TD error
        td_error = Variable(target) - Q1

        # Perform TD-error clipping
        td_tmp = td_error.data + 1000.0 * (abs(td_error.data) <= 1)  # Avoid zero division
        td_clip = td_error * (abs(td_error.data) <= 1) + td_error/abs(td_tmp) * (abs(td_error.data) > 1)

        # Compute MSE of the error against zero
#        zero_val = Variable(cuda.to_gpu(np.zeros((self.nreplay, self.noutput), dtype=np.float32)))
        zero_val = Variable(np.zeros((self.nreplay, self.noutput), dtype=np.float32))
        loss = F.mean_squared_error(td_clip, zero_val)

        return loss

    def e_greedy(self, obs, epsilon = 0.05):
        """
        Epsilon greedy policy
        """

        if np.random.rand() < epsilon:
            action = np.random.randint(self.noutput)
        else:
            Q = self.model(Variable(obs)).data
            action = np.argmax(Q)

        return action
