from chainer import Chain
import chainer.links as L
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# define model
class JordyNet(Chain):
    """
    Basic convolutional neural network
    """

    def __init__(self, ksize=20, sigma=10):

        stride = 1
        pad = ksize//2

        super(JordyNet, self).__init__(
            l1=L.Convolution2D(1, 1, ksize, stride, pad),
        )

        # set gaussian filters
        for i in range((ksize-1)/2,1+(ksize-1)/2):
            for j in range((ksize-1)/2,1+(ksize-1)/2):
                self.l1.W.data[0,0, i, j] = ss.multivariate_normal.pdf([i,j], [0, 0], [[sigma, 0], [0, sigma]])
                self.l1.W.data[0,0,:,:] /= np.sum(self.l1.W.data[0,0,:,:].flatten())
        self.h = {}

    def __call__(self, x):
        """
        :param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])
        """

        y = self.l1(x)

        return y

    def reset_state(self):
        pass


model = JordyNet()

# create random image
img = np.random.rand(1,1,100,100).astype('float32')

x = model(img).data.squeeze()

plt.subplot(121)
plt.imshow(img.squeeze())
plt.subplot(122)
plt.imshow(x)
plt.show()


