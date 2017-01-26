<<<<<<< Updated upstream
import sys
sys.path.append('U:/Documents/GitHub/ANN/')

import chainer
import chainer.functions as F
import numpy as np
import scipy.io
from scipy.stats import zscore

import datasets
import models.custom_links as CL
import supervised_learning
from analysis import Analysis
from models import neural_networks as models
from models.utilities import Regressor


mat = scipy.io.loadmat('U:\Documents\Projects\Project_SingleNeuron\data_Bart.mat')
_in = np.transpose(mat['input']).astype('float32')
_out = np.transpose(mat['output']).astype('float32')

### flat input
nin = 22*22
_in = np.reshape(_in,[900,18,nin])


img_train = _in[0:800]
neu_train = _out[0:800]
img_val = _in[800:]
neu_val = _out[800:]

# get data
training_data = datasets.SupervisedData(img_train, neu_train, batch_size=50, shuffle=False)
validation_data = datasets.SupervisedData(img_val, neu_val, batch_size=50, shuffle=False)

# define model
model = Regressor(models.RecurrentNeuralNetwork(nin, 20, nout, link=CL.Elman))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=100)

# plot loss and throughput
ann.report('results/tmp')

# create analysis object
ana = Analysis(ann.model, fname='results/tmp')

# handle sequential data; deal with classifier analysis separately

# analyse data
=======
import chainer
import chainer.functions as F
import numpy as np
import scipy.io
from scipy.stats import zscore

import datasets
import models.custom_links as CL
import supervised_learning
from analysis import Analysis
from models import neural_networks as models
from models.utilities import Regressor


def normalize(x,y):
    """

    zscores x and applies the same transformation to y

    :param x: train dataset
    :param y: test dataset
    :return:
    """

    if 1:

        mu = np.mean(x, 0)
        x -= mu
        sd = np.std(x, 0)
        sd[sd == 0] = 1
        x /= sd
        y -= mu
        y /= sd

    else:

        x = zscore(x,1)
        y = zscore(y,1)


mat = scipy.io.loadmat('/Users/marcelvangerven/People/Phd/Silvan Quax/data_Bart.mat')
_in = np.transpose(mat['input']).astype('float32')
_out = np.transpose(mat['output']).astype('float32')

### flat input
nin = 18*22*22
nout = 18*20
_in = np.reshape(_in,[900,nin])
_out = np.reshape(_out,[900,nout])

img_train = _in[0:800]
neu_train = _out[0:800]
img_val = _in[800:]
neu_val = _out[800:]

# standard preprocessing
normalize(img_train, img_val)
normalize(neu_train, neu_val)

# get data
training_data = datasets.SupervisedData(img_train, neu_train, batch_size=50, shuffle=False)
validation_data = datasets.SupervisedData(img_val, neu_val, batch_size=50, shuffle=False)

# define model
model = Regressor(models.RecurrentNeuralNetwork(nin, 20, nout, link=CL.Elman))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=100)

# plot loss and throughput
ann.report('results/tmp')

# create analysis object
ana = Analysis(ann.model, fname='results/tmp')

# handle sequential data; deal with classifier analysis separately

# analyse data
>>>>>>> Stashed changes
ana.regression_analysis(validation_data.X, validation_data.T)