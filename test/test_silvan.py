import chainer
import numpy as np
import scipy.io

import datasets
from chainer.functions.activation import sigmoid
import models.custom_links as CL
import supervised_learning
from analysis import Analysis
from models import neural_networks as models
from models.utilities import Regressor


mat = scipy.io.loadmat('/Users/marcelvangerven/People/Phd/Silvan Quax/data_Bart.mat')
_in = np.transpose(mat['input']).astype('float32')
_out = np.transpose(mat['output']).astype('float32')

nin = 22*22 # number of pixels
nout = 20 # number of electrodes

### flat input
_in = np.reshape(_in,[900,18,nin])

img_train = _in[0:800]
neu_train = _out[0:800]
img_val = _in[800:]
neu_val = _out[800:]

# get data
training_data = datasets.DynamicData(img_train, neu_train, batch_size=50)
validation_data = datasets.DynamicData(img_val, neu_val, batch_size=50)

# define model
model = Regressor(models.RecurrentNeuralNetwork(nin, 20, nout, link=CL.Elman))

# overload activation function of Elman layer
model.predictor[0][0].actfun = sigmoid.sigmoid

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.RecurrentLearner(optimizer, cutoff=10)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=100)

# plot loss and throughput
ann.report('results/tmp')

# create analysis object
ana = Analysis(ann.model, fname='results/tmp')

# create analysis object
ana = Analysis(ann.model, fname='results/tmp')

# analyse data
ana.regression_analysis(validation_data.X, validation_data.T)