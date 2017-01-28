import chainer
import chainer.functions as F
import chainer.links as L

import datasets
import models.custom_links as CL
import supervised_learning
from analysis import Analysis
from models import neural_networks as models
from models.utilities import Regressor

# get data
training_data = datasets.DynamicDataRegressionBatch()
validation_data = datasets.DynamicDataRegressionBatch()
#training_data = datasets.DynamicDataRegression(batch_size=32)
#validation_data = datasets.DynamicDataRegression(batch_size=32)

# define model
nin = training_data.nin
nout = training_data.nout
#model = Regressor(models.RecurrentNeuralNetwork(nin, 10, nout, link=L.LSTM))
model = Regressor(models.RecurrentNeuralNetwork(nin, 10, nout, link=CL.Elman))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.RecurrentLearner(optimizer, cutoff=None)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=300)

# Save model
ann.save('models/supervised_recurrent_regressor')

# plot loss and throughput
ann.report('results/tmp')

# create analysis object
ana = Analysis(ann.model, fname='results/tmp')

# handle sequential data; deal with classifier analysis separately

# analyse data
ana.regression_analysis(validation_data.X, validation_data.T)