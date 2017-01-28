import chainer
import chainer.functions as F
import chainer.links as L

import datasets
import supervised_learning
import models.custom_links as CL
from analysis import Analysis
from models import neural_networks as models
from models.utilities import Classifier

# get data
training_data = datasets.DynamicDataClassification(batch_size=32)
validation_data = datasets.DynamicDataClassification(batch_size=32)

# define model
nin = training_data.nin
nout = training_data.nout
#model = Classifier(models.RecurrentNeuralNetwork(nin, 10, nout, link=L.LSTM))
model = Classifier(models.RecurrentNeuralNetwork(nin, 10, nout, link=CL.Elman))

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

# analyse data
ana.classification_analysis(validation_data.X, validation_data.T)