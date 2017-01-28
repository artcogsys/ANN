import chainer

import datasets
import supervised_learning
from analysis import Analysis
from models import neural_networks as models
from models.utilities import Classifier

# get data
training_data = datasets.StaticDataClassification(batch_size=32)
validation_data = datasets.StaticDataClassification(batch_size=32)

# define model
nin = training_data.nin
nout = training_data.nout
model = Classifier(models.DeepNeuralNetwork(nin, 10, nout, nlayer=3))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.FeedforwardLearner(optimizer)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=100)

# plot loss and throughput
ann.report('results/tmp')

# create analysis object
ana = Analysis(ann.model, fname='results/tmp')

# analyse data
ana.classification_analysis(validation_data.X, validation_data.T)