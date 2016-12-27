import chainer
from analysis import Analysis
from environment import datasets
from models import neural_networks as models
from paradigms import supervised_learning
from models.utilities import Classifier

# get data
training_data = datasets.SupervisedFeedforwardClassificationData(batch_size=32)
validation_data = datasets.SupervisedFeedforwardClassificationData(batch_size=32)

# define model
nin = training_data.nin
nout = training_data.nout
model = Classifier(models.DeepNeuralNetwork(nin, 10, nout))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=100)

# plot loss and throughput
ann.report('tmp')

# create analysis object
ana = Analysis(ann.model, fname='tmp')

# analyse data
ana.classification_analysis(validation_data.X, validation_data.T)