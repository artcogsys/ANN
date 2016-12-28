import chainer
from analysis import Analysis
from environment import datasets
from models import neural_networks as models
from models.utilities import Classifier
from paradigms import supervised_learning

# get data
training_data = datasets.MNISTData(validation=False, batch_size=32)
validation_data = datasets.MNISTData(validation=True, batch_size=32)

# define model
nin = training_data.nin
nout = training_data.nout
model = Classifier(models.ConvNet(nin, 10, nout))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

# instantiate learning algorithm
ann = supervised_learning.SupervisedLearner(optimizer)

# run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=100)

# plot loss and throughput
ann.report('results/tmp')

# create analysis object
ana = Analysis(ann.model, fname='results/tmp')

# handle sequential data; deal with classifier analysis separately

# analyse data
ana.classification_analysis(validation_data.X, validation_data.T)