import chainer
from environment import datasets
from models import neural_networks as models
from models.utilities import Regressor
from paradigms import supervised_learning
import models.custom_links as CL
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np

# number of training epochs
nepochs = 50

# number of repetitions of the comparison
nreps = 5

# number of models/optimizers to compare
nmodels = 2

# colors associated with each models plot
colors = ['k', 'r']

# get data
training_data = datasets.SupervisedRecurrentRegressionData(batch_size=32)
validation_data = datasets.SupervisedRecurrentRegressionData(batch_size=32)
nin = training_data.nin
nout = training_data.nout

training_loss = {}
validation_loss = {}
training_throughput = {}
validation_throughput = {}

for i in range(nmodels):

    training_loss[i] = np.zeros([nreps,nepochs])
    validation_loss[i] = np.zeros([nreps,nepochs])
    training_throughput[i] = np.zeros([nreps,nepochs])
    validation_throughput[i] = np.zeros([nreps,nepochs])

    for j in range(nreps):

        # reset datasets
        training_data.reset()
        validation_data.reset()

        # define model
        if i==0:
            model = Regressor(models.RecurrentNeuralNetwork(nin, 10, nout, link=CL.Elman, actfun=F.relu))
        else:
            model = Regressor(models.RecurrentNeuralNetwork(nin, 10, nout, link=L.LSTM, actfun=F.identity))

        # Set up optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(5))
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

        ann = supervised_learning.SupervisedLearner(optimizer)

        # Finally we run the optimization
        ann.optimize(training_data, validation_data=validation_data, epochs=nepochs)

        training_loss[i][j,:] = ann.log[('training', 'loss')]
        validation_loss[i][j,:] = ann.log[('validation', 'loss')]
        training_throughput[i][j,:] = ann.log[('training', 'throughput')]
        validation_throughput[i][j,:] = ann.log[('validation', 'throughput')]

### compare losses

plt.clf()

plt.subplot(121)
plt.hold(True)
x = np.arange(nepochs)
for i in range(nmodels):
    plt.errorbar(x,np.mean(training_loss[i],axis=0), yerr=np.std(training_loss[i],axis=0)/np.sqrt(nreps), fmt='-', color=colors[i], ecolor=colors[i], label='training' + str(i))
    plt.errorbar(x,np.mean(validation_loss[i],axis=0), yerr=np.std(validation_loss[i],axis=0)/np.sqrt(nreps), fmt='--', color=colors[i], ecolor=colors[i], label='validation' + str(i))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.subplot(122)
plt.hold(True)
for i in range(nmodels):
    plt.errorbar(x,np.mean(training_throughput[i],axis=0), yerr=np.std(training_throughput[i],axis=0)/np.sqrt(nreps), fmt='-', color=colors[i], ecolor=colors[i], label='training' + str(i))
    plt.errorbar(x,np.mean(validation_throughput[i],axis=0), yerr=np.std(validation_throughput[i],axis=0)/np.sqrt(nreps), fmt='--', color=colors[i], ecolor=colors[i], label='validation' + str(i))
plt.xlabel('epoch')
plt.ylabel('throughput')
plt.legend()
plt.savefig('results/comparison.png')
