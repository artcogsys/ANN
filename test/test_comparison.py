import chainer
from environment import datasets
from models import neural_networks as models
from models.utilities import Regressor
from paradigms import supervised_learning
import models.custom_links as CL
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt

### MODEL 1

# get data
training_data = datasets.SupervisedRecurrentRegressionData(batch_size=32)
validation_data = datasets.SupervisedRecurrentRegressionData(batch_size=32)

# define model 1
nin = training_data.nin
nout = training_data.nout
model = Regressor(models.RecurrentNeuralNetwork(nin, 10, nout, link=CL.Elman, actfun=F.relu))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=100)

### MODEL 2

# get data
training_data = datasets.SupervisedRecurrentRegressionData(batch_size=32)
validation_data = datasets.SupervisedRecurrentRegressionData(batch_size=32)

# define model 2
nin = training_data.nin
nout = training_data.nout
model2 = Regressor(models.RecurrentNeuralNetwork(nin, 10, nout, link=L.LSTM, actfun=F.identity))
#model2 = Regressor(models.RecurrentNeuralNetwork(nin, 20, nout, link=CL.Elman, actfun=F.relu))

# Set up an optimizer
optimizer2 = chainer.optimizers.Adam()
optimizer2.setup(model2)
optimizer2.add_hook(chainer.optimizer.GradientClipping(5))
optimizer2.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann2 = supervised_learning.SupervisedLearner(optimizer2)

# Finally we run the optimization
ann2.optimize(training_data, validation_data=validation_data, epochs=100)

### compare losses

plt.clf()
plt.subplot(121)
plt.hold(True)
plt.plot(ann.log[('training', 'loss')], 'k', label='training 1')
plt.plot(ann.log[('validation', 'loss')], 'k--', label='validation 1')
plt.plot(ann2.log[('training', 'loss')], 'r', label='training 2')
plt.plot(ann2.log[('validation', 'loss')], 'r--', label='validation 2')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.subplot(122)
plt.hold(True)
plt.plot(ann.log[('training', 'throughput')], 'k', label='training 1')
plt.plot(ann.log[('validation', 'throughput')], 'k--', label='validation 1')
plt.plot(ann2.log[('training', 'throughput')], 'r', label='training 2')
plt.plot(ann2.log[('validation', 'throughput')], 'r--', label='validation 2')
plt.xlabel('epoch')
plt.ylabel('throughput')
plt.legend()
plt.savefig('results/comparison.png')
