import chainer
import matplotlib.pyplot as plt

from analysis import basic
from environment import datasets
from models import neural_networks as models
from models.utilities import Regressor
from paradigms import supervised_learning

# get data
[X, T, nin, nout] = datasets.get_supervised_feedforward_regression_data()

# define model
model = Regressor(models.DeepNeuralNetwork(nin, 10, nout))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
# Note: to use a model after optimization, the predict method should be used; train and test
# methods are for internal use only.
ann.optimize(X, T, epochs=100, batch_size=32)

# plot loss and throughput
plt.figure()
plt.subplot(121)
plt.plot(ann.log[('training', 'loss')], 'r', ann.log[('validation', 'loss')], 'g')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend({'training','validation'})
plt.subplot(122)
plt.plot(ann.log[('training', 'throughput')], 'r', ann.log[('validation', 'throughput')], 'g')
plt.xlabel('epoch')
plt.ylabel('throughput')
plt.legend({'training','validation'})
#plt.tight_layout()
plt.show()

#  return states
Y, H = ann.predict(X['validation'])

# perform an analysis on the optimal model
basic.scatterplot(T['validation'], Y)