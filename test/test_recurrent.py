import chainer
import chainer.links as L
import matplotlib.pyplot as plt

from environment import datasets
from models import recurrent
from paradigms import supervised
from analysis import connectivity

# get data
[X, T, nin, nout] = datasets.get_random_timeseries()

# define model
model = L.Classifier(recurrent.RNN(nin, 10, nout))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised.RecurrentNeuralNetwork(optimizer)

# Finally we run the optimization
# Note: to use a model after optimization, the predict method should be used; train and test
# methods are for internal use only.
ann.optimize(X, T)

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
data = [X['validation']]
[data.append(H[i]) for i in range(len(H))]
data.append(Y)
connectivity.functional_connectivity(data)