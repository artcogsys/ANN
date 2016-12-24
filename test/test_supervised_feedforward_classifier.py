import chainer
import chainer.links as L
import matplotlib.pyplot as plt

from environment import datasets
from models import supervised_learning_models as models
from paradigms import supervised_learning

# get data
#[X, T, nin, nout] = datasets.get_mnist()
[X, T, nin, nout] = datasets.get_supervised_feedforward_classification_data()

# define model
model = L.Classifier(models.DNN(nin, 10, nout))

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
# Note: to use a model after optimization, the predict method should be used; train and test
# methods are for internal use only.
ann.optimize(X, T, epochs=20, batch_size=200)

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