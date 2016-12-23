import numpy as np
import os
import chainer.links as L
from chainer import training
from chainer import datasets, iterators, optimizers
from chainer.training import extensions
from chainer.serializers import npz
from models import feedforward as ff
import matplotlib.pyplot as plt

# Also see:
# https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py

# get file name
directory = os.path.splitext(os.path.basename(__file__))[0]
if not os.path.exists(directory):
    os.makedirs(directory)

nepochs = 10 # number of epochs
nbatch = 100 # number of items per batch
nhidden = 10 # number of hidden units

# get train and validation data as TupleDatasets
train, validation = datasets.get_mnist()

# infer input and output size
ninput = train._datasets[0].shape[1]
noutput = np.unique(train._datasets[1]).size

# MultiprocessIterator could be used to prefetch (large) batch sizes
train_iter = iterators.SerialIterator(train, batch_size=nbatch, repeat=True, shuffle=True)
validation_iter = iterators.SerialIterator(validation, batch_size=nbatch, repeat=False, shuffle=False)

# For classification, we use cross-entropy as the cost function
model = L.Classifier(ff.DNN(ninput, nhidden, noutput))

optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer) # ParallelUpdater for GPU
trainer = training.Trainer(updater, (nepochs, 'epoch'), out=directory)

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(validation_iter, model))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot of the training object at the end; we probably don't care too much
#trainer.extend(extensions.snapshot(), trigger=(nepochs, 'epoch'))

# Take snapshot of model at each epoch
trainer.extend(extensions.snapshot_object(model, 'model_{.updater.epoch}', trigger=(1, 'epoch')))

# Print a progress bar to stdout
# trainer.extend(extensions.ProgressBar())

# run training
trainer.run()

# read log file
import json
with open(directory + '/log') as data_file:

    data = json.load(data_file)

    # extract validation loss and accuracy
    loss = map(lambda x: x['validation/main/loss'], data)
    accuracy = map(lambda x: x['validation/main/accuracy'], data)

# print loss and accuracy
plt.figure()
plt.subplot(121)
plt.plot(range(len(loss)),loss)
plt.title('loss')
plt.subplot(122)
plt.plot(range(len(accuracy)),accuracy)
plt.title('accuracy')
plt.show()

# determine best model based on validation error
idx = np.argmin(loss)

# get snapshot of the best model
npz.load_npz('result_feedforward/model_' + str(idx+1), model)

# example of getting the weights and biases for a deep neural network
W = {}
b = {}
for i in range(model.predictor.nlayer):

    W[i] = model.predictor[0][i].W.data
    b[i] = model.predictor[0][i].b.data

