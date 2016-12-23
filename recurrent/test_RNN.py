import chainer
import modelzoo as m
import numpy
import random
import recurrent_neural_network as r
import matplotlib.pyplot as plt
plt.ion()

### Classification example ###

# First, we define the data. For the sake of simplicity, we define the inputs as randomly
# generated timeseries. That is, each input time point comprises two random numbers between
# zero and one. Then, we define the targets as a function of the inputs. That is, each output
# time point is zero if the sum of the previous input time point is less than one or one if
# the sum of the previous input time point is greater than one.

X = {}
T = {}

for phase in ['training', 'validation']:
    X[phase] = [[random.random(), random.random()] for _ in xrange(1000)]
    T[phase] = [0] + [0 if sum(i) < 1.0 else 1 for i in X[phase]][:-1]
    X[phase] = numpy.array(X[phase], 'float32')
    T[phase] = numpy.array(T[phase], 'int32')

# Next we define the model:
compute_accuracy = True # optional for classification, must be false for regression.
experiment = 'classification' # this is just a name for the experiment
id = None # gpu id; if None, cpu is used
lossfun = chainer.functions.softmax_cross_entropy # chainer lossfun. SCE for classification.
optimizer = chainer.optimizers.Adam() # chainer optimizer
predictor = m.RNN(2, 10, 2) # a predictor; either from  model_zoo or custom defined

RNN = r.RecurrentNeuralNetwork(compute_accuracy, experiment, id, lossfun, optimizer, predictor)

# And the optimization parameters:
cutoff = 10 # cutoff steps for truncated backprop.
epochs = 100 # number of epochs to train (# passes over the entire data)
seqs = 32 # number of mini batches
callback = r.callback # an optional function called after each epoch (see example in script)
rate_lasso = None # regularization coefficient for L1 term
rate_weight_decay = 1e-5  # regularization coefficient for L2 term
threshold = 5 # threshold for gradient clipping

# Finally we run the optimization
# Note: to use a model after optimization, the predict method should be used; train and test
# methods are for internal use only.
RNN.optimize(T, X, cutoff, epochs, seqs, callback, rate_lasso, rate_weight_decay, threshold)