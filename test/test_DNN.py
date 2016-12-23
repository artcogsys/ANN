import numpy as np
import os
import chainer
import chainer.links as L
from chainer import training
from chainer import datasets, iterators, optimizers
from chainer.training import extensions
from chainer.serializers import npz
from models import feedforward as ff
import matplotlib.pyplot as plt
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--nepochs', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--nhidden', '-n', type=int, default=10,
                        help='Number of hidden units in each layer')
    parser.add_argument('--weight_decay', '-w', type=int, default=1e-5,
                        help='Weight decay parameter (regularization coefficient for L2 term)')
    args = parser.parse_args()

    # Also see:
    # https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py

    # get file name
    directory = os.path.splitext(os.path.basename(__file__))[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    # get train and validation data as TupleDatasets
    train, validation = datasets.get_mnist()

    # infer input and output size
    ninput = train._datasets[0].shape[1]
    noutput = np.unique(train._datasets[1]).size

    # MultiprocessIterator could be used to prefetch (large) batch sizes
    train_iter = iterators.SerialIterator(train, args.batch_size, repeat=True, shuffle=True)
    validation_iter = iterators.SerialIterator(validation, args.batch_size, repeat=False, shuffle=False)

    # For classification, we use cross-entropy as the cost function
    model = L.Classifier(ff.DNN(ninput, args.nhidden, noutput))

    # Activate GPU if needed
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    # Use ParallelUpdater for multiple GPUs
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.nepochs, 'epoch'), out=directory)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(validation_iter, model, device=args.gpu))

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

if __name__ == '__main__':
    main()