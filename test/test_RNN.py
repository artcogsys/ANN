import numpy as np
import os
import chainer.links as L
import chainer
from chainer import training
from chainer import datasets, iterators, optimizers
from chainer.training import extensions
from chainer.serializers import npz
from models import recurrent as rc
import matplotlib.pyplot as plt
import random
from rnn_utils import ParallelSequentialIterator, BPTTUpdater
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--cutoff', '-l', type=int, default=10,
                        help='Number of items in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--nepochs', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--nhidden', '-n', type=int, default=10,
                        help='Number of hidden units in each layer')
    parser.add_argument('--weight_decay', '-w', type=int, default=1e-5,
                        help='Weight decay parameter (regularization coefficient for L2 term)')
    args = parser.parse_args()

    # get file name
    directory = os.path.splitext(os.path.basename(__file__))[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    # For the sake of simplicity, we define the inputs as randomly
    # generated timeseries. That is, each input time point comprises two random numbers between
    # zero and one. Then, we define the targets as a function of the inputs. That is, each output
    # time point is zero if the sum of the previous input time point is less than one or one if
    # the sum of the previous input time point is greater than one.
    X = {}
    T = {}
    for phase in ['training', 'validation']:
        X[phase] = [[random.random(), random.random()] for _ in xrange(1000)]
        T[phase] = [0] + [0 if sum(i) < 1.0 else 1 for i in X[phase]][:-1]
        X[phase] = np.array(X[phase], 'float32')
        T[phase] = np.array(T[phase], 'int32')

    # set train and validation data as TupleDatasets
    train = datasets.tuple_dataset.TupleDataset(X['training'], T['training'])
    validation = datasets.tuple_dataset.TupleDataset(X['validation'], T['validation'])

    # infer input and output size
    ninput = train._datasets[0].shape[1]
    noutput = np.unique(train._datasets[1]).size

    # MultiprocessIterator could be used to prefetch (large) batch sizes
    train_iter = ParallelSequentialIterator(train, args.batch_size, repeat=True)
    validation_iter = ParallelSequentialIterator(validation, args.batch_size,
                                                 repeat=False)  # batch_size = 1 in standard approach

    # For classification, we use cross-entropy as the cost function
    model = L.Classifier(rc.RNN(ninput, args.nhidden, noutput))

    # Activate GPU if needed
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.cutoff, args.gpu)
    trainer = training.Trainer(updater, (args.nepochs, 'epoch'), out=directory)

    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    # eval_rnn.train = False
    trainer.extend(extensions.Evaluator(
        validation_iter, eval_model, device=args.gpu,
        # Reset the RNN state at the beginning of each evaluation
        eval_hook=lambda _: eval_rnn.reset_state()))

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # trainer.extend(extensions.snapshot_object(
    #     model, 'model_iter_{.updater.epoch}'))

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
    plt.plot(range(len(loss)), loss)
    plt.title('loss')
    plt.subplot(122)
    plt.plot(range(len(accuracy)), accuracy)
    plt.title('accuracy')
    plt.show()

if __name__ == '__main__':
    main()