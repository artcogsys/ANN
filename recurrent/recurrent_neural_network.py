from chainer import Chain, Variable, cuda, optimizers, serializers
import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import termcolor
import time
import tqdm

class RecurrentNeuralNetwork(object):
    def __init__(self, compute_accuracy, experiment, id, lossfun, optimizer, predictor):
        self.experiment = experiment
        self.log = {}
        self.log[('training', 'loss')] = []
        self.log[('training', 'throughput')] = []
        self.log[('validation', 'loss')] = []
        self.log[('validation', 'throughput')] = []
        self.model = L.Classifier(predictor, lossfun) if id is None else L.Classifier(predictor, lossfun).to_gpu()
        self.model.compute_accuracy = compute_accuracy
        self.optimizer = optimizer
        
        self.optimizer.setup(self.model)
        
        self.xp = np if id is None else cuda.cupy
    
    def load(self, postfix, prefix):
        with open('{}log{}'.format(prefix, postfix), 'rb') as f:
            self.log = pickle.load(f)
        
        serializers.load_npz('{}optimizer{}'.format(prefix, postfix), self.optimizer)
        serializers.load_npz('{}model{}'.format(prefix, postfix), self.model)
    
    def optimize(self, T, X, cutoff, epochs, seqs, callback = None, rate_lasso = None, rate_weight_decay = None, threshold = None):
        if threshold is not None:
            if 'GradientClipping' in self.optimizer._hooks:
                del self.optimizer._hooks['GradientClipping']
            self.optimizer.add_hook(chainer.optimizer.GradientClipping(threshold))
        
        if rate_lasso is not None:
            if 'Lasso' in self.optimizer._hooks:
                del self.optimizer._hooks['Lasso']
            self.optimizer.add_hook(chainer.optimizer.Lasso(rate_lasso))
        
        if rate_weight_decay is not None:
            if 'WeightDecay' in self.optimizer._hooks:
                del self.optimizer._hooks['WeightDecay']
            self.optimizer.add_hook(chainer.optimizer.WeightDecay(rate_weight_decay))
        
        for epoch in tqdm.tqdm(xrange(self.optimizer.epoch, self.optimizer.epoch + epochs)):
            then = time.time()
            loss = self.train(T['training'], X['training'], cutoff, seqs)
            now = time.time()
            throughput = T['training'].shape[0] / (now - then)
            
            self.log[('training', 'loss')].append(loss)
            self.log[('training', 'throughput')].append(throughput)
            
            if 'validation' in T:
                then = time.time()
                loss = self.test(T['validation'], X['validation'], seqs)
                now = time.time()
                throughput = T['validation'].shape[0] / (now - then)
            else:
                loss = float('nan')
                throughput = float('nan')
            
            self.log[('validation', 'loss')].append(loss)
            self.log[('validation', 'throughput')].append(throughput)
            
            if callback is not None:
                callback(self, X, cutoff, epochs, seqs)
            
            self.optimizer.new_epoch()
    
    def predict(self, X):
        self.model.predictor.reset_state()
        
        self.model.predictor.test = True
        self.model.predictor.train = False
        Y = []
        
        for step in xrange(X.shape[0]):
            x = Variable(self.xp.asarray(X[step][None]), True)
            Y.append(self.model.predictor(x).data)
            
            if step == 0:
                H = [[self.model.predictor.h[i].data[0]] for i in xrange(len(self.model.predictor.h))]
            else:
                _ = [H[i].append(self.model.predictor.h[i].data[0]) for i in xrange(len(self.model.predictor.h))]
        
        H = [self.xp.asarray(H[i]) for i in xrange(len(H))]
        Y = self.xp.asarray(Y)
        
        return H, Y
    
    def save(self, postfix, prefix):
        with open('{}log{}'.format(prefix, postfix), 'wb') as f:
            pickle.dump(self.log, f, -1)
        
        serializers.save_npz('{}optimizer{}'.format(prefix, postfix), self.optimizer)
        serializers.save_npz('{}model{}'.format(prefix, postfix), self.model)
    
    def test(self, T, X, seqs):
        loss = Variable(self.xp.zeros((), 'float32'), True)
        model = self.model.copy()
        
        model.predictor.reset_state()
        
        model.predictor.test = True
        model.predictor.train = False
        steps = len(X) // seqs
        
        for step in xrange(steps):
            x = Variable(self.xp.asarray([X[(seq * steps + step) % len(X)] for seq in xrange(seqs)]), True)
            t = Variable(self.xp.asarray([T[(seq * steps + step) % len(T)] for seq in xrange(seqs)]), True)
            loss += model(x, t)
        
        loss = float(loss.data / steps)
        
        return loss
    
    def train(self, T, X, cutoff, seqs):
        cumloss = self.xp.zeros((), 'float32')
        loss = Variable(self.xp.zeros((), 'float32'))
        self.model.predictor.test = False
        self.model.predictor.train = True
        steps = len(X) // seqs
        
        for step in xrange(steps):
            x = Variable(self.xp.asarray([X[(seq * steps + step) % len(X)] for seq in xrange(seqs)]))
            t = Variable(self.xp.asarray([T[(seq * steps + step) % len(T)] for seq in xrange(seqs)]))
            loss += self.model(x, t)
            
            if (step + 1) % cutoff == 0 or (step + 1) == steps:
                self.optimizer.zero_grads()
                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()
                
                cumloss += loss.data
                loss = Variable(self.xp.zeros((), 'float32'))
        
        cumloss = float(cumloss / steps)
        
        return cumloss

def callback(model, X, cutoff, epochs, seqs):
    print 'epoch: {}'.format(model.optimizer.epoch)
    print 'legend: {}, {}'.format(termcolor.colored('training', 'red'), termcolor.colored('validation', 'green'))
    print 'loss: {}, {}'.format(termcolor.colored(model.log[('training', 'loss')][-1], 'red'), termcolor.colored(model.log[('validation', 'loss')][-1], 'green'))
    print 'throughput: {}, {}'.format(termcolor.colored(model.log[('training', 'throughput')][-1], 'red'), termcolor.colored(model.log[('validation', 'throughput')][-1], 'green'))
    
    plt.subplot(121)
    plt.plot(model.log[('training', 'loss')], 'r', model.log[('validation', 'loss')], 'g')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(122)
    plt.plot(model.log[('training', 'throughput')], 'r', model.log[('validation', 'throughput')], 'g')
    plt.xlabel('epoch')
    plt.ylabel('throughput')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # if not os.path.exists('model/{}/{}/'.format(model.experiment, model.optimizer.epoch)):
    #     os.makedirs('model/{}/{}/'.format(model.experiment, model.optimizer.epoch))
    # 
    # model.save('', 'model/{}/{}/'.format(model.experiment, model.optimizer.epoch))
