import chainer
from analysis import Analysis
from environment import datasets
from models import neural_networks as models
from models.utilities import Classifier
from paradigms import supervised_learning
from chainer.links.caffe import CaffeFunction
from chainer import Variable
import numpy as np
from chainer.links import VGG16Layers
from PIL import Image


model = VGG16Layers()

img = Image.open("path/to/image.jpg")
feature = model.extract([img], layers=["fc7"])["fc7"]


# Load the model
func = CaffeFunction('bvlc_googlenet.caffemodel')

# Minibatch of size 10
x_data = np.ndarray((10, 3, 227, 227), dtype=np.float32)

# Forward the pre-trained net
x = Variable(x_data)
y, = func(inputs={'data': x}, outputs=['fc8'])

# create caffemodel neural network


# create analysis object
ana = Analysis(ann.model, fname='tmp')

# handle sequential data; deal with classifier analysis separately

# analyse data
ana.classification_analysis(validation_data.X, validation_data.T)