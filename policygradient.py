import numpy as np
from chainer import cuda, Chain, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import copy
import modelzoo
from ringbuffer import RingBuffer
import matplotlib.pyplot as plt

###
# Policy gradient learning

# see pg-pong.py for example

class PG(object):
    """
    Policy Gradient
    """

class RPG(object):
    """
    Recurrent Policy Gradient
    """



