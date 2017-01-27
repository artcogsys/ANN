import numpy as np
import chainer
from chainer.functions.activation import relu
from chainer import link
from chainer.links.connection import linear

###
# Implementation of custom layers

class ElmanBase(link.Chain):

    def __init__(self, n_units, n_inputs=None, initU=None,
                 initW=None, bias_init=0):
        """

        :param n_units: Number of hidden units
        :param n_inputs: Number of input units
        :param initU: Input-to-hidden weight matrix initialization
        :param initW: Hidden-to-hidden weight matrix initialization
        :param bias_init: Bias initialization

        """

        if n_inputs is None:
            n_inputs = n_units

        super(ElmanBase, self).__init__(
            U=linear.Linear(n_inputs, n_units,
                            initialW=initU, initial_bias=bias_init),
            W=linear.Linear(n_units, n_units,
                            initialW=initW, nobias=True),
        )

class Elman(ElmanBase):
    """
    Implementation of simple linear Elman layer

    Consider using initW=chainer.initializers.Identity(scale=0.01)
    as in https://arxiv.org/pdf/1504.00941v2.pdf
    (scale=1.0 led to divergence issues in our example)

    """

    def __init__(self, in_size, out_size, initU=None,
                 initW=None, bias_init=0, actfun=relu.relu):
        super(Elman, self).__init__(
            out_size, in_size, initU, initW, bias_init)
        self.state_size = out_size
        self.reset_state()
        self.actfun = actfun

    def to_cpu(self):
        super(Elman, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(Elman, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == np:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):

        z = self.U(x)
        if self.h is not None:
            z += self.W(self.h)

        # must be part of layer since the transformed value is part of the
        # representation of the previous hidden state
        self.h = self.actfun(z)

        return self.h