from chainer import Chain
import chainer.functions as F

#####
## Regressor object

class Regressor(Chain):
    def __init__(self, predictor):
        super(Regressor, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        return loss
