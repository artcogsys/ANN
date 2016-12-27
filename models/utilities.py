from chainer import Chain
import chainer.functions as F

#####
## Classifier object

class Classifier(Chain):

    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def predict(self,x):
        """
        Returns prediction, which can be different than raw output (e.g. for softmax function)
        :param x:
        :return: prediction
        """

        return F.softmax(self.predictor(x)).data

#####
## Regressor object

class Regressor(Chain):

    def __init__(self, predictor):
        super(Regressor, self).__init__(predictor=predictor)

    def __call__(self, x, t):

        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        return loss

    def predict(self,x):
        """
        Returns prediction, which can be different than raw output (e.g. for softmax function)
        :param x:
        :return: prediction
        """

        return self.predictor(x).data

