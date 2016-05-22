import numpy as np

class CircularBuffer(object):
    """
    Circular buffer
    """

    def __init__(self, nbuffer, nvars, dtype):
        """
        Create circular buffer which contains nbuffer items of length nvariables and type dtype
        """

        # size of the buffer
        self.nbuffer = nbuffer

        # nr of variables for each item
        self.nvars = nvars

        # data type of the buffer
        self.dtype = dtype

        # Initialize buffer
        self.buffer = np.empty([self.nbuffer, self.nvars], dtype=self.dtype)
        self.buffer[:] =  np.nan

        # Index of the newest element in the buffer (modulo nbuffer)
        self.idx = None

    def put(self, item):
        """
        Store new item in buffer
        :param item
        """

        if self.idx is None:
            self.idx = 0
        else:
            self.idx += 1

        # Store experience
        self.buffer[self.idx % self.nbuffer, :] = item

    def get(self, idxs):
        """
        Get elements from the buffer.

        :param idxs: array of numbers of size nexamples x nframes
        :return: buffer contents of size nexamples x nframes x nvariables
        """

        if idxs.size == 0:
            return np.array([], dtype=self.dtype)

        return self.buffer[map(lambda x: x % self.nbuffer, idxs)]

    def buffer_size(self):
        """
        :return: Size of the buffer (maximal nbuffer)
        """

        return np.min(self.idx+1,self.nbuffer)
