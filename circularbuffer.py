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

        # Index of the first element in the buffer
        self.offset = None

        # Flags whether the whole buffer has been filled
        self.full = False

    def put(self, item):
        """
        Store new item in buffer
        :param item
        """

        if self.offset is None:
            self.offset = 0
        else:
            self.offset += 1

        # Index of circular memory
        self.offset = self.offset % self.nbuffer

        # Store experience
        self.buffer[self.offset, :] = item

        # Set buffer to filled
        if self.offset == self.nbuffer - 1:
            self.full = True

    def get(self, idxs):
        """
        Get elements from the buffer

        :param idxs: array of numbers of size nexamples x nframes
        :return: buffer contents of size nexamples x nframes x nvariables
        """

        if self.full:

            # take buffer offset and size into account
            idxs = np.array(map(lambda x: (x + self.offset + 1) % self.nbuffer, idxs))

        return self.buffer[idxs]