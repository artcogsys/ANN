import numpy as np

class RingBuffer:
    """ class that implements a not-yet-full buffer """

    def __init__(self,nbuffer, nvars, dtype):

        # size of the buffer
        self.nbuffer = nbuffer

        # nr of variables for each item
        self.nvars = nvars

        # data type of the buffer
        self.dtype = dtype

        # Initialize buffer
        self.data = np.empty([self.nbuffer, self.nvars], dtype=self.dtype)
        self.data[:] = np.nan

        # Index into current item
        self.cur = 0

        # Keeps track of whether the buffer is filled
        self.full = False

    def size(self):
        """"
        return number of items in data buffer
        """

        if self.full:
            return self.nbuffer
        else:
            return self.cur

    def indices(self):
        """
        Returns valid indices ordered form old to new
        :return: indices
        """

        if self.full:
            return np.hstack([np.arange(self.cur, self.nbuffer), np.arange(0, self.cur)])
        else:
            return np.arange(0, self.cur)

    def append(self,item):
        """append an element at the end of the buffer"""

        if self.full:

            self.data[self.cur, :] = item
            self.cur = (self.cur + 1) % self.nbuffer

        else:

            self.data[self.cur,:] = item

            self.cur += 1

            if self.cur == self.nbuffer:

                self.cur = 0

                # Permanently change self's class from non-full to full
                self.full = True

    def get(self, n = None):
        """
        Return data items sorted from the oldest to the newest.

        :param n: when specified only returns the n most recent items ordered from oldest to newest
        :return: n x nvars data matrix

        If indices are beyond the buffer then we return an empty array
        """

        # Retrieve indices
        idx = self.indices()

        # Reduce to last n frames
        if n:
            idx = idx[idx.size - n:]

        # handle empty array and missing observations
        if not idx.size or idx.size < n:
            return np.array([])
        else:
            return self.data[idx]

    def getRandom(self, k, n=1):
        """
        Get k random examples of length n frames

        :param k: number of examples
        :param n: number of frames per example
        :return: k x n x nvars data matrix

        Note: Random sampling without replacement. If there are not enough buffer items then the returned buffer is augmented with nans
        """

        # Select random examples in the buffer
        idx = self.indices()

        # Select at most k indices while taking number of frames into account
        idx = np.random.permutation(idx[n-1:])[0:k]

        # Get n frames for each index
        idx = np.array(map(lambda x: np.arange(x - n + 1, x + 1), idx))

        # handle empty array and missing observations
        if not idx.size or idx.size < n:
            return np.array([])
        else:
            return self.data[idx]

    def getByIdx(self, idx, n=1):
        """
            Get items by index

            :param idx: indices of items to retrieve [0,...,self.size()-1] from old to new
            :param n: when specified returns the n most recent items ordered from oldest to newest (default is all)

            :return: nidx x n x nvars data matrix
            """

        assert((idx < self.size()).all())

        # Get n frames for each index
        idx = np.array(map(lambda x: np.arange(x - n + 1, x + 1), idx))

        if not idx.size or (idx<0).any():
            return np.array([])
        else:

            # Transform unordered to ordered indices
            oidx = self.indices()[idx]

            return self.data[oidx]

if __name__ == '__main__':

    x = RingBuffer(5, 2, 'float32')

    for i in xrange(5):
        x.append(np.array([i, i]))

    # get all data items
    print x.get()

    # get last two data items
    print  "Get last two data items:"
    print x.get(2)

    # get random items of length 2 frames
    print  "Get 4 random items of length 2:"
    print x.getRandom(6,2)

    # get 3 indexed items of length 2 frames
    print  "Get 3 indexed items of length 2:"
    print x.getByIdx(np.array([0, 3, 4]), 3)

