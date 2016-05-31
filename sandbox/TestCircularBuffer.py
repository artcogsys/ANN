import unittest
import numpy as np
from circularbuffer import CircularBuffer

class MyTestCase(unittest.TestCase):

    def test_multi(self):

        print "Testing multiple floats"

        x = CircularBuffer(5, 2, np.float32)

        for i in xrange(8):
            x.put([i,i])

        print x.buffer

        print x.get(np.arange(0,5))


    def test_single(self):

        print "Testing single integers"

        x = CircularBuffer(5, 1, np.int8)

        for i in xrange(8):
            x.put([i])

        print x.buffer

        print x.get(np.arange(0, 5))


if __name__ == '__main__':
    unittest.main()
