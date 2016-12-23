import numpy as np
import matplotlib.pyplot as plt

def functional_connectivity(data):
    """
    Plot functional connectivity matrix (full correlation)

    :param fname: file name
    :param data: list containing T x Mi timeseries data
    """

    x = np.hstack(data)
    M = np.corrcoef(x.transpose())

    plt.clf()
    plt.pcolor(M)
    plt.title('Functional connectivity')
    plt.show()
