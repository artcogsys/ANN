import numpy as np
import matplotlib.pyplot as plt

def scatterplot(T, Y, idx=0):
    """
    scatterplot for targets T versus predictions Y

    :param T: T x M array
    :param Y: T x M array
    :param idx: output index
    """



    plt.clf()
    plt.scatter(T[:,idx], Y[:,idx])
    plt.axis('equal')
    plt.grid(True)
    plt.title('R = ' + str(np.corrcoef(T[:,idx], Y[:,idx])[0,1]))
    plt.show()
