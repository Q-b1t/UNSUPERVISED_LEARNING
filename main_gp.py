from GAUSSIAN_PARAM.gaussian_parametrization import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal as mvn

if __name__ == "__main__":
    # hyperparameters
    SAMPLE_NUM = 10_000

    # initialie thr weights
    W1,W2 = np.random.randn(4,3),np.random.randn(3,2*2)

    # initialize a random input 
    x = np.random.randn(4)

    # get the parameters of the Gaussian
    mean, stddev = forward(x, W1, W2)
    print("[~]: Mean:", mean)
    print("[~]: Stddev:", stddev)

    # draw samples
    samples = mvn.rvs(mean=mean, cov=stddev**2, size=SAMPLE_NUM)

    # plot the samples
    plt.scatter(samples[:,0], samples[:,1], alpha=0.5)
    plt.show()