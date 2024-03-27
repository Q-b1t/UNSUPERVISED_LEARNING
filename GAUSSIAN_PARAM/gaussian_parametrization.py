import numpy as np



def softplus(x):
   return np.log1p(np.exp(x))

def forward(x,W1,W2):
    """
    Computes a feed forward perceptron where each element of the R4 output layer are the parameters to model a multivariate distribution
    [0:1] -> mean
    [2:3] -> stdev
    """
    hidden = np.tanh(x.dot(W1))
    output = hidden.dot(W2)
    mean = output[:2]
    stddev = softplus(output[2:])
    return mean, stddev
