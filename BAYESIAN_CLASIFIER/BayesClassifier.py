from __future__ import print_function,division
from builtins import range,input
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt



def clamp_sample(x):
  x = np.minimum(x, 1)
  x = np.maximum(x, 0)
  return x

class BayesClassifier:
  def fit(self,X,y):
    self.K = len(set(y)) # number of labels
    self.encoded_labels = list(set(y))
    self.gaussians = list() # store the means and covariance value we can use sample from a gaussian distribution
    self.p_y = np.zeros(self.K) # store the probability values
    for k in range(self.K):
      # extract the samples that belong to the target class
      xk = X[y == k] 
      # update the probability with the number of samples that belong to the target class
      self.p_y[k] = len(xk) 
      # compute the mean and the covariance
      mean,cov = np.mean(xk,axis=0),np.cov(xk.T)
      g = {
        "m":mean,
        "c":cov
      }
      self.gaussians.append(g)
    # normalizw the probabilities between 0 and 1
    self.p_y /= self.p_y.sum()

  def sample_given_y(self, y): # P(x|y) -> generate new samples given a gaussian
    g = self.gaussians[y]
    return clamp_sample( mvn.rvs(mean=g['m'], cov=g['c']) )

  def sample(self): # p(y) -> Get a y sample given the probability distribution obtained
    y = np.random.choice(self.K, p=self.p_y)
    return clamp_sample( self.sample_given_y(y) )
  
  def plot_probability(self):
    plt.figure()
    plt.bar(self.encoded_labels,self.p_y)
    plt.title("P(y)")
    plt.xlabel("encoded samples")
    plt.ylabel("normalized probability [0:1]")
    plt.show()

