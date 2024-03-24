from __future__ import print_function,division
from builtins import range,input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture


def clamp_sample(x):
  x = np.minimum(x, 1)
  x = np.maximum(x, 0)
  return x

class BayesClassifierGMM:
  def fit(self,X,y,cluster_num = 10):
    """
    So this is the same as the BayesClassifier class but this uses the Gaussian Mixture Model algorithm to fit multimodal
    distributions to the training data. This is useful for clustering on probability sensity functions that have more than 
    one mode (as the algorithm's name states)
    X: The X samples
    y: The y samples
    cluster_num: due to variational inference we have an infinite number of clusters, which we cut down to a finite number
    """
    self.K = len(set(y)) # number of labels
    self.encoded_labels = list(set(y)) # store the labels (asuming they are numecally encoded before passing the, to the classifer)
    self.gaussians = list() # store the BayesianGaussianMixtureModel instanced made for each sample ground belonging to a class
    self.p_y = np.zeros(self.K) # store the probability values
    print(f"[~] GMM set {cluster_num} to clusters")
    for k in range(self.K):
      print(f"[*] Fitting Bayesian GMM for class {k}...")
      xk = X[k == y] # get the subset of training samples labeles as the class that is being fit
      self.p_y[k] = len(xk) # set the corresponding probability for the sample being fit (experimental)
      gmm = BayesianGaussianMixture(n_components=cluster_num)
      gmm.fit(xk)
      self.gaussians.append(gmm)
    # normalize the experimental probability density function
    self.p_y /= self.p_y.sum()

  def sample_given_y(self, y): # P(x|y) -> generate new samples given a gaussian
    gmm = self.gaussians[y]
    sample = gmm.sample()
    mean = gmm.means_[sample[1]]
    return clamp_sample( sample[0].reshape(28, 28) ), mean.reshape(28, 28)
  
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

