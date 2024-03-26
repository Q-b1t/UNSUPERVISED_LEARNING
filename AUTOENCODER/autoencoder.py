import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class AutoencoderHidden(Layer):
  def __init__(self,M,D):
    """
    D: Input Dimention
    M: Num Hidden Units
    """
    super().__init__()
    # weights and bias of the hidden layer
    self.W = tf.Variable(tf.random.normal(shape=(D, M)) * np.sqrt(2.0 / M))
    self.b = tf.Variable(np.zeros(M).astype(np.float32))

    # reverse shape matrix to get the data back the the original dimentions (reconstruct the input)
    self.V = tf.Variable(tf.random.normal(shape=(M, D)) * np.sqrt(2.0 / D))
    self.c = tf.Variable(np.zeros(D).astype(np.float32))

  def call(self,X):
    Z = tf.nn.relu(tf.matmul(X,self.W) + self.b)
    logits = tf.matmul(Z, self.V) + self.c
    #X_hat =  tf.nn.sigmoid(logits)
    return logits


class Autoencoder(Model):
  def __init__(self,X,D,M):
    """
    Simple autoencoder consisting on one single hidden layer
    D: Input Dimention
    M: Num Hidden Units
    """
    super().__init__()
    # hyperparameters
    self.D = D
    self.M = M
    self.X = X
    # hidden layer
    self. hl = AutoencoderHidden(M = self.M,D = self.D)

    # use for predictions
    #self.X_t = tf.nn.sigmoid(self.hl(X))

  def call(self,X):
    X_hat = self.hl(X)
    return X_hat
