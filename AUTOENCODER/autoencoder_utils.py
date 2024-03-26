import numpy as np
import tensorflow as tf
import tqdm

# preprocessing utils
def normalize(X):
  """
  Normalize all the samples of the dataset
  """
  return X.astype(np.float32) / 255.0

def normalize_sample(sample):
  """
  Use it to prepare individual samples for the autoencoder to make predictions on it.
  The sample should not have the batch dimention added yet (totally raw sample)
  """
  # add batch dimention
  sample_n = sample[np.newaxis,:]
  # normalize 
  sample_n =  sample_n.astype(np.float32) / 255.0
  return sample_n


def denormalize_sample(prediction):
  """
  Receives an autoencoder prediction and denormalizes it for it to be visualized again
  """
  denormalized = tf.squeeze(prediction,axis=0).numpy()
  denormalized *= 255.0
  denormalized = denormalized.astype(np.int64)  
  return denormalized


# training utils
def loss_function(labels,logits):
  return tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, # X
        logits=logits # X_hat
      )
    )


def training_step(X,model,optimizer):
  with tf.GradientTape() as tape:
    X_hat = model(X,training=True)
    loss = loss_function(X,X_hat)
    gradients = tape.gradient(loss,model.trainable_variables)
  optimizer.apply_gradients(zip(gradients,model.trainable_variables))
  return loss


def training_loop(X_,batch_size,model,optimizer,epochs,module = 5):
  costs = list()
  batch_costs = list()
  num_batches = int(len(X_) / batch_size)

  print(f"[~] Batch Num: {num_batches}")
  for epoch in range(epochs):
    print(f"[~] Epoch {epoch}...")
    np.random.shuffle(X_)
    for j in tqdm.tqdm(range(num_batches)):
      batch = X_[j * batch_size : (j+1) * batch_size]
      loss = training_step(
          X = batch,
          model = model,
          optimizer = optimizer
      )
      batch_costs.append(loss.numpy())
    overall_cost = np.mean(batch_costs)
    costs.append(overall_cost)
    if epoch % module == 0:
      print(f"[~] Current loss: {overall_cost}")
  return costs
