from AUTOENCODER.autoencoder import Autoencoder
from AUTOENCODER.autoencoder_utils import *
import matplotlib.pyplot as plt
import os
from UTILS import utils as utils


if __name__ == "__main__":

    # data import parameters
    dataset_name = "mnist_kaggle"
    lower_limit = 0.5

    # model's hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 16
    HIDDEN_SIZE = 300
    LEARNING_RATE = 1e-03
    INPUT_DIMENTION = None # established after model import 

    fp = utils.verify_input_folder(dataset_directory=dataset_name)
    fpp = utils.verify_files(["train.csv","test.csv","sample_submission.csv"],full_path=fp)
    if fpp:
        files = utils.list_directory(full_path=fp)
        train_file = files[2]
        X,y = utils.get_mnist(data_path=train_file,shuffle_data=False)
        print(f"[+] Retrived {len(X)} samples")
        print(f"[~] Features shape: {X.shape}\n[~] Labels shape: {y.shape}")
        # get input dimention based on the data
        INPUT_DIMENTION = X.shape[1]

        # preprocess the data
        X_ = normalize(X=X)

        # instance model
        autoencoder = Autoencoder(
            X=X,
            D=INPUT_DIMENTION,
            M=HIDDEN_SIZE
        )
        print(f"[+] Created autoencoder instance with the following parameters:\n[*] Num Hidden Layers: {HIDDEN_SIZE}\n[*] Input Dimention: {INPUT_DIMENTION}")
        optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE) 
        print(f"[+] Optimizer instanced with learning rate {LEARNING_RATE}")
        print("[~] Traning Model...")
        costs = training_loop(
            X_ = X_,
            batch_size = BATCH_SIZE,
            model = autoencoder,
            optimizer = optimizer,
            epochs = EPOCHS
        )

        plt.plot(costs)
        plt.show()

        # choose a random image sample
        index = np.random.randint(0,len(X))
        sample = X[index,:]
        sample_n = normalize_sample(sample)

        # make predictions
        logits = autoencoder(sample_n,training = False)
        pred = tf.nn.sigmoid(logits)

        # denormalize prediction
        denormalized = denormalize_sample(pred)

        sample_r,denormalized_r = sample.reshape(28,28),denormalized.reshape(28,28)

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(sample_r,cmap="gray")
        plt.title("Original Image")
        plt.axis(False)
        plt.subplot(1,2,2)
        plt.imshow(denormalized_r,cmap="gray")
        plt.title("Reconstruction")
        plt.axis(False)
        plt.show()

    else:
        print("[-] Error in reading files, aborting...")