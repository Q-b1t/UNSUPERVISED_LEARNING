from BAYESIAN_CLASIFIER import BayesClassifier as BayesClassifier
import matplotlib.pyplot as plt
from UTILS import utils as utils
import os

if __name__ == "__main__":
    dataset_name = "mnist_kaggle"
    lower_limit = 0.5
    fp = utils.verify_input_folder(dataset_directory=dataset_name)
    fpp = utils.verify_files(["train.csv","test.csv","sample_submission.csv"],full_path=fp)
    if fpp:
        files = utils.list_directory(full_path=fp)
        train_file = files[2]
        X,y = utils.get_mnist(data_path=train_file,lower_limit=lower_limit)
        print(f"[+] Retrived {len(X)} samples")
        print(f"[~] Features shape: {X.shape}\n[~] Labels shape: {y.shape}")
        bc = BayesClassifier.BayesClassifier()
        bc.fit(X,y)
        bc.plot_probability()
        sample = bc.sample_given_y(0)
        for k in range(bc.K):
            # show one sample for each class
            # also show the mean image learned

            sample = bc.sample_given_y(k).reshape(28, 28)
            mean = bc.gaussians[k]['m'].reshape(28, 28)

            plt.subplot(1,2,1)
            plt.imshow(sample, cmap='gray')
            plt.title("Sample")
            plt.subplot(1,2,2)
            plt.imshow(mean, cmap='gray')
            plt.title("Mean")
            plt.show()
        
        # generate a random sample
        sample = bc.sample().reshape(28, 28)
        plt.imshow(sample, cmap='gray')
        plt.title("Random Sample from Random Class")
        plt.show()

    else:
        print("[-] Error in reading files, aborting...")