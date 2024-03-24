import glob
import matplotlib.pyplot as plt
import  numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle

def verify_input_folder(dataset_directory,input_directory = "input_data"):
    # verify input folder
    full_path = os.path.join(input_directory,dataset_directory)
    if not os.path.exists(input_directory) or (os.path.exists(input_directory) and not os.path.exists(full_path)):
        print(f"[-] Either input directory {input_directory} or dataset directory {full_path} does not exist\n[*] Creating input and dataset directory...")
        os.makedirs(os.path.join(input_directory,dataset_directory),exist_ok=True)
        print(f"[+] Dataset directory created\n[~] Upload files to {full_path}")
    else:
        print("[~] Dataset Folder already exist, skipping...")

    return full_path

def verify_files(target_files,full_path):
    files = os.listdir(full_path) #glob.glob(os.path.join(full_path,"*"))
    res = True
    if len(files) == 0:
        print(f"[-] {full_path} is empty")
        res = False
    else:
        print(f"[*] Searching for target files in {full_path}")
        for t_file in target_files:
            if t_file not in files:
                res = False
                print(f"[-] {t_file} was not found in {full_path}")
                break
            print(f"[+] {t_file} present in {full_path}")
    return res

def list_directory(full_path):
    return glob.glob(os.path.join(full_path,"*"))

def get_mnist(data_path,lower_limit = None,shuffle_data = False):
    """
    The function extracts the training data from the kaggle digit recognizer dataset.
    Arguments
        data_path: the full path of the file we wish to process. The code is thought to work in kaggle's dataset format
        lower_limit: float between 0 and 1 denoting the percentage of the data we wish to extract
        shuffle_data: boolean value denoting whether to shuffle or not the data. The convention is to shuffle for the training set
    """
    shuffle_data = float(shuffle_data)
    dataset = pd.read_csv(data_path)
    data = dataset.to_numpy()
    # dataset shape:  (42000, 785) where columns 0 are the labels and col [1:785] are pixel values
    # divide dataset into features and labels
    X,y = data[:,1:],data[:,0]
    x = X / 255.0 # normalize the data between 0 and 1
    # shuffle the data
    if shuffle_data:
        X,y = shuffle(X,y)
    if lower_limit is not None:
        # verify the data
        if lower_limit > 0.0 and lower_limit <= 1.0:
            threshold = len(data) * lower_limit
            threshold = int(threshold)
            X,y = X[:threshold],y[:threshold]     
        else:
            print(f"[-] The limit {lower_limit} is not in the required interval [0.0:1.0]")
            raise TypeError
    return X,y
    
    






