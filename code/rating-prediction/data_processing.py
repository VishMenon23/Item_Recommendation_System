# Importing the required libraries
from surprise import Reader, Dataset
from surprise import SVD, accuracy, SVDpp, SlopeOne, BaselineOnly, CoClustering
import datetime
import requests, zipfile, io
from os import path
import pandas as pd
import tqdm as tqdm
from numpy import *
from sklearn.model_selection import train_test_split
import time
import pickle
import pickle
import matplotlib.pyplot as plt
import json
import glob

# Accessing and manipulating the data
def load_process_data():
    merged_data = {}
    unique_reviewer_id = []

    filename = '../dataset/Cell_Phones_and_Accessories_5.json'

    # create a pandas dataframe from the python object
    df = pd.read_json(filename, lines=True)

    # Removing unnecessary columns 
    df = df.drop('reviewerName', axis=1)
    df = df.drop('vote', axis=1)
    df = df.drop('style', axis=1)
    df = df.drop('reviewText', axis=1)
    df = df.drop('summary', axis=1)
    df = df.drop('reviewTime', axis=1)
    df = df.drop('image', axis=1)
    
    print("DF",df.head)
    # Final Dataframe structure - 
    #     "reviewerID": str
    #     "asin": int
    #     "overall": float
    #     "unixReviewTime": int

    reviewer_id = df["reviewerID"].tolist()
    unique_reviewer_id.extend(set(reviewer_id))
    product_id = df["asin"].tolist()
    overall = df["overall"].tolist()
    unixReviewTime = df["unixReviewTime"].tolist()

    print("LENGTH",len(unique_reviewer_id),len(product_id),len(overall))

    combine_data = [list(a) for a in zip(reviewer_id, product_id, overall, unixReviewTime)]
    for a in combine_data:
        if a[0] in merged_data.keys():
            merged_data[a[0]].extend([[a[0], a[1], a[2], a[3]]])
        else:
            merged_data[a[0]] = [[a[0], a[1], a[2], a[3]]]

    return merged_data, unique_reviewer_id  

# Split the data into training and testing
def spilt_data(merged_data, unique_reviewer_id):
    training_data = []
    testing_data = []
    t0 = time.time()
    t1 = time.time()
    for u in unique_reviewer_id:
        if len(merged_data[u]) == 1:
            x_test = merged_data[u]
            x_train = merged_data[u]
        else:
            x_train, x_test = train_test_split(merged_data[u], test_size=0.2)
        training_data.extend(x_train)
        testing_data.extend(x_test)
    total = t1 - t0
    print(int(total))

    return training_data, testing_data


def get_train_test_data(new_sample = False):
    if new_sample:
        merged_data, unique_reviewer_id = load_process_data()
        training_data, testing_data = spilt_data(merged_data, unique_reviewer_id)
        training_data = pd.DataFrame.from_records(training_data)
        training_data.columns = ["reviewerID","asin","overall","unixReviewTime"]
        testing_data = pd.DataFrame.from_records(testing_data)
        testing_data.columns=["reviewerID","asin","overall","unixReviewTime"]
        file = open('training_data.txt', 'wb')
        pickle.dump(training_data, file)
        file.close()

        file = open('testing_data.txt', 'wb')
        pickle.dump(testing_data, file)
        file.close()

    else:
        file = open('training_data.txt', 'rb')
        training_data = pickle.load(file)
        file.close()

        file = open('testing_data.txt', 'rb')
        testing_data = pickle.load(file)
        file.close()

    return training_data, testing_data

print("Loading and processing the data :")
merged_data, unique_reviewer_id = load_process_data()

print("Building the training and testing dataset :")
training_data, testing_data = spilt_data(merged_data, unique_reviewer_id)
print("Length of train-test split")
print(len(training_data),len(testing_data))    