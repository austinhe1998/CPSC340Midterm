# basics
import os
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from pandas import Series
from auto_regressive import AutoRegressive

# sklearn imports
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

# from random_forest import RandomForest

from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == '2.1':
        filename = "phase1_training_data.csv"

        with open(os.path.join(filename), 'rb') as f:
            dataset = pd.read_csv(f)

        X = dataset.values[:,:]
        D = X[X[:, 0] == 'CA', 3]
        model = AutoRegressive(7)
        model.fit(D)
        prediction = model.predict(D, 11)
        print(prediction)
        
    else:
        print("Unknown question: %s" % question)
