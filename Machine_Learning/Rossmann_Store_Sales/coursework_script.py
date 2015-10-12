#!/usr/bin/python

# Created: 11/10/2015
# Last Modified: 11/10/2015
# Dan Dixey
# Machine Learning Coursework

# Bayes Optimisation Reference: http://goo.gl/PCvRTV
# Working Example of Decision Tree using the IRIS Dataset and hyperopt: https://goo.gl/M8pqgc

import pandas as pd
import numpy as np
from time import time
import os


def importData(name):
    assert isinstance(name, str), 'Enter the NAME of the File to be imported!'
    fileName = "/".join([os.getcwd(),name])
    print "Importing '{}' as a Dataframe".format(name)
    return pd.DataFrame.from_csv(fileName)
    

def main():
    # Import the Data
    trainingDF = importData('train.csv')
    testDF = importData('test.csv')
    storesDF = importData('store.csv')
    # Data Preparation
        # Joins, to Numpy Arrays, Categorical to Numerical
    # Basic Statistics
        # Missing Data - Table + Chart
        # Column Statistics - Table + Chart
        # Summary table
        # Data dictionary
    # Preprocessing Steps
        # Removing Spurious Data
        # Sampling, Setting up Cross Validation
    # Machine Learning Processing
        # ID2 ML Algorithms
        # Optimisation Wrapper
    # Access Results
        # Feature Engineering
        # Evaluate Models
    # Results and Metrics
        # Typical Regression Metrics - Kaggle
        # Graphical and Tabular Results
    return

if __name__ == "main":
    main()
