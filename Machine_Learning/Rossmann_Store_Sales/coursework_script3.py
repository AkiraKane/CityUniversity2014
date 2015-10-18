#!/usr/bin/python

# Created: 11/10/2015
# Last Modified: 18/10/2015
# Dan Dixey
# Machine Learning Coursework

# Bayes Optimisation Reference Paper: http://goo.gl/PCvRTV
# Bayes Optimisaion Module: https://goo.gl/Miyb4B
# Scoring Parameter: http://goo.gl/khqrqO
# Scoring: http://arxiv.org/pdf/1209.5111v1.pdf

import pandas as pd
import numpy as np
from time import time
import os
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import normalize, scale
from sklearn.ensemble import RandomForestRegressor  # ML Algo 1
from sklearn.tree import DecisionTreeRegressor  # ML Algo 2
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from datetime import datetime
from sklearn.cross_validation import KFold
import json


# Seaborn Parameters - Plotting Library
sns.set(style="whitegrid")


# Scikit uses Numpy for Random Number Generation, setting a random seed value
# ensures that the result can be repeated without worry of lossing analysis
np.random.seed(1989)


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind]**2)
    return w


def RMSPE(yhat, y):
    ''' Calculates the RMSPE as defined: https://goo.gl/gD3JKe
    '''
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return rmspe


def importData(name):
    ''' Import the data into a Pandas Dataframe
    '''
    assert isinstance(name, str), 'Enter the NAME of the File to be imported!'
    # Windows - Resolves the issue with the direction of backslashes
    #fileName = "\\".join([os.getcwd(), name])
    # Linux - Resolves the issue with the direction of backslashes
    fileName = "/".join([os.getcwd(), name])
    print "Importing '{}' as a Dataframe".format(name)
    return pd.read_csv(fileName, low_memory=False)


def cross_validation(
        max_depth,
        max_features,
        criterion,
        normV,
        n_estimators,
        scaleV,
        log_y):
    ''' Run the model and return the 'score', where 'score' in this case
    is model 'RMSPE'
    '''
    # Get Data for Testing
    X_ = X
    y_ = y.values.ravel()
    # Normalization
    if int(np.round(normV)) == 1:
        for col in scaleNorm:
            X_[col] = normalize(X_[col]).T
    # Scaling
    if int(np.round(scaleV)) == 1:
        for col in scaleNorm:
            X_[col] = scale(X_[col]).T
    # Log the Class Variable
    if int(np.round(log_y)) == 1:
        y_ = np.log(y_)
    # Machine Learning Criteria
    if int(np.round(criterion)) == 0:
        criterion = 'mse'
    else:
        criterion = 'friedman_mse'
    # Convery to Numpy Arrays
    X_ = X_.values
    clf = RandomForestRegressor(
        max_depth=int(np.round(max_depth)),
        max_features=int(np.round(max_features)),
        n_jobs=-1,
        criterion=criterion)
    # Using k-fold Cross Validation(5 folds), Scoring criteria = RMSPE
    crossVal = KFold(X_.shape[0], n_folds=5)
    data = []
    for train, test in crossVal:
        clf.fit(X_[train], y_[train])
        if int(np.round(log_y)) == 1:
            yhat_ = np.exp(clf.predict(X_[test]))
        else:
            yhat_ = clf.predict(X_[test])
        data.append(RMSPE(yhat_, y_[test]))
    # Save Data as a Array
    data = np.array(data)
    return -data.mean()


def cross_validation2(
        max_depth,
        max_features,
        criterion,
        normV,
        scaleV,
        log_y):
    ''' Run the model and return the 'score', where 'score' in this case
    is model 'RMSPE'
    '''
    # Get Data for Testing
    X_ = X
    y_ = y.values.ravel()
    # Normalization
    # Normalization
    if int(np.round(normV)) == 1:
        for col in scaleNorm:
            X_[col] = normalize(X_[col]).T
    # Scaling
    if int(np.round(scaleV)) == 1:
        for col in scaleNorm:
            X_[col] = scale(X_[col]).T
    # Log the Class Variable

    if int(np.round(log_y)) == 1:
        y_ = np.log(y_)
    # Machine Learning Criteria
    if int(np.round(criterion)) == 0:
        criterion = 'mse'
    else:
        criterion = 'friedman_mse'
    # Convery to Numpy Arrays
    X_ = X_.values
    clf = DecisionTreeRegressor(
        max_depth=int(np.round(max_depth)),
        max_features=int(np.round(max_features)))
    # Using k-fold Cross Validation(5 folds), Scoring criteria = RMSPE
    crossVal = KFold(X_.shape[0], n_folds=5)
    data = []
    for train, test in crossVal:
        clf.fit(X_[train], y_[train])
        if int(np.round(log_y)) == 1:
            yhat_ = np.exp(clf.predict(X_[test]))
        else:
            yhat_ = clf.predict(X_[test])
        data.append(RMSPE(yhat_, y_[test]))
    # Save Data as a Array
    data = np.array(data)
    return -data.mean()


def preproc_dataset(dataframe, floatList, deleteCols, delOpen):
    ''' Converting the Dictionary of parameters from the trials data back
    into a format such that the Machine Learning Algorithms can process.
    '''
    # Categorical to Integer Representation - StoreType
    dataframe.loc[dataframe['StoreType'] == 'a', 'StoreType'] = '1'
    dataframe.loc[dataframe['StoreType'] == 'b', 'StoreType'] = '2'
    dataframe.loc[dataframe['StoreType'] == 'c', 'StoreType'] = '3'
    dataframe.loc[dataframe['StoreType'] == 'd', 'StoreType'] = '4'
    dataframe['StoreType'] = dataframe['StoreType'].astype(float)
    # Categorical to Integer Representation - Assortment
    dataframe.loc[dataframe['Assortment'] == 'a', 'Assortment'] = '1'
    dataframe.loc[dataframe['Assortment'] == 'b', 'Assortment'] = '2'
    dataframe.loc[dataframe['Assortment'] == 'c', 'Assortment'] = '3'
    dataframe['Assortment'] = dataframe['Assortment'].astype(float)
    # Categorical to Integer Representation - StateHoliday
    dataframe.loc[dataframe['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    dataframe.loc[dataframe['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    dataframe.loc[dataframe['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    dataframe['StateHoliday'] = dataframe['StateHoliday'].astype(float)
    # Splitting out the dates
    dataframe['year'] = dataframe.Date.apply(lambda x: x.split('-')[0])
    dataframe['year'] = dataframe['year'].astype(float)
    dataframe['month'] = dataframe.Date.apply(lambda x: x.split('-')[1])
    dataframe['month'] = dataframe['month'].astype(float)
    dataframe['day'] = dataframe.Date.apply(lambda x: x.split('-')[2])
    dataframe['day'] = dataframe['day'].astype(float)
    # Boolean Flag if there is Compeition or Not
    dataframe.loc[
        dataframe['CompetitionDistance'] > 0,
        'CompetitionDistance'] = '1'
    dataframe.loc[
        dataframe['CompetitionDistance'] <= 0,
        'CompetitionDistance'] = '0'
    dataframe.CompetitionDistance.fillna(0, inplace=True)
    dataframe['CompetitionDistance'] = dataframe[
        'CompetitionDistance'].astype(float)
    # For each col in floatList covert integers to type = float
    for item in floatList:
        dataframe[item] = dataframe[item].astype(float)
    # Use only Store that are Open '1' as predictors
    if delOpen:
        dataframe = dataframe[dataframe.Open == 1]
        dataframe = dataframe[dataframe.Sales != 0]
    # Remove Unwanted Columns
    dataframe = dataframe[dataframe.columns - deleteCols]
    return dataframe


def plot_data(parameters, results, name):
    # First Plot
    print('Plotting each Parameter by Cost Function')
    cols = len(parameters)
    f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(18, 5), sharey=True)
    for i, val in enumerate(parameters):
        xs = np.array([int(np.round(item[val]))
                       for item in results['params']]).ravel()
        ys = [-value for value in results['values']]
        cs = normalize(ys)
        temp = np.array(sorted(zip(xs, ys)))
        xs, ys = temp[:, 0], temp[:, 1]
        axes[i].scatter(
            xs,
            ys,
            s=40,
            linewidth=0.01,
            alpha=0.4,
            c=cs,
            cmap=mpl.colors.ListedColormap(sns.color_palette("hls", 8)))
        axes[i].set_title(val)
        # axes[i].set_ylim([0.8,1])
    plt.savefig(name + '_Paramenters.png', bbox_inches='tight')
    plt.close()
    # Second Plot
    print('Plotting the Trial number by Cost Function')
    f, ax = plt.subplots(1)
    xs = np.arange(1, len(results['values']) + 1)
    ys = [-value for value in results['values']]
    cs = normalize(ys)
    ax.set_xlim(0, xs[-1] + 10)
    ax.scatter(xs, ys, c=cs, s=40,
               linewidth=0.01, alpha=0.75,
               cmap=mpl.colors.ListedColormap(sns.color_palette("hls", 8)))
    ax.set_title('Loss vs Trial Number ', fontsize=18)
    ax.set_xlabel('Trial Number', fontsize=16)
    ax.set_ylabel('Model Loss', fontsize=16)
    plt.savefig(name + '_Trial.png', bbox_inches='tight')
    plt.close()


def fit_predict(X, y, X_test, params, model):
    # Get Data for Testing
    X_ = X
    X_test_ = X_test
    y_ = y.values.ravel()
    # Floats to Int Transform
    for k, v in params.iteritems():
        params[k] = int(np.round(v))
    # Normalization Transform
    if 'normV' in params:
        if int(np.round(params['normV'])) == 1:
            for col in scaleNorm:
                X_[col] = normalize(X_[col]).T
                X_test_[col] = normalize(X_test_[col]).T
        del params['normV']
    # Scale Transformation
    if 'scaleV' in params:
        if int(np.round(params['scaleV'])) == 1:
            for col in scaleNorm:
                X_[col] = scale(X_[col]).T
                X_test_[col] = scale(X_test_[col]).T
        del params['scaleV']
    # Log y transformation
    if 'log_y' in params:
        if int(np.round(params['log_y'])) == 1:
            logValues = 1
            y_ = np.log(y_)
        del params['log_y']
    # Machine Learning Criteria
    if 'criterion' in params:
        if int(np.round(params['criterion'])) == 0:
            params['criterion'] = 'mse'
        else:
            params['criterion'] = 'friedman_mse'
    # Convery to Numpy Arrays
    X_ = X_.values
    X_test_ = X_test_.values
    if model == 1:
        clf = RandomForestRegressor(**params)
    else:
        clf = DecisionTreeRegressor(**params)
    clf.fit(X_, y_)
    # Return the predicted results
    if logValues == 1:
        return np.exp(clf.predict(X_test_))
    else:
        clf.predict(X_test_)


# Start Stopwatch
print(datetime.now())
startT = time()
# Import the Data
trainingDF = importData('train.csv')
testDF = importData('test.csv')
storesDF = importData('store.csv')
submission = importData('sample_submission.csv')
testDF = testDF.sort_index(axis=1).set_index('Id')
# Data Preparation
# Joins, seperate columns
trainingDF = pd.merge(trainingDF, storesDF, on='Store')
testDF = pd.merge(testDF, storesDF, on='Store')
# Identifying Columns to datatype
numericalCols = ['Sales', 'Customers', 'CompetitionDistance']
categorCols = ['Assortment', 'StoreType']
dateCols = ['CompetitionOpenSinceYear', 'Date']
binaryCols = list(
    trainingDF.columns -
    numericalCols -
    categorCols -
    dateCols)
print('{} Binary Columns, {} Numerical Columns, {} Date Columns, {} Catgorical Columns'). \
    format(len(binaryCols), len(numericalCols), len(dateCols), len(categorCols))
# Basic Statistics
# Missing Data - Table + Chart
missingData = []
for col in trainingDF:
    value = float(trainingDF[col].isnull().sum()) / trainingDF.shape[0]
    missingData.append(np.float(format(value * 100, '.3f')))
missingData = pd.DataFrame(missingData,
                           index=trainingDF.columns,
                           columns=['PercentageMissing'])
# Removing the following columns as to much is missing
removeCols = ['PromoInterval', 'Promo2SinceYear', 'Promo2SinceWeek']
# Column Statistics - Table + Chart
summary = pd.DataFrame(trainingDF.describe().T)
summary = summary.drop(['count'], axis=1)
# Data dictionary
# TODO
# Removing Spurious Data
noCompetition = trainingDF[
    trainingDF.CompetitionDistance.isnull()].Store.unique()
print('Number of Stores with "NO" Competition: {}').format(len(noCompetition))
# Compeition Opening Since not known?
nStores = trainingDF[
    trainingDF.CompetitionOpenSinceMonth.isnull()].Store.unique()
print('Number of Stores with "NO KNOWLEDGE" of when Compeitiors Opened: {}'). \
    format((len(nStores) - len(noCompetition)))
removeCols.extend(['CompetitionOpenSinceYear',
                   'CompetitionOpenSinceMonth',
                   'Date',
                   'Customers',
                   'Open'])
# Columns that can be Scaled or Normalised as they are Continuous
global scaleNorm
scaleNorm = ['CompetitionDistance']
# Preprocessing Steps - Refer to the function above
trainingDF = preproc_dataset(trainingDF, scaleNorm, removeCols, True)
removeCols.remove('Customers')
# Pre-processing the Test Dataset with the same preprocessing steps
testDF = preproc_dataset(testDF, scaleNorm, removeCols, False)
# Feature Engineering - Pre - Training
# Ideas for 'feature enginerring' in Notes.md File within directory
# Sampling, Setting up Cross Validation - Make X and Y Global (for reuse)
global X, y
X = trainingDF[trainingDF.columns - removeCols - ['Sales']]
y = trainingDF[['Sales']]
testDF = testDF.sort_index(axis=1).set_index('Id')
ID_Data = testDF['Id']
X_test = testDF[testDF.columns - ['Id']]
print('Size of Training Set: Columns = {}, Rows = {}'). \
    format(X.shape[1], X.shape[0])
print('Size of Test Set: Columns = {}, Rows = {}'). \
    format(testDF.shape[1], testDF.shape[0])
# Machine Learning Algorithm #1 - Define ranges of Hyperparameters
ML1_BO = BayesianOptimization(cross_validation, {'max_depth': (40, 500),
                                                 'max_features': (1, 12),
                                                 'criterion': (0, 1),
                                                 'normV': (0, 1),
                                                 'n_estimators': (10, 200),
                                                 'scaleV': (0, 1),
                                                 'log_y': (0, 1)})
ML1_BO.explore({'max_depth': [200],
                'max_features': [12],
                'criterion': [1],
                'normV': [1],
                'n_estimators': [100],
                'scaleV': [0],
                'log_y': [1]})
# Machine Learning Algorithm #2 - Define ranges of Hyperparameters
ML2_BO = BayesianOptimization(cross_validation2, {'max_depth': (40, 500),
                                                  'max_features': (1, 12),
                                                  'criterion': (0, 1),
                                                  'normV': (0, 1),
                                                  'scaleV': (0, 1),
                                                  'log_y': (0, 1)})
ML2_BO.explore({'max_depth': [200],
                'max_features': [12],
                'criterion': [1],
                'normV': [1],
                'scaleV': [0],
                'log_y': [1]})
# Optimisation of Machine Learning Algorithm #1 = RandomForestRegressor
ML1_BO.maximize(init_points=250, n_iter=1)
# Optimisation of Machine Learning Algorithm #2 = DecisionTreeRegressor
ML2_BO.maximize(init_points=250, n_iter=1)
# Feature Engineering - Post - Recommendations
# Investigate - Feature Importance
# Evaluate Models - Graphical and Tabular Results - Plot Trial Data
plot_data(ML1_BO.res['all']['params'][0].keys(),
          ML1_BO.res['all'], 'RandomForestRegressor')
plot_data(ML2_BO.res['all']['params'][0].keys(),
          ML2_BO.res['all'], 'DecisionTreeRegressor')
# Refit and Predict Result from the Testing Set
output1 = fit_predict(X, y, X_test, ML1_BO.res['max']['max_params'], 1)
output2 = fit_predict(X, y, X_test, ML2_BO.res['max']['max_params'], 2)
# Join with IDs, label and remove unwanted columns
Submission = pd.DataFrame(data=[np.arange(1, len(output1) + 1),
                                output1,
                                output2]).T
Submission.columns = ['Id', 'ML1', 'ML2']
# Save output for Submission to Kaggle
Submission.to_csv('submission_xF.csv', sep=',', index=False)
# Save Parameters
with open('bestResults.json', 'w') as f:
    json.dump([ML1_BO.res['max'], ML2_BO.res['max']], f, ensure_ascii=False)
# Print Final Statement - Time to Train
print('Total time to Load, Optimise and Present Details: {} seconds').format(
    time() - startT)
print(datetime.now())
