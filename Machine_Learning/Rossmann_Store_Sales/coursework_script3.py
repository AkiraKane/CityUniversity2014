#!/usr/bin/python

# Created: 11/10/2015
# Last Modified: 18/10/2015
# Dan Dixey
# Machine Learning Coursework

# Bayes Optimisation Reference Paper: http://goo.gl/PCvRTV
# Bayes Optimisation Module: https://goo.gl/Miyb4B
# Applied Use of Bayes Optimisation: http://bit.ly/1aRpyYE
# Scoring Parameter: http://goo.gl/khqrqO
# Scoring: http://arxiv.org/pdf/1209.5111v1.pdf

from time import time

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import json
from math import sqrt

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.cross_validation import KFold

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from bayes_opt import BayesianOptimization


# Seaborn Parameters - Plotting Library
sns.set(style="whitegrid")


# Scikit uses Numpy for Random Number Generation, setting a random seed value
# ensures that the result can be repeated without worry of loosing analysis
np.random.seed(191989)


def get_optimal(ml1_bo, ml2_bo, ml3_bo):
    loc1, loc2, loc3 = None, None, None
    for i, val in enumerate(ml1_bo.res['all']['values']):
        if val == ml1_bo.res['max']['max_val']:
            loc1 = i
    for i, val in enumerate(ml2_bo.res['all']['values']):
        if val == ml2_bo.res['max']['max_val']:
            loc2 = i
    for i, val in enumerate(ml3_bo.res['all']['values']):
        if val == ml3_bo.res['max']['max_val']:
            loc3 = i
    return loc1, loc2, loc3


def RMSPE(labels, predictions):
    """
    Calculates the Root Mean Squared Percentage Error
    :rtype : Float
    """
    if len(predictions) != len(labels):
        raise Exception("Labels and predictions must be of same length")
    # Remove pairs where label == 0
    labels, predictions = tuple(
        zip(*filter(lambda x: x[0] != 0, zip(labels, predictions)))
    )
    labels = np.array(labels, dtype=float)
    predictions = np.array(predictions, dtype=float)
    return sqrt(np.power((labels - predictions) /
                         labels, 2.0).sum() / len(labels))


def import_data(name):
    ''' Import the data into a Pandas Dataframe
    :rtype : Pandas Dataframe
    '''
    assert isinstance(name, str), 'Enter the NAME of the File to be imported!'
    # Linux - Resolves the issue with the direction of backslashes
    filename = "/".join([os.getcwd(), name])
    print "Importing '{}' as a Dataframe".format(name)
    # Merge with Stores Data - Complementary
    stores_df = pd.read_csv("/".join([os.getcwd(), 'store.csv']))
    stores_df = stores_df.fillna(-1)
    stores_df['StoreType'] = LabelEncoder(
    ).fit_transform(stores_df['StoreType'])
    stores_df['Assortment'] = LabelEncoder(
    ).fit_transform(stores_df['Assortment'])
    stores_df = stores_df.drop('PromoInterval', axis=1)
    # Import the Dataset Requested
    master_df = pd.read_csv(
        filename, parse_dates=['Date'], dtype={
            'StateHoliday': object})
    master_df = master_df.drop('StateHoliday', axis=1)
    # Date Conversion
    (
        master_df['DayInt'],
        master_df['Weekend'],
        master_df['Day'],
        master_df['Month'],
        master_df['Year']
    ) = zip(*master_df['Date'].map(process_dates))
    # Post Processing - Remove the Date Columns
    master_df = master_df.drop('Date', axis=1)
    # Fill NAs with -1 to avoid Confusion
    master_df = master_df.fillna(-1)
    # Merging Data with the stores data
    return pd.merge(master_df, stores_df, on='Store', how='inner', sort=False)


def cross_validation(
        max_features,
        criterion,
        normv,
        n_estimators,
        log_y):
    ''' Run the model and return the 'score', where 'score' in this case
    is model 'RMSPE'
    :rtype : Float
    '''
    # Get Data for Testing
    X_ = X
    y_ = y
    # Normalization
    if int(np.round(normv)) == 1:
        for col in scaleNorm:
            X_[col] = normalize(X_[col].values).T
    # Log the Class Variable
    if int(np.round(log_y)) == 1:
        y_ = y_['Sales'].apply(lambda x: np.log(x + 1))
    # Machine Learning Criteria
    if int(np.round(criterion)) == 0:
        metric = 'mse'
    else:
        metric = 'friedman_mse'
    # Using k-fold Cross Validation(5 folds)
    KFolds = KFold(X.shape[0], 5, random_state=191989)
    X_ = X_.values
    y_ = y_.values.ravel()
    data = []
    for train, test in KFolds:
        model = RandomForestRegressor(
                                max_features=1 / max_features,
                                random_state=191989,
                                n_estimators=int(n_estimators),
                                n_jobs=-1,
                                criterion=metric)
        model.fit(X=X_[train], y=y_[train])
        # Scoring criteria = RMSPE
        data.append(RMSPE(y_[test], model.predict(X_[test])))
    # Save Data as a Array
    data = np.array(data)
    return -data.mean()


def cross_validation2(
        max_depth,
        max_features,
        criterion,
        normv,
        log_y):
    """ Run the model and return the 'score', where 'score' in this case
    is model 'RMSPE'
    :rtype : Float
    """
    # Get Data for Testing
    X_ = X
    y_ = y
    # Normalization
    if int(np.round(normv)) == 1:
        for col in scaleNorm:
            X_[col] = normalize(X_[col].values).T
    # Log the Class Variable
    if int(np.round(log_y)) == 1:
        y_ = y_['Sales'].apply(lambda x: np.log(x + 1))
    # Convert to Numpy Arrays
    X_ = X_.values
    y_= y_.values.ravel()
    # Using k-fold Cross Validation(5 folds)
    KFolds = KFold(X.shape[0], 5, random_state=191989)
    data = []
    for train, test in KFolds:
        model = DecisionTreeRegressor(
                                max_depth=int(np.round(max_depth)),
                                max_features=1 / max_features)
        model.fit(X_[train], y_[train])
        # Scoring criteria = RMSPE
        data.append(RMSPE(model.predict(X_[test]), y_[test]))
    # Save Data as a Array
    data = np.array(data)
    return -data.mean()


def cross_validation3(
        l_rate,
        n_estimators,
        max_depth,
        normv,
        log_y):
    """ Run the model and return the 'score', where 'score' in this case
    is model 'RMSPE'
    :rtype : Float
    """
    # Get Data for Testing
    X_ = X
    y_ = y
    # Normalization
    if int(np.round(normv)) == 1:
        for col in scaleNorm:
            X_[col] = normalize(X_[col].values).T
    # Log the Class Variable
    if int(np.round(log_y)) == 1:
        y_ = y_['Sales'].apply(lambda x: np.log(x + 1))
    # Convert to Numpy Arrays
    X_ = X_.values
    y_= y_.values.ravel()
    # Using k-fold Cross Validation(5 folds)
    KFolds = KFold(X.shape[0], 5, random_state=191989)
    data = []
    for train, test in KFolds:
        gbm = xgb.XGBRegressor(max_depth=int(max_depth),
                                n_estimators=int(n_estimators),
                                learning_rate=l_rate).fit(X_[train], y_[train])
        # Scoring criteria = RMSPE
        data.append(RMSPE(y_[test], gbm.predict(X_[test])))
    # Save Data as a Array
    data = np.array(data)
    return -data.mean()


def process_dates(dt):
    """
    :param dt: Datetime
    :return: Various Metrics Derived from the input Date
    """
    weekday = dt.weekday()
    return (
        weekday,
        1 if weekday >= 5 else 0,
        dt.day,
        dt.month,
        dt.year,
    )


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
    plt.savefig(name + '_Parameters.png', bbox_inches='tight')
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
    """
    # Get Final Predictions using the Best Result from the Bayesian Optimisation
    :rtype : Numpy Array
    """
    X_ = X
    X_test_ = X_test
    y_ = y
    # Floats to Int Transform
    for k, v in params.iteritems():
        if k != 'l_rate':
            params[k] = int(np.round(v))
    # Normalization Transform
    if 'normv' in params:
        if int(np.round(params['normv'])) == 1:
            for col in scaleNorm:
                X_[col] = normalize(X_[col].values).T
                X_test_[col] = normalize(X_test_[col].values).T
        del params['normv']
    # Log y transformation
    if 'log_y' in params:
        if int(np.round(params['log_y'])) == 1:
            y_ = y_['Sales'].apply(lambda x: np.log(x + 1))
            logY = True
        del params['log_y']
    # Machine Learning Criteria
    if 'criterion' in params:
        if int(np.round(params['criterion'])) == 0:
            params['criterion'] = 'mse'
        else:
            params['criterion'] = 'friedman_mse'
    # Convert to Numpy Arrays
    X_ = X_.values
    X_test_ = X_test_.values
    y_ = y_.values.ravel()
    if model == 1:
        model = RandomForestRegressor(**params)
        model.fit(X_, y_)
        # Feature Importance
        print "Feature Importances:"
        pairs = zip(X.columns, model.feature_importances_)
        pairs.sort(key=lambda x: -x[1])
        for column, importance in pairs:
            print " ", column, importance
    elif model == 2:
        model = DecisionTreeRegressor(**params)
        model.fit(X_, y_)
    else:
        model = xgb.XGBRegressor(max_depth=params['max_depth'],
                                n_estimators=params['n_estimators'],
                                learning_rate=params['l_rate']).fit(X_, y_)
    # Return the predicted results
    if logY == True:
        return np.exp(model.predict(X_test_)) - 1
    else:
        return model.predict(X_test_)


# Start Stopwatch
print(datetime.now())
startT = time()
# Import the Data
trainingDF = import_data('train.csv')
# Apply A Shuffle - to assist with KFold which is ordered
trainingDF = trainingDF.iloc[np.random.permutation(len(trainingDF))]
testDF = import_data('test.csv')
# Identifying Columns by DType - Manual
numCols = [
    'Sales',
    'Customers',
    'CompetitionDistance',
    'CompetitionOpenSinceMonth']
catCols = [
    'Assortment',
    'StoreType',
    'Store',
    'CompetitionOpenSinceYear',
    'CompetitionOpenSinceMonth',
    'Promo2SinceWeek',
    'Promo2SinceYear']
dtCols = [
    'CompetitionOpenSinceYear',
    'DayOfWeek',
    'DayInt',
    'Day',
    'Month',
    'Year']
binCols = trainingDF.columns.difference(
    numCols).difference(catCols).difference(dtCols)
print('{} Total Columns, {} Binary Columns, {} Numerical Columns, {} Date Columns, {} Categorical Columns'). \
    format(len(trainingDF.columns), len(binCols),
           len(numCols), len(dtCols), len(catCols))
# Basic Statistics
summary = pd.DataFrame(trainingDF.describe().T)
summary = summary.drop(['count'], axis=1)
# Data dictionary
dataDict = trainingDF.info()
# Removing Spurious Data
noCompetition = trainingDF[
    trainingDF.CompetitionDistance == -1].Store.unique()
print('Number of Stores with "NO" Competition: {}').format(len(noCompetition))
# Competition Opening Since not known?
nStores = trainingDF[
    trainingDF.CompetitionOpenSinceMonth == -1].Store.unique()
print('Number of Stores with "NO KNOWLEDGE" of when Competitors Opened: {}'). \
    format((len(nStores) - len(noCompetition)))
# Columns that can be Scaled or Normalised as they are Continuous
global scaleNorm
scaleNorm = ['CompetitionDistance']
# Add Feature Engineering Here - Ideas in Notes.md
# TODO
X_columns = trainingDF.columns.difference(['Customers',
                                           'Id',
                                           'DayOfWeek',
                                           'Sales'])
y_columns = ['Sales']
# Sampling, Setting up Cross Validation - Make X and Y Global (for reuse)
global X, y
X = trainingDF[X_columns]
y = trainingDF[y_columns]
idDF = testDF['Id']
X_Final = testDF[X_columns.difference(['Sales'])]
print 'Size of Training Set: Columns = {}, Rows = {}'. \
    format(X.shape[1], X.shape[0])
print 'Size of Test Set: Columns = {}, Rows = {}'. \
    format(X_Final.shape[1], X_Final.shape[0])
# Machine Learning Algorithm #1 - Define ranges of Hyperparameters
ml1_bo = BayesianOptimization(cross_validation, {'max_features': (1, 12),
                                                 'criterion': (0, 1),
                                                 'normv': (0, 1),
                                                 'n_estimators': (100, 300),
                                                 'log_y': (0, 1)})
ml1_bo.explore({'max_features': [3.0],
                'criterion': [0],
                'normv': [0],
                'n_estimators': [250],
                'log_y': [1]})
# Machine Learning Algorithm #2 - Define ranges of Hyperparameters
ml2_bo = BayesianOptimization(cross_validation2, {'max_depth': (5, 500),
                                                  'max_features': (1, 17),
                                                  'criterion': (0, 1),
                                                  'normv': (0, 1),
                                                  'log_y': (0, 1)})
ml2_bo.explore({'max_depth': [200],
                'max_features': [3.0],
                'criterion': [0],
                'normv': [0],
                'log_y': [1]})
# Machine Learning Algorithm #3 - Define ranges of Hyperparameters
ml3_bo = BayesianOptimization(cross_validation3, {'l_rate': (0.01, 0.15),
                                                  'n_estimators': (200, 600),
                                                  'max_depth': (0, 1),
                                                  'normv': (0, 1),
                                                  'log_y': (0, 1)})
ml3_bo.explore({'l_rate': [0.04],
                'n_estimators': [600],
                'max_depth': [5],
                'normv': [0],
                'log_y': [1]})
# Optimisation of Machine Learning Algorithm #1 = RandomForestRegressor
ml1_bo.maximize(init_points=10, n_iter=1)
# Optimisation of Machine Learning Algorithm #2 = DecisionTreeRegressor
ml2_bo.maximize(init_points=10, n_iter=1)
# Optimisation of Machine Learning Algorithm #2 = DecisionTreeRegressor
ml3_bo.maximize(init_points=10, n_iter=1)
# Feature Engineering - Post - Recommendations
# Investigate - Feature Importance
# Evaluate Models - Graphical and Tabular Results - Plot Trial Data
plot_data(ml1_bo.res['all']['params'][0].keys(),
          ml1_bo.res['all'], 'RandomForestRegressor')
plot_data(ml2_bo.res['all']['params'][0].keys(),
          ml2_bo.res['all'], 'DecisionTreeRegressor')
plot_data(ml3_bo.res['all']['params'][0].keys(),
          ml3_bo.res['all'], 'XgBoost_Regressor')
# Get the Parameters of the 'Best' Result
loc1, loc2, loc3 = get_optimal(ml1_bo, ml2_bo, ml3_bo)
# Refit and Predict Result from the Testing Set
max_ml3 = {'n_estimators': 600.0, 'log_y': 1.0, 'max_depth': 5.0, 'l_rate': 0.040000000000000001, 'normv': 0.0}
max_ml2 = {'max_features': 3.0, 'log_y': 1.0, 'criterion': 0.0, 'max_depth': 200.0, 'normv': 0.0}
max_ml1 = {'max_features': 2.1897849917350625, 'n_estimators': 108.01612211934166, 'log_y': 0.53980534381173073, 'criterion': 0.26032666518704972, 'normv': 0.60012625039277268}
output1 = fit_predict(X, y, X_Final, max_ml1, 1)
output2 = fit_predict(X, y, X_Final, max_ml2, 2)
output3 = fit_predict(X, y, X_Final, max_ml3, 3)
# Write Predictions to CSV Files - Machine Learning 1
with open('ML1.csv', 'wb') as f:
    f.write("Id,Sales\n")
    for i, predicted_val in enumerate(output1):
        f.write("%d,%d\n" % (idDF[i], predicted_val))
# Write Predictions to CSV Files - Machine Learning 2
with open('ML2.csv', 'wb') as f:
    f.write("Id,Sales\n")
    for i, predicted_val in enumerate(output2):
        f.write("%d,%d\n" % (idDF[i], predicted_val))
# Write Predictions to CSV Files - Machine Learning 3
with open('ML3.csv', 'wb') as f:
    f.write("Id,Sales\n")
    for i, predicted_val in enumerate(output3):
        f.write("%d,%d\n" % (idDF[i], predicted_val))
# Save Parameters - For Presentation Outcome
with open('bestResults.json', 'w') as f:
    json.dump([ml1_bo.res['max'],
               ml2_bo.res['max'],
               ml3_bo.res['max']], f, ensure_ascii=False)
# Print Final Statement - Time to Train
print('Total time to Load, Optimise and Present Details: {} seconds').format(
    time() - startT)
print(datetime.now())
