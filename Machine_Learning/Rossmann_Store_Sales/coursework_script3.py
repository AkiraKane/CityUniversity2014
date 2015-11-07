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
from sklearn.linear_model import BayesianRidge
import seaborn as sns
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from bayes_opt import BayesianOptimization


# Seaborn Parameters - Plotting Library
sns.set(style="whitegrid")


# Scikit uses Numpy for Random Number Generation, setting a random seed value
# ensures that the result can be repeated without worry of loosing analysis
np.random.seed(191989)


def feature_engineering(dataframe, showtTResults=False):
    # Investigte the Relationship between Means between on-off School Holiday
    MedianSalesSchoolHoliday = dataframe.pivot_table(index='Store', 
                                                    columns=['Month', 
                                                    'SchoolHoliday'], 
                                                    values='Sales', 
                                                    aggfunc=np.median)
    # The t statistic to test whether the means are different
    from scipy.stats import ttest_ind
    # Paired t-test
    if showtTResults:
        for i in np.arange(2013,2016,1):
            temp = MedianSalesSchoolHoliday[i].dropna()
            # P-Values H0: (mew)1 = (mew)2, Ha: (mew)1 != (mew)2
            ttest_ind(a=temp[0], b=temp[1])[1]
    # Since the Means are not significantly different at 0.05 significance
    # Paramenters will not be created and used in the model
    MedianSalesPromo = dataframe.pivot_table(index='Store', 
                                                    columns=['Month', 
                                                    'Promo'], 
                                                    values='Sales', 
                                                    aggfunc=np.median)
    # The median has been used since the data is positively skewed 
    # Paired t-test
    if showtTResults:
        for i in np.arange(1,13,1):
            temp = MedianSalesPromo[i].dropna()
            # P-Values H0: (mew)1 = (mew)2, Ha: (mew)1 != (mew)2
            ttest_ind(a=temp[0], b=temp[1])[1]
    # Stack to Make Merging Simple
    MedianSalesPromo = MedianSalesPromo.stack()
    MedianSalesSchoolHoliday = MedianSalesSchoolHoliday.stack()
    # Update Column Names    
    MedianSalesPromo.columns = ["_".join([str(val),"MedSalesPromo"]) 
                                    for val in MedianSalesPromo.columns]
    MedianSalesSchoolHoliday.columns = ["_".join([str(val),"MedSalesSchoolHoliday"]) 
                                    for val in MedianSalesSchoolHoliday.columns]
    return MedianSalesPromo, MedianSalesSchoolHoliday
    

def get_optimal(ml1_bo, ml2_bo):
    loc1, loc2 = None, None
    for i, val in enumerate(ml1_bo.res['all']['values']):
        if val == ml1_bo.res['max']['max_val']:
            loc1 = i
    for i, val in enumerate(ml2_bo.res['all']['values']):
        if val == ml2_bo.res['max']['max_val']:
            loc2 = i
    return loc1, loc2


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
        max_depth,
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
                                max_depth = max_depth,
                                random_state=191989,
                                n_estimators=int(n_estimators),
                                n_jobs=-1,
                                criterion=metric)
        model.fit(X=X_[train], y=y_[train])
        # Scoring criteria = RMSPE
        data.append(mean_absolute_error(y_[test], model.predict(X_[test])))
    # Save Data as a Array
    data = np.array(data)
    return -data.mean()


def cross_validation2(
        n_iter,
        tol,
        fit_intercept,
        normv,
        alpha_1,
        alpha_2,
        log_y):
    """ Run the model and return the 'score', where 'score' in this case
    is model 'RMSPE'
    :rtype : Float
    """
    # Get Data for Testing
    X_ = X
    y_ = y
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
        model = BayesianRidge(n_iter = int(n_iter),
                              fit_intercept = int(np.round(fit_intercept)),
                              tol = tol,
                              alpha_1 = alpha_1,
                              alpha_2 = alpha_2,
                              normalize = int(np.round(normv))
        )
        model.fit(X_[train], y_[train])
        # Scoring criteria = RMSPE
        data.append(mean_absolute_error(model.predict(X_[test]), y_[test]))
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
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X_.shape[1]), indices)
        plt.xlim([-1, X_.shape[1]])
        plt.savefig(model + ' Feature importances.png', bbox_inches='tight')
        plt.close()
    elif model == 2:
        model = BayesianRidge(**params)
        model.fit(X_, y_)
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
print('{} Total Columns, {} Binary Columns, {} Numerical Columns, \
{} Date Columns, {} Categorical Columns'). \
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
# Add the Engineered Features
MedSalesPro, MedSalesSchHol = feature_engineering(trainingDF, 
                                                           showtTResults=False)
# Sampling, Setting up Cross Validation - Make X and Y Global (for reuse)
global X, y
X = trainingDF[X_columns]
y = trainingDF[y_columns]
# Append Engineer Features
X = pd.merge(X, MedSalesPro, 
             how='left', 
             right_index=True, 
             left_on=['Store','Promo'])
X = pd.merge(X, MedSalesSchHol, 
             how='left', 
             right_index=True, 
             left_on=['Store','SchoolHoliday'])
# If there is a missing value due to the Merge, replace with the Mean value
X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
idDF = testDF['Id']
X_Final = testDF[X_columns.difference(['Sales'])]
X_Final = pd.merge(X_Final, 
                   MedSalesPro, 
                   how='left', 
                   right_index=True, 
                   left_on=['Store','Promo'])
X_Final = pd.merge(X_Final, 
                   MedSalesSchHol, 
                   how='left', 
                   right_index=True, 
                   left_on=['Store','SchoolHoliday'])
# If there is a missing value due to the Merge, replace with the Mean value
X_Final = X_Final.apply(lambda x: x.fillna(x.mean()),axis=0)
print 'Size of Training Set: Columns = {}, Rows = {}'. \
    format(X.shape[1], X.shape[0])
print 'Size of Test Set: Columns = {}, Rows = {}'. \
    format(X_Final.shape[1], X_Final.shape[0])
##############################################################################
# Bayesian Optimisation - 50 Iterations for Each Algorithm
##############################################################################
# Machine Learning Algorithm #1 - Define ranges of Hyperparameters
ml1_bo = BayesianOptimization(cross_validation, {'max_features': (1, 20),
                                                 'criterion': (0, 1),
                                                 'normv': (1, 1),
                                                 'max_depth': (1, 40),
                                                 'n_estimators': (100, 300),
                                                 'log_y': (1, 1)})
ml1_bo.explore({'max_features': [3.0],
                'criterion': [0],
                'normv': [1],
                'max_depth': [15],
                'n_estimators': [50],
                'log_y': [1]})
# Machine Learning Algorithm #2 - Define ranges of Hyperparameters
ml2_bo = BayesianOptimization(cross_validation2, {'n_iter': (10, 500),
                                                  'tol': (0, 1e-2),
                                                  'fit_intercept': (1, 1),
                                                  'normv': (1, 1),
                                                  'alpha_1': (0, 1e-2),
                                                  'alpha_2': (0, 1e-2),
                                                  'log_y': (1, 1)})
ml2_bo.explore({'n_iter': [350],
                'tol': [1e-3],
                'fit_intercept': [1],
                'normv': [1],
                'alpha_1': [1e-6],
                'alpha_2': [1e-6],
                'log_y': [1]})
# Optimisation of Machine Learning Algorithm #1 = RandomForestRegressor
ml1_bo.maximize(init_points=50, n_iter=1)
# Optimisation of Machine Learning Algorithm #2 = DecisionTreeRegressor
ml2_bo.maximize(init_points=50, n_iter=1)
# Feature Engineering - Post - Recommendations
# Investigate - Feature Importance
# Evaluate Models - Graphical and Tabular Results - Plot Trial Data
plot_data(ml1_bo.res['all']['params'][0].keys(),
          ml1_bo.res['all'], 'Random_Forest_Regressor')
plot_data(ml2_bo.res['all']['params'][0].keys(),
          ml2_bo.res['all'], 'Bayesian_Ridge_Regression')
# Get the Parameters of the 'Best' Result
loc1, loc2 = get_optimal(ml1_bo, ml2_bo)
# Refit and Predict Result from the Testing Set
max_ml1 = ml1_bo.res['max']['max_params']
max_ml2 = ml2_bo.res['max']['max_params']
# Add Normv and Log y
for item in ['normv','log_y']:
    max_ml1[item] = 1
    max_ml2[item] = 1
# Train a model with the Optimal Results, Generate Prediction of Test Dataset
output1 = fit_predict(X, y, X_Final, max_ml1, 1)
output2 = fit_predict(X, y, X_Final, max_ml2, 2)
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
# Save Parameters - For Presentation Outcome
with open('bestResults.json', 'w') as f:
    json.dump([ml1_bo.res['max'],
               ml2_bo.res['max']
               ], f, ensure_ascii=False)
# Save Parameters - For Presentation Outcome
with open('allResults.json', 'w') as f:
    json.dump([ml1_bo.res['all'],
               ml2_bo.res['all']
               ], f, ensure_ascii=False)
##############################################################################
# Print Final Statement - Time to Train
print('Total time to Load, Optimise and Present Details: {} seconds').format(
    time() - startT)
print(datetime.now())
