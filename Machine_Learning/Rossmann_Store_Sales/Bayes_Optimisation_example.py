#!/usr/bin/python

# Created: 12/10/2015
# Last Modified: 12/10/2015
# Dan Dixey
# Machine Learning Coursework
# Bayes Optimisation Demo

from sklearn import datasets
from hyperopt import fmin, tpe, hp
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.preprocessing import normalize, scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from time import time
import seaborn as sns
import matplotlib as mpl


# Seaborn Parameters
sns.set(style="whitegrid")


# Scikit uses Numpy for Random Number Generation, setting a random seed value
# ensures that the result can be repeated without worry of lossing analysis
np.random.seed(1989)


def import_data():
    ''' Function imports the IRIS dataset from the Scikit learn module
    '''
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


def cross_validation(params):
    ''' Run the model and return the 'score', where 'score' in this case is model 'accuracy'
    '''
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']
    clf = DecisionTreeClassifier(**params)
    # Using Stratified Sampling, 5 folds, Scoring criteria = Accuracy
    data = cross_val_score(clf, X_, y, cv=5, n_jobs=-1, scoring='accuracy')
    return [data.mean(), data.std()]


def results(params):
    ''' Intermediate step to return a dictionary of the loss
    '''
    acc = cross_validation(params)
    return {'loss': -acc[0], 'std': acc[1], 'status': STATUS_OK}


#########################################
''' - Define the hyperparameter space
    - find the best model
    - plot by each parameter
    - plot the loss by each attempted trial
'''
startT = time()
X, y = import_data()
# Defining Parameter Ranges (Stochastic): https://goo.gl/ktblo5
hypParameters = dict(max_depth=hp.choice('max_depth', range(1, 30)),
                max_features=hp.choice('max_features', range(1, 5)),
                criterion=hp.choice('criterion', ["gini", "entropy"]),
                normalize=hp.choice('normalize', [0, 1]))

trials = Trials()
best = fmin(results, hypParameters, algo=tpe.suggest, max_evals=50, trials=trials)
print 'Best Solution (Parameters): {} Loss (Result): {} +/- {}'.format(best,
                                                                       np.abs(trials.best_trial['result']['loss']),
                                                                       np.abs(trials.best_trial['result']['std']))

print('Plotting Parameters by Loss')
parameters = ['max_depth', 'max_features', 'criterion', 'normalize']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(18, 5), sharey=True)
#cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
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
plt.show()

print('Plotting Trail by Loss')
f, ax = plt.subplots(1)
xs = [t['tid'] for t in trials.trials]
ys = [np.abs(t['result']['loss']) for t in trials.trials]
ys = [-t['result']['loss'] for t in trials.trials]
cs = normalize(ys)
ax.set_xlim(xs[0] - 10, xs[-1] + 10)
ax.scatter(xs, ys, c=cs, s=40, 
           linewidth=0.01, alpha=0.75,
           cmap=mpl.colors.ListedColormap(sns.color_palette("hls", 8)))
ax.set_title('Loss vs Trial Number ', fontsize=18)
ax.set_xlabel('Trial Number', fontsize=16)
ax.set_ylabel('Model Loss', fontsize=16)
plt.show()
print('Total time to Load, Optimise and Present Details: {} seconds').format(
    time() - startT)
