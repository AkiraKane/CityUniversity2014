from sklearn import datasets
from hyperopt import fmin, tpe, hp
import hyperopt.pyll.stochastic
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.preprocessing import normalize, scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation  import cross_val_score
import matplotlib.pyplot as plt
import numpy as np


def import_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


def hyperopt_train_test(params):
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
    return cross_val_score(clf, X, y).mean()


def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}


def main():
    space4dt = dict(max_depth=hp.choice('max_depth', range(1, 20)),
                    max_features=hp.choice('max_features', range(1, 5)),
                    criterion=hp.choice('criterion', ["gini", "entropy"]),
                    normalize=hp.choice('normalize', [0, 1]))

    trials = Trials()
    best = fmin(f, space4dt, algo=tpe.suggest, max_evals=300, trials=trials)
    print 'Best Solution: {}'.format(best)

    print('Plotting Parameters $by$ Loss')
    parameters = ['max_depth', 'max_features', 'criterion', 'normalize'] # decision tree
    cols = len(parameters)
    f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        temp = np.array(sorted(zip(xs, ys)))
        xs, ys = temp[:,0], temp[:,1]
        axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i)/len(parameters)))
        axes[i].set_title(val)
        axes[i].set_ylim([0.8,1])

    print('Plotting Trail $by$ Loss')
    f, ax = plt.subplots(1)
    xs = [t['tid'] for t in trials.trials]
    ys = [np.abs(t['result']['loss']) for t in trials.trials]
    ax.set_xlim(xs[0]-10, xs[-1]+10)
    ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('Loss $vs$ Trial Number ', fontsize=18)
    ax.set_xlabel('Trial Number', fontsize=16)
    ax.set_ylabel('Model Loss', fontsize=16)

if __name__ == "__main__":
    main()
