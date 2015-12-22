import numpy as np
from csv import reader, writer
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from time import clock

if __name__ == '__main__':
    train = []
    with open('train.csv') as csv:
        for row in reader(csv):
            train.append(row)

    train_labels = []
    with open('trainlabels.csv') as csv:
        for row in reader(csv):
            train_labels.extend(row)

    oos = []
    with open('test.csv') as csv:
        for row in reader(csv):
            oos.append(row)

    n_features = len(train[0])
    n_data = len(train)

    seed = 0
    folds = 10
    test_frac = 0
    forest_size = 50
    train_threshold = 1
    test_threshold = 0.9
    crange = (1e-4, 0.001, 0.01, 0.1, 1, 10)
    maxfeat = (None, 'sqrt', 0.5*n_features, 0.2*n_features, 0.1*n_features)
    maxdepth = (None, 5, 10, 20)
    classifiers = {
                  'Logistic Regression (L1 penalty)': (LogisticRegression(penalty='l1', random_state=seed), {'C': crange}),
                  'Logistic Regression (L2 penalty)': (LogisticRegression(penalty='l2', random_state=seed), {'C': crange}),
                  'Linear SVM (L1 loss, L2 penalty)': (LinearSVC(loss='l1', penalty='l2', random_state=seed), {'C': crange}),
                  'Linear SVM (L2 loss, L1 penalty)': (LinearSVC(loss='l2', penalty='l1', dual=False, random_state=seed), {'C': crange}),
                  'Linear SVM (L2 loss, L2 penalty)': (LinearSVC(loss='l2', penalty='l2', random_state=seed), {'C': crange, 'dual': (True, False)}),
                  'Gaussian SVM': (SVC(probability=True), {'C': crange, 'gamma': (0, 0.001, 0.01, 0.1, 1)}),
                  'Linear Discriminant Analysis ': (LDA(), {}),
                  'Quadratic Discriminant Analysis ': (QDA(), {}),
                  'Gaussian NB': (GaussianNB(), {}),
                  'Decision Tree (Gini Impurity)': (DecisionTreeClassifier(random_state=seed), {'max_depth': maxdepth, 'max_features': maxfeat}),
                  'Decision Tree (Mutual Information)': (DecisionTreeClassifier(criterion='entropy', random_state=seed), {'max_depth': maxdepth, 'max_features': maxfeat}),
                  'Random Forest (Gini Impurity)': (RandomForestClassifier(n_estimators=forest_size, random_state=seed), {'max_depth': maxdepth, 'max_features': maxfeat, 'bootstrap': (True, False)}),
                  'Random Forest (Mutual Information)': (RandomForestClassifier(n_estimators=forest_size, criterion='entropy', random_state=seed), {'max_depth': maxdepth, 'max_features': maxfeat, 'bootstrap': (True, False)}),
                  'Extremely Randomized Trees (Gini Impurity)': (ExtraTreesClassifier(n_estimators=forest_size, random_state=seed), {'max_depth': maxdepth, 'max_features': maxfeat, 'bootstrap': (True, False)}),
                  'Extremely Randomized Trees (Mutual Information)': (ExtraTreesClassifier(n_estimators=forest_size, criterion='entropy', random_state=seed), {'max_depth': maxdepth, 'max_features': maxfeat, 'bootstrap': (True, False)}),
                  'Gradient Boosting': (GradientBoostingClassifier(n_estimators=50, random_state=seed), {'learning_rate': (0.02, 0.05, 0.1, 0.2), 'subsample': (0.1, 0.2, 0.5, 1), 'max_depth': (5, 10, 20), 'max_features': (None, 0.5*n_features, 0.2*n_features, 0.1*n_features)})
                  }

    n_classifiers = len(classifiers)
        
    X = np.array(train, dtype=np.float)
    y = np.array(train_labels, dtype=np.float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_oos = scaler.transform(np.array(oos, dtype=np.float))

    out_train_all = None
    for name, (classifier, params) in sorted(classifiers.items()):
        tm = clock()
        if params and folds > 1:
            clf = GridSearchCV(classifier, params, cv=folds)
            clf.fit(X_train, y_train)    
            print(clf.best_params_)
            out_train = clf.predict(X_train)
            out_test = clf.predict(X_test)
            out_oos = clf.predict(X_oos)
            if hasattr(classifier, 'predict_proba'):
                out_train_prob = clf.predict_proba(X_train)[:,1]
                out_test_prob = clf.predict_proba(X_test)[:,1]
                out_oos_prob = clf.predict_proba(X_oos)[:,1]
            else:
                out_train_prob = out_train
                out_test_prob = out_test
                out_oos_prob = out_oos
        else:
            classifier.fit(X_train, y_train)
            out_train = classifier.predict(X_train)
            out_test = classifier.predict(X_test)
            out_oos = classifier.predict(X_oos)
            if hasattr(classifier, 'predict_proba'):
                out_train_prob = classifier.predict_proba(X_train)[:,1]
                out_test_prob = classifier.predict_proba(X_test)[:,1]
                out_oos_prob = classifier.predict_proba(X_oos)[:,1]
            else:
                out_train_prob = out_train
                out_test_prob = out_test
                out_oos_prob = out_oos
        train_acc = np.mean(out_train == y_train)
        test_acc = np.mean(out_test == y_test)
        print('accuracy for ' + name + ': train=' + str(round(train_acc,2)) + ' test=' + str(round(test_acc,2)))
        if train_acc >= train_threshold and (test_acc >= test_threshold or test_acc != test_acc):
            print('passed thresholds!')
            if out_train_all is None:
                out_train_all = out_train
                out_test_all = out_test
                out_oos_all = out_oos
                out_train_prob_all = out_train_prob
                out_test_prob_all = out_test_prob
                out_oos_prob_all = out_oos_prob
            else:
                out_train_all = np.column_stack((out_train_all, out_train))
                out_test_all = np.column_stack((out_test_all, out_test))
                out_oos_all = np.column_stack((out_oos_all, out_oos))
                out_train_prob_all = np.column_stack((out_train_prob_all, out_train_prob))
                out_test_prob_all = np.column_stack((out_test_prob_all, out_test_prob))
                out_oos_prob_all = np.column_stack((out_oos_prob_all, out_oos_prob))
        print(str(round((clock()-tm))) + ' sec \n')
    if out_train_all is None:
        print('no classifier has passed the thresholds')
    else:
        if out_train_all.ndim == 2:
            out_train_maj = stats.mode(out_train_all.T)[0][0]
            if len(y_test) > 0:
                out_test_maj = stats.mode(out_test_all.T)[0][0]
            else:
                out_test_maj = out_test_all
            out_oos_maj = stats.mode(out_oos_all.T)[0][0]
            out_train_prob_mean = out_train_prob_all.mean(axis=1)
            out_test_prob_mean = out_test_prob_all.mean(axis=1)
            out_oos_prob_mean = out_oos_prob_all.mean(axis=1)
            n_used = out_train_all.shape[1]
        else:
            out_train_maj = out_train_all
            out_test_maj = out_test_all
            out_oos_maj = out_oos_all
            out_train_prob_mean = out_train_prob_all
            out_test_prob_mean = out_test_prob_all
            out_oos_prob_mean = out_oos_prob_all
            n_used = 1
        train_maj = np.mean(out_train_maj == y_train)
        train_prob_mean = np.mean((out_train_prob_mean>=0.5) == y_train)
        test_maj = np.mean(out_test_maj == y_test)
        test_prob_mean = np.mean((out_test_prob_mean>=0.5) == y_test)
        print('accuracy for Majority Vote: train=' + str(round(train_maj,2)) + ' test=' + str(round(test_maj,2)))
        print('accuracy for Mean Probabilities: train=' + str(round(train_prob_mean,2)) + ' test=' + str(round(test_prob_mean,2)))
        print(str(n_used) + ' classifiers used')
        with open('submit.csv', 'w', newline='') as csv:
            writer(csv).writerows([int(pred>=0.5)] for pred in out_oos_prob_mean)
    input()

    #cross validation with relative fraction of train to test; kfold vs anti-k fold
    #neighbour methods/more classifiers
    #how do ensemble methods work vs cv? oob?
    #save clfs
    #calibrate GBM
    #unsupervised clustering on test/semisupervised
    #parallelization not working, need main function?
    #look at correlation between sensors, weights?
    #fix scikit to allow full random seed for all functions
    #deal with ordered and non-ordered categories
