import statistics

from pgmpy.estimators import BayesianEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Robust Bayesian network import (name changed across pgmpy versions)
try:  # pgmpy >= 0.1.26
    from pgmpy.models import DiscreteBayesianNetwork as _BayesianNetwork
except ImportError:  # pragma: no cover
    try:
        from pgmpy.models import BayesianNetwork as _BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianModel as _BayesianNetwork


def KFold_splitting(X, y, splits=10):
    """
    Split dataset into train and test sets.
    :param X: X DataFrame - known values
    :param y: y column(s) - target values to predict
    :param splits: number of folds
    :return: list of fold combinations (train/test sets)
    """
    # for column in X.columns:
    #     X[column] = X[column].astype('int64')
    kf = KFold(n_splits=splits, shuffle=True, random_state=0)
    folds = []
    for train_index, test_index in kf.split(X):
        l = []
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        l.append(X_train)
        l.append(X_test)
        l.append(y_train)
        l.append(y_test)
        folds.append(l)
    return folds


def kFold_cross_validation_bayesian(X, y, splits=10):
    """
    Cross-validation for the Bayesian network.
    :param X: X DataFrame - known values
    :param y: y column(s) - target values to predict
    :param splits: number of folds
    :return: mean accuracy
    """
    folds = KFold_splitting(X, y, splits)
    scores = []
    for fold in folds:
        model = _BayesianNetwork(
            [('fat_value', 'saturated-fat_value'), ('carbohydrates_value', 'sugars_value'),
             ('proteins_value', 'salt_value'),
             ('fat_value', 'energy_value'), ('carbohydrates_value', 'energy_value'), ('salt_value', 'nutri_value'),
             ('energy_value', 'nutri_value'), ('saturated-fat_value', 'nutri_value'), ('sugars_value', 'nutri_value')])
        predict_data = fold[1].copy()
        real_data = fold[3].copy()
        X['nutri_value'] = y
        model.fit(X, estimator=BayesianEstimator, prior_type="BDeu")
        y_pred = model.predict(predict_data)
        scores.append(accuracy_score(y_pred, real_data))
    avg_scores = statistics.mean(scores)
    std_scores = statistics.stdev(scores)
    print('Accuracy: %.3f (Standard Dev: %.3f)' % (avg_scores, std_scores))
    return avg_scores


def kFold_cross_validation_knn(X, y, hypers, classifier: bool = True, splits=10):
    """
    Run K-fold cross-validation on a KNN classifier or regressor.
    :param hypers: optimal hyperparameter values
    :param X: X DataFrame - known values
    :param y: y column(s) - target values to predict
    :param classifier: True for KNN Classifier (default), False for KNN Regressor
    :param splits: number of folds
    :return: mean accuracy
    """
    folds = KFold_splitting(X, y, splits)
    scores = []
    if classifier:
        print('Cross validation (knn_classifier) via K-Fold [{} splits]'.format(splits))
        for fold in folds:
            knn_class = KNeighborsClassifier(n_neighbors=hypers['n_neighbors'],
                                             leaf_size=hypers['leaf_size'],
                                             weights=hypers['weights'],
                                             p=hypers['p'],
                                             n_jobs=-1)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(fold[0])
            X_test_scaled = scaler.transform(fold[1])
            knn_class.fit(X_train_scaled, fold[2])
            accuracy = knn_class.score(X_test_scaled, fold[3])
            scores.append(accuracy)
    else:
        print('Cross validation (knn_regressor) via K-Fold [{} splits]'.format(splits))
        for fold in folds:
            knn_regr = KNeighborsRegressor(n_neighbors=hypers['n_neighbors'],
                                           leaf_size=hypers['leaf_size'],
                                           weights=hypers['weights'],
                                           p=hypers['p'],
                                           n_jobs=-1)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(fold[0])
            X_test_scaled = scaler.transform(fold[1])
            knn_regr.fit(X_train_scaled, fold[2])
            predicted = knn_regr.predict(X_test_scaled)
            y_pred = nutriscore_converter(predicted)
            y_test = nutriscore_converter(fold[3])
            accuracy = accuracy_score(y_test, y_pred)
            scores.append(accuracy)
    avg_scores = statistics.mean(scores)
    std_scores = statistics.stdev(scores)
    print('Accuracy: %.3f (Standard Dev: %.3f)' % (avg_scores, std_scores))
    return avg_scores


def kFold_cross_validation_rf(X, y, hypers, splits=10):
    """
    Run K-fold cross-validation on a Random Forest model.
    :param hypers: optimal hyperparameter values
    :param X: X DataFrame - known values
    :param y: y column(s) - target values to predict
    :param splits: number of folds
    :return: mean accuracy
    """
    folds = KFold_splitting(X, y, splits)
    print('Cross validation (random_forest) via K-Fold [{} splits]'.format(splits))
    scores = []
    for fold in folds:
        random_for = RandomForestClassifier(n_estimators=hypers['n_estimators'],
                                            criterion=hypers['criterion'],
                                            max_features=hypers['max_features'],
                                            min_samples_split=hypers['min_samples_split'],
                                            min_samples_leaf=hypers['min_samples_leaf'],
                                            bootstrap=hypers['bootstrap'],
                                            random_state=0,
                                            n_jobs=-1)
        random_for.fit(fold[0], fold[2])
        accuracy = random_for.score(fold[1], fold[3])
        scores.append(accuracy)

    avg_scores = statistics.mean(scores)
    std_scores = statistics.stdev(scores)
    print('Accuracy: %.3f (Standard Dev: %.3f)' % (avg_scores, std_scores))
    return avg_scores


def nutriscore_converter(y):
    scores = []
    for score in y:
        if score <= -1:
            scores.append('a')
        elif score <= 2:
            scores.append('b')
        elif score <= 10:
            scores.append('c')
        elif score <= 18:
            scores.append('d')
        else:
            scores.append('e')
    return scores

