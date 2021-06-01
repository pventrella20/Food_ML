import statistics

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def KFold_splitting(X, y, splits=10):
    """
    divisione del dataset in train e test set
    :param X: X dataframe - valori noti
    :param y: y column(s) - valori da predire
    :param splits: numero di folds da utilizzare
    :return: lista delle varie combinazioni di folds (train/test sets)
    """
    kf = KFold(n_splits=splits, shuffle=True, random_state=0)
    folds = []
    for train_index, test_index in kf.split(X):
        l = []
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        l.append(X_train_scaled)
        l.append(X_test_scaled)
        l.append(y_train)
        l.append(y_test)
        folds.append(l)
    return folds


def kFold_cross_validation_knn(X, y, hypers, classifier: bool=True, splits=10):
    """
    esegue cross validation utilizzando la tecnica K-fold su un knn Classifier o Regressor
    :param hypers: valori ottimali degli iperparametri
    :param X: X dataframe - valori noti
    :param y: y column(s) - valori da predire
    :param classifier: True se knn Classifier (default), False se knn Regressor
    :param splits: numero di folds
    :return:
    """
    folds = KFold_splitting(X, y, splits)
    print('Cross validation via K-Fold [{} splits]'.format(splits))
    scores = []
    if classifier:
        for fold in folds:
            knn_class = KNeighborsClassifier(n_neighbors=hypers['n_neighbors'], leaf_size=hypers['leaf_size'],
                                             weights=hypers['weights'], p=hypers['p'])
            knn_class.fit(fold[0], fold[2])
            accuracy = knn_class.score(fold[1], fold[3])
            scores.append(accuracy)
    else:
        for fold in folds:
            knn_regr = KNeighborsRegressor(n_neighbors=hypers['n_neighbors'], leaf_size=hypers['leaf_size'],
                                             weights=hypers['weights'], p=hypers['p'])
            knn_regr.fit(fold[0], fold[2])
            predicted = knn_regr.predict(fold[1])
            y_pred = nutriscore_converter(predicted)
            y_test = nutriscore_converter(fold[3])
            accuracy = accuracy_score(y_test, y_pred)
            scores.append(accuracy)
    avg_scores = statistics.mean(scores)
    std_scores = statistics.stdev(scores)
    print('Accuracy: %.3f (Standard Dev: %.3f)' % (avg_scores, std_scores))

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