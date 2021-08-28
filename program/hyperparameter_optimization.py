from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def hyper_opt_knn(X, y, folds, classifier: bool = True):
    """
    ottimizzazione dei parametri di un modello KNN
    :param X: X dataframe - valori noti
    :param y: y column(s) - valori da predire
    :param folds: numero di folds per la cross-validation
    :param classifier: True se classificatore KNN, False se regressore KNN
    :return: parametri ottimizzati
    """
    leaf_size = list(range(1, 50))
    n_neighbors = list(range(1, 30))
    p = [1, 2]
    weights = ['uniform', 'distance']

    hyperparameters = dict(knn__leaf_size=leaf_size, knn__n_neighbors=n_neighbors, knn__p=p, knn__weights=weights)

    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    if classifier:
        pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
    else:
        pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
    clf = GridSearchCV(param_grid=hyperparameters, cv=kf, n_jobs=-1, estimator=pipeline)
    # Fit the model
    best_model = clf.fit(X, y)

    # Print The value of best Hyperparameters
    print('Best leaf_size:', best_model.best_estimator_.get_params()['knn__leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['knn__p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['knn__n_neighbors'])
    print('Best weights:', best_model.best_estimator_.get_params()['knn__weights'])

    best_hyper_params = {'leaf_size': best_model.best_estimator_.get_params()['knn__leaf_size'],
                         'p': best_model.best_estimator_.get_params()['knn__p'],
                         'n_neighbors': best_model.best_estimator_.get_params()['knn__n_neighbors'],
                         'weights': best_model.best_estimator_.get_params()['knn__weights']}
    return best_hyper_params


def hyper_opt_rf(X, y, folds):
    """
    ottimizzazione dei parametri di un modello RF
    :param X: X dataframe - valori noti
    :param y: y column(s) - valori da predire
    :param folds: numero di folds per la cross-validation
    :return: parametri ottimizzati
    """
    n_estimators = [100, 150, 200]  # The number of trees in the forest
    criterion = ['gini', 'entropy']  # The function to measure the quality of a split. Supported criteria are “gini” for
                                    # the Gini impurity and “entropy” for the information gain
    max_features = ['auto', 'sqrt', 'log2']  # The number of features to consider when looking for the best split
    min_samples_split = [5, 10, 15, 20]  # Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as
                                    # relative reduction in impurity. If None then unlimited number of leaf nodes
    min_samples_leaf = [8, 16, 32, 64]  # A node will be split if this split induces a decrease of the impurity greater than
                                  # or equal to this value
    bootstrap = [True, False]  # Whether bootstrap samples are used when building trees. If False, the whole dataset is
                               # used to build each tree

    hyperparameters = dict(rf__n_estimators=n_estimators, rf__criterion=criterion, rf__max_features=max_features,
                           rf__min_samples_split=min_samples_split, rf__min_samples_leaf=min_samples_leaf,
                           rf__bootstrap=bootstrap)

    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=0, n_jobs=-1))])
    clf = GridSearchCV(param_grid=hyperparameters, cv=kf, n_jobs=-1, estimator=pipeline)
    # Fit the model
    best_model = clf.fit(X, y)
    # Print The value of best Hyperparameters
    print('Best n estimators:', best_model.best_estimator_.get_params()['rf__n_estimators'])
    print('Best criteration:', best_model.best_estimator_.get_params()['rf__criterion'])
    print('Best max features:', best_model.best_estimator_.get_params()['rf__max_features'])
    print('Best min samples split:', best_model.best_estimator_.get_params()['rf__min_samples_split'])
    print('Best min samples leaf:', best_model.best_estimator_.get_params()['rf__min_samples_leaf'])
    print('Best bootstrap:', best_model.best_estimator_.get_params()['rf__bootstrap'])
    best_hyper_params = {'n_estimators': best_model.best_estimator_.get_params()['rf__n_estimators'],
                         'criterion': best_model.best_estimator_.get_params()['rf__criterion'],
                         'max_features': best_model.best_estimator_.get_params()['rf__max_features'],
                         'min_samples_split': best_model.best_estimator_.get_params()['rf__min_samples_split'],
                         'min_samples_leaf': best_model.best_estimator_.get_params()['rf__min_samples_leaf'],
                         'bootstrap': best_model.best_estimator_.get_params()['rf__bootstrap']}
    return best_hyper_params
