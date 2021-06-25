from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def hyper_opt_knn(X, y, folds, classifier: bool = True):
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
    n_estimators = [100, 125, 150, 175, 200]
    criterion = ['gini', 'entropy']
    max_features = ['auto', 'sqrt', 'log2']
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

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
