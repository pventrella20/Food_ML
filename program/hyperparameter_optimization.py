from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def hyper_opt_knn(X, y):
    leaf_size = list(range(1, 50))
    n_neighbors = list(range(1, 30))
    p = [1, 2]
    weights = ['uniform', 'distance']

    hyperparameters = dict(knn__leaf_size=leaf_size, knn__n_neighbors=n_neighbors, knn__p=p, knn__weights=weights)

    kf = KFold(n_splits=26, shuffle=True, random_state=0)
    pipeline = Pipeline([('scaler',  StandardScaler()), ('knn', KNeighborsClassifier())])
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
