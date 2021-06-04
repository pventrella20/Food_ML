from program.cross_validation import kFold_cross_validation_knn, kFold_cross_validation_rf
from program.hyperparameter_optimization import hyper_opt_knn, hyper_opt_rf


def knn_classifier(df, col_list, folds, hyp_opt: bool = False):
    X_food = df[col_list]
    y_food = df['nutriscore_grade']
    # hyperparameter optimization
    hypers = {'leaf_size': 1, 'p': 1, 'n_neighbors': 7, 'weights': 'distance'}
    if hyp_opt:
        print("> Hyperparameter optimization started for knn_classifier...")
        hypers = hyper_opt_knn(X_food, y_food, folds, classifier=True)
    # cross-validation
    return kFold_cross_validation_knn(X_food, y_food, hypers, classifier=True, splits=folds)


def knn_regressor(df, col_list, folds, hyp_opt: bool = False):
    X_food = df[col_list]
    y_food = df['nutriscore_score']
    # hyperparameter optimization
    hypers = {'leaf_size': 1, 'p': 1, 'n_neighbors': 7, 'weights': 'distance'}
    if hyp_opt:
        print("> Hyperparameter optimization started for knn_regressor...")
        hypers = hyper_opt_knn(X_food, y_food, folds, classifier=False)
    # cross-validation
    return kFold_cross_validation_knn(X_food, y_food, hypers, classifier=False, splits=folds)


def rf_classifier(df, col_list, folds, hyp_opt: bool = False):
    X_food = df[col_list]
    y_food = df['nutriscore_grade']
    # parametri migliori
    hypers = {'n_estimators': 150, 'criterion': 'gini', 'max_features': 'log2', 'min_samples_split': 2,
              'min_samples_leaf': 1, 'bootstrap': True}
    if hyp_opt:
        print("> Hyperparameter optimization started for random_forest...")
        hypers = hyper_opt_rf(X_food, y_food, folds)
    # cross-validation
    return kFold_cross_validation_rf(X_food, y_food, hypers, folds)
