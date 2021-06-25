import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from program.cross_validation import kFold_cross_validation_knn, kFold_cross_validation_rf
from program.hyperparameter_optimization import hyper_opt_knn, hyper_opt_rf


def knn_classifier(df, col_list, folds, hyp_opt: bool = False):
    """
    testa un classificatore KNN
    :param df: dataframe in input
    :param col_list: lista di nomi di features da considerare
    :param folds: numero di fold
    :param hyp_opt: True se ottimizzazione degli iperparametri richiesta, False altrimenti
    :return: valore medio di accuracy su tutte le fold
    """
    X_food = df[col_list]
    y_food = df['nutriscore_grade']
    # hyperparameter optimization
    hypers = {'leaf_size': 1, 'p': 1, 'n_neighbors': 7, 'weights': 'distance'}
    if hyp_opt:
        print("> Hyperparameter optimization started for knn_classifier...")
        hypers = hyper_opt_knn(X_food, y_food, folds, classifier=True)
    # cross-validation
    return kFold_cross_validation_knn(X_food, y_food, hypers, classifier=True, splits=folds)


def knn_model(df, col_list, hypers, values):
    knn_class = KNeighborsClassifier(n_neighbors=hypers['n_neighbors'],
                                     leaf_size=hypers['leaf_size'],
                                     weights=hypers['weights'],
                                     p=hypers['p'],
                                     n_jobs=-1)
    new_col = []
    for col in col_list:
        if col in values:
            new_col.append(values[col])
        else:
            new_col.append(np.nan)
    X = df[col_list]
    dataX = df[col_list]
    y = df['nutriscore_grade']
    X.loc[-1] = new_col
    imputer = KNNImputer(n_neighbors=7)
    X = imputer.fit_transform(X)
    df_predict = pd.DataFrame(columns=col_list)
    df_predict.loc[0] = X[-1]
    scaler = StandardScaler()
    dataX = scaler.fit_transform(dataX)
    knn_class.fit(dataX, y)

    df_predict = scaler.transform(df_predict)
    return knn_class.predict(df_predict)


def knn_regressor(df, col_list, folds, hyp_opt: bool = False):
    """
    testa un regressore KNN
    :param df: dataframe in input
    :param col_list: lista di nomi di features da considerare
    :param folds: numero di fold
    :param hyp_opt: True se ottimizzazione degli iperparametri richiesta, False altrimenti
    :return: valore medio di accuracy su tutte le fold
    """
    X_food = df[col_list]
    y_food = df['nutriscore_score']
    # hyperparameter optimization
    hypers = {'leaf_size': 1, 'p': 1, 'n_neighbors': 5, 'weights': 'distance'}
    if hyp_opt:
        print("> Hyperparameter optimization started for knn_regressor...")
        hypers = hyper_opt_knn(X_food, y_food, folds, classifier=False)
    # cross-validation
    return kFold_cross_validation_knn(X_food, y_food, hypers, classifier=False, splits=folds)


def rf_classifier(df, col_list, folds, hyp_opt: bool = False):
    """
    testa un classificatore RF
    :param df: dataframe in input
    :param col_list: lista di nomi di features da considerare
    :param folds: numero di fold
    :param hyp_opt: True se ottimizzazione degli iperparametri richiesta, False altrimenti
    :return: valore medio di accuracy su tutte le fold
    """
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
