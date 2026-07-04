import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from program.config import KNN_HYPERS, KNN_IMPUTER_NEIGHBORS, TARGET_GRADE, TARGET_SCORE
from program.cross_validation import kFold_cross_validation_knn, kFold_cross_validation_rf
from program.hyperparameter_optimization import hyper_opt_knn, hyper_opt_rf


class NutriScoreKNN:
    """
    Classificatore KNN del Nutriscore addestrato UNA sola volta.

    Rispetto alla vecchia funzione ``knn_model`` (che ri-addestrava imputer,
    scaler e modello ad ogni predizione), qui l'addestramento avviene nel
    costruttore e ogni predizione riusa gli oggetti gia' fittati.
    """

    def __init__(self, df, col_list, hypers=None):
        hypers = hypers or KNN_HYPERS
        self.col_list = list(col_list)

        X = df[self.col_list]
        y = df[TARGET_GRADE]

        # imputer e scaler fittati sui soli dati di training
        self.imputer = KNNImputer(n_neighbors=KNN_IMPUTER_NEIGHBORS).fit(X)
        X_imputed = self.imputer.transform(X)
        self.scaler = StandardScaler().fit(X_imputed)

        self.model = KNeighborsClassifier(
            n_neighbors=hypers["n_neighbors"],
            leaf_size=hypers["leaf_size"],
            weights=hypers["weights"],
            p=hypers["p"],
            n_jobs=-1,
        )
        self.model.fit(self.scaler.transform(X_imputed), y)

    def predict(self, values: dict) -> str:
        """Predice il Nutriscore (A-E) per i valori forniti dall'utente."""
        row = [values.get(col, np.nan) for col in self.col_list]
        row_df = pd.DataFrame([row], columns=self.col_list)
        row_imputed = self.imputer.transform(row_df)
        row_scaled = self.scaler.transform(row_imputed)
        return str(self.model.predict(row_scaled)[0])


def knn_model(df, col_list, hypers, values):
    """
    Wrapper retro-compatibile: costruisce un ``NutriScoreKNN`` e predice.
    Preferire l'uso diretto della classe per evitare ri-addestramenti.
    :return: nutriscore predetto (stringa)
    """
    return NutriScoreKNN(df, col_list, hypers).predict(values)


def knn_classifier(df, col_list, folds, hyp_opt: bool = False):
    """
    testa un classificatore KNN
    :return: valore medio di accuracy su tutte le fold
    """
    X_food = df[col_list]
    y_food = df[TARGET_GRADE]
    hypers = dict(KNN_HYPERS)
    if hyp_opt:
        print("> Hyperparameter optimization started for knn_classifier...")
        hypers = hyper_opt_knn(X_food, y_food, folds, classifier=True)
    return kFold_cross_validation_knn(X_food, y_food, hypers, classifier=True, splits=folds)


def knn_regressor(df, col_list, folds, hyp_opt: bool = False):
    """
    testa un regressore KNN
    :return: valore medio di accuracy su tutte le fold
    """
    X_food = df[col_list]
    y_food = df[TARGET_SCORE]
    hypers = {'leaf_size': 1, 'p': 1, 'n_neighbors': 5, 'weights': 'distance'}
    if hyp_opt:
        print("> Hyperparameter optimization started for knn_regressor...")
        hypers = hyper_opt_knn(X_food, y_food, folds, classifier=False)
    return kFold_cross_validation_knn(X_food, y_food, hypers, classifier=False, splits=folds)


def rf_classifier(df, col_list, folds, hyp_opt: bool = False):
    """
    testa un classificatore RF
    :return: valore medio di accuracy su tutte le fold
    """
    X_food = df[col_list]
    y_food = df[TARGET_GRADE]
    # 'log2' sostituisce 'auto', rimosso in sklearn >= 1.3
    hypers = {'n_estimators': 150, 'criterion': 'gini', 'max_features': 'log2', 'min_samples_split': 2,
              'min_samples_leaf': 1, 'bootstrap': True}
    if hyp_opt:
        print("> Hyperparameter optimization started for random_forest...")
        hypers = hyper_opt_rf(X_food, y_food, folds)
    return kFold_cross_validation_rf(X_food, y_food, hypers, folds)

