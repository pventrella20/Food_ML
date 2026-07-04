import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import MinMaxScaler
from program.cross_validation import kFold_cross_validation_bayesian
# pgmpy ha rinominato BayesianModel -> BayesianNetwork -> DiscreteBayesianNetwork
# a seconda della versione. Import robusto con fallback.
try:  # pgmpy >= 0.1.26
    from pgmpy.models import DiscreteBayesianNetwork as _BayesianNetwork
except ImportError:  # pragma: no cover
    try:
        from pgmpy.models import BayesianNetwork as _BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianModel as _BayesianNetwork
# Struttura (archi) della rete bayesiana, definita una sola volta
BAYES_EDGES = [
    ("fat_value", "saturated-fat_value"), ("carbohydrates_value", "sugars_value"),
    ("proteins_value", "salt_value"), ("fat_value", "energy_value"),
    ("carbohydrates_value", "energy_value"), ("salt_value", "nutriscore_value"),
    ("energy_value", "nutriscore_value"), ("saturated-fat_value", "nutriscore_value"),
    ("sugars_value", "nutriscore_value"),
]
features = ["energy", "fat", "saturated-fat", "carbohydrates", "sugars", "proteins", "salt", "nutriscore"]
features_ext = ["energy_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g",
                "proteins_100g", "salt_100g", "nutriscore_score"]
def _fit_bayesian(model, data, prior_type="BDeu", equivalent_sample_size=5):
    """
    Addestra i CPD di una rete bayesiana con estimator bayesiano (prior BDeu).
    In pgmpy >= 1.0 ``model.fit`` non accetta piu' ``prior_type``: i parametri
    vengono quindi stimati con ``BayesianEstimator.get_parameters`` e aggiunti
    al modello. Mantiene un fallback per le versioni precedenti.
    """
    try:
        estimator = BayesianEstimator(model, data)
        cpds = estimator.get_parameters(prior_type=prior_type,
                                        equivalent_sample_size=equivalent_sample_size)
        model.add_cpds(*cpds)
    except TypeError:  # pragma: no cover - pgmpy < 1.0
        model.fit(data, estimator=BayesianEstimator, prior_type=prior_type)
    return model
def values_to_range(new_food_df, f_old, f_val, i, cont, step):
    """converte in range da 0 a 4 i valori (normalizzati) di un dataframe"""
    if i == 0:
        new_food_df.loc[new_food_df[f_old] < step / 10, f_val] = cont
    elif i == (10 - step):
        new_food_df.loc[new_food_df[f_old] >= i / 10, f_val] = cont
    else:
        new_food_df.loc[
            (new_food_df[f_old] >= i / 10) & (new_food_df[f_old] < (i + step) / 10), f_val] = cont
    return new_food_df
def bayesian_preprocessing(food_df, values=None):
    """operazioni preliminari per rendere il dataframe idoneo ad una rete bayesiana"""
    new_food_df = food_df[features_ext]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(new_food_df)
    df_predict = None
    if values is not None:
        col_list = []
        feat_sel = []
        new_values = []
        for col in features:
            if col in ("energy", "proteins", "fat", "carbohydrates"):
                col_list.append(col + "_100g")
                feat_sel.append(col)
                new_values.append(values[col + "_100g"])
            else:
                col_list.append(col + "_100g")
                new_values.append(np.nan)
        df_predict = pd.DataFrame(columns=col_list)
        df_predict.loc[0] = new_values
        scaled_df = scaler.transform(df_predict)
        df_predict = pd.DataFrame(scaled_df, index=df_predict.index, columns=df_predict.columns)
        for col in feat_sel:
            cont, step = 0, 2
            for i in range(0, 10, step):
                df_predict = values_to_range(df_predict, col + "_100g", col + "_value", i, cont, step)
                cont += 1
    new_food_df = pd.DataFrame(scaled_features, index=new_food_df.index, columns=new_food_df.columns)
    for feat in features:
        if "nutriscore" in feat:
            new_food_df.loc[new_food_df["nutriscore_score"] < 0.5, "nutriscore_value"] = 0
            new_food_df.loc[new_food_df["nutriscore_score"] >= 0.5, "nutriscore_value"] = 1
        else:
            cont, step = 0, 2
            for i in range(0, 10, step):
                new_food_df = values_to_range(new_food_df, feat + "_100g", feat + "_value", i, cont, step)
                cont += 1
    new_food_df = new_food_df.drop(features_ext, axis=1)
    if values is not None:
        return new_food_df, df_predict
    return new_food_df
class BayesianHealthModel:
    """Rete bayesiana per la salubrita' di un alimento, addestrata UNA volta."""
    def __init__(self, food_df):
        self.scaler = MinMaxScaler().fit(food_df[features_ext])
        train_df = self._discretize_training(food_df)
        self.model = _BayesianNetwork(BAYES_EDGES)
        _fit_bayesian(self.model, train_df)
        self.infer = VariableElimination(self.model)
    def _discretize_training(self, food_df):
        scaled = self.scaler.transform(food_df[features_ext])
        df = pd.DataFrame(scaled, index=food_df.index, columns=features_ext)
        for feat in features:
            if "nutriscore" in feat:
                df.loc[df["nutriscore_score"] < 0.5, "nutriscore_value"] = 0
                df.loc[df["nutriscore_score"] >= 0.5, "nutriscore_value"] = 1
            else:
                cont, step = 0, 2
                for i in range(0, 10, step):
                    df = values_to_range(df, feat + "_100g", feat + "_value", i, cont, step)
                    cont += 1
        return df.drop(features_ext, axis=1)
    def _discretize_prediction(self, values):
        known = ["energy", "proteins", "fat", "carbohydrates"]
        col_list = list(features_ext)
        row = [values[c] if c in ("energy_100g", "proteins_100g", "fat_100g", "carbohydrates_100g") else np.nan for c in features_ext]
        df_predict = pd.DataFrame([row], columns=col_list)
        scaled = self.scaler.transform(df_predict)
        df_predict = pd.DataFrame(scaled, index=df_predict.index, columns=col_list)
        for feat in known:
            cont, step = 0, 2
            for i in range(0, 10, step):
                df_predict = values_to_range(df_predict, feat + "_100g", feat + "_value", i, cont, step)
                cont += 1
        return df_predict
    def predict(self, values):
        """Restituisce una stringa decisionale sulla salubrita' dell'alimento."""
        predict_data = self._discretize_prediction(values)
        q = self.infer.query(
            variables=["nutriscore_value"],
            evidence={
                "proteins_value": predict_data.loc[0, "proteins_value"],
                "fat_value": predict_data.loc[0, "fat_value"],
                "carbohydrates_value": predict_data.loc[0, "carbohydrates_value"],
                "energy_value": predict_data.loc[0, "energy_value"],
            },
        )
        if int(np.argmax(q.values, axis=0)) == 0:
            return "Alimento prevalentemente salutare"
        return "Alimento prevalentemente nocivo"
def bayesianNetwork(food_df, values):
    """Wrapper retro-compatibile: costruisce una BayesianHealthModel e predice."""
    return BayesianHealthModel(food_df).predict(values)
def bayesianTest(food_df, folds):
    """test di una rete bayesiana tramite cross-validation"""
    new_food_df = bayesian_preprocessing(food_df)
    X_food = new_food_df.drop("nutriscore_value", axis=1)
    y_food = new_food_df["nutriscore_value"]
    return kFold_cross_validation_bayesian(X_food, y_food, folds)

