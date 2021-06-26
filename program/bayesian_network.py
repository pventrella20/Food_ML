import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from sklearn.preprocessing import MinMaxScaler

from program.cross_validation import kFold_cross_validation_bayesian

features = ['energy', 'fat', 'saturated-fat', 'carbohydrates', 'sugars', 'proteins', 'salt', 'nutriscore']
features_ext = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g',
                'salt_100g', 'nutriscore_score']


def bayesian_preprocessing(food_df, values=None):
    """
    operazioni preliminari da effettuare sul dataframe per renderlo idoneo ad una rete bayesiana
    :param food_df: dataframe in input
    :param values: None se non c'é bisogno di predizione
    :return: dataset idoneo ad una rete bayesiana
    """
    new_food_df = food_df[features_ext]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(new_food_df)
    if values is not None:
        col_list = []
        feat_sel = []
        new_values = []
        for col in features:
            if col == 'energy' or col == 'proteins' or col == 'fat' or col == 'carbohydrates':
                col_list.append(col + '_100g')
                feat_sel.append(col)
                new_values.append(values[col + '_100g'])
            else:
                col_list.append(col + '_100g')
                new_values.append(np.nan)
        df_predict = pd.DataFrame(columns=col_list)
        df_predict.loc[0] = new_values
        scaled_df = scaler.transform(df_predict)
        df_predict = pd.DataFrame(scaled_df, index=df_predict.index, columns=df_predict.columns)
        for col in feat_sel:
            cont = 0
            step = 2
            for i in range(0, 10, step):
                df_predict = values_to_range(df_predict, col + '_100g', col + '_value', i, cont, step)
                cont += 1
    new_food_df = pd.DataFrame(scaled_features, index=new_food_df.index, columns=new_food_df.columns)
    for feat in features:
        cont = 0
        step = 2
        if 'nutriscore' in feat:
            new_food_df.loc[new_food_df['nutriscore_score'] < 0.5, 'nutriscore_value'] = 0
            new_food_df.loc[new_food_df['nutriscore_score'] >= 0.5, 'nutriscore_value'] = 1
        else:
            for i in range(0, 10, step):
                new_food_df = values_to_range(new_food_df, feat + '_100g', feat + '_value', i, cont, step)
                cont += 1
    new_food_df = new_food_df.drop(features_ext, axis=1)
    if values is not None:
        return new_food_df, df_predict
    else:
        return new_food_df


def bayesianNetwork(food_df, values):
    """
    previsione tramite rete bayesiana
    :param food_df: dataframe in input
    :param values: valori da predire
    :return: stringa decisionale
    """
    new_food_df, predict_data = bayesian_preprocessing(food_df, values)
    model = BayesianModel(
        [('fat_value', 'saturated-fat_value'), ('carbohydrates_value', 'sugars_value'),
         ('proteins_value', 'salt_value'),
         ('fat_value', 'energy_value'), ('carbohydrates_value', 'energy_value'), ('salt_value', 'nutriscore_value'),
         ('energy_value', 'nutriscore_value'), ('saturated-fat_value', 'nutriscore_value'),
         ('sugars_value', 'nutriscore_value')])
    model.fit(new_food_df, estimator=BayesianEstimator, prior_type="BDeu")
    model_infer = VariableElimination(model)
    q = model_infer.query(variables=['nutriscore_value'],
                          evidence={'proteins_value': predict_data.loc[0, 'proteins_value'],
                                    'fat_value': predict_data.loc[0, 'fat_value'],
                                    'carbohydrates_value': predict_data.loc[0, 'carbohydrates_value'],
                                    'energy_value': predict_data.loc[0, 'energy_value']})
    val = q.values
    max = np.argmax(val, axis=0)
    if max == 0:
        return 'Alimento prevalentemente salutare'
    else:
        return 'Alimento prevalentemente nocivo'


def bayesianTest(food_df, folds):
    """
    test di una rete bayesiana tramite cross-validation
    :param food_df: dataframe in input
    :param folds: numero di folds
    :return: accuracy media
    """
    new_food_df = bayesian_preprocessing(food_df)
    X_food = new_food_df.drop('nutri_value', axis=1)
    y_food = new_food_df['nutri_value']
    return kFold_cross_validation_bayesian(X_food, y_food, folds)


def values_to_range(new_food_df, f_old, f_val, i, cont, step):
    """
    converte in range da 0 a 4 i valori di un dataframe
    :param new_food_df: dataframe
    :param f_old: nome precedente delle features
    :param f_val: nome aggiornato delle features
    :param i: percentuale
    :param cont: contatore
    :param step: passo
    :return: dataframe trasformato
    """
    if i == 0:
        new_food_df.loc[new_food_df[f_old] < step / 10, f_val] = cont
    elif i == (10 - step):
        new_food_df.loc[new_food_df[f_old] >= i / 10, f_val] = cont
    else:
        new_food_df.loc[
            (new_food_df[f_old] >= i / 10) & (new_food_df[f_old] < (i + step) / 10), f_val] = cont
    return new_food_df
