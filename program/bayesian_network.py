import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from sklearn.preprocessing import MinMaxScaler

from program.data_manager import readCSV

features = ['energy', 'fat', 'saturated-fat', 'carbohydrates', 'sugars', 'proteins', 'salt', 'nutriscore']
features_ext = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g',
                'salt_100g', 'nutriscore_score']


def bayesian_preprocessing(food_df, values=None):
    new_food_df = food_df[features_ext]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(new_food_df)
    if values is not None:
        col_list = []
        new_values = []
        for col in features:
            if col == 'energy' or col == 'proteins' or col == 'fat' or col == 'carbohydrates':
                col_list.append(col + '_100g')
                new_values.append(values[col + '_100g'])
            else:
                col_list.append(col + '_100g')
                new_values.append(np.nan)
        df_predict = pd.DataFrame(columns=col_list)
        df_predict.loc[0] = new_values
        scaled_df = scaler.transform(df_predict)
        df_predict = pd.DataFrame(scaled_df, index=df_predict.index, columns=df_predict.columns)
        for col in features:
            cont = 0
            step = 2
            for i in range(0, 10, step):
                if i == 0:
                    df_predict.loc[df_predict[col + '_100g'] < step / 10, col + '_value'] = cont
                elif i == (10 - step):
                    df_predict.loc[df_predict[col + '_100g'] >= i / 10, col + '_value'] = cont
                else:
                    df_predict.loc[
                        (df_predict[col + '_100g'] >= i / 10) & (df_predict[col + '_100g'] < (i + step) / 10), col + '_value'] = cont
                cont += 1
    new_food_df = pd.DataFrame(scaled_features, index=new_food_df.index, columns=new_food_df.columns)
    for feat in features:
        cont = 0
        step = 2
        if 'nutriscore' in feat:
            new_food_df.loc[new_food_df['nutriscore_score'] < 0.5, 'nutri_value'] = 0
            new_food_df.loc[new_food_df['nutriscore_score'] >= 0.5, 'nutri_value'] = 1
        else:
            f_val = feat + '_value'
            f_old = feat + '_100g'
            for i in range(0, 10, step):
                if i == 0:
                    new_food_df.loc[new_food_df[f_old] < step / 10, f_val] = cont
                elif i == (10 - step):
                    new_food_df.loc[new_food_df[f_old] >= i / 10, f_val] = cont
                else:
                    new_food_df.loc[
                        (new_food_df[f_old] >= i / 10) & (new_food_df[f_old] < (i + step) / 10), f_val] = cont
                cont += 1
    new_food_df = new_food_df.drop(features_ext, axis=1)
    return new_food_df, df_predict
    # X_food = new_food_df.drop('nutri_value', axis=1)
    # y_food = new_food_df['nutri_value']
    # kFold_cross_validation_bayesian(X_food, y_food, 26)


def bayesianNetwork(food_df, values):
    new_food_df, predict_data = bayesian_preprocessing(food_df, values)
    model = BayesianModel(
        [('fat_value', 'saturated-fat_value'), ('carbohydrates_value', 'sugars_value'),
         ('proteins_value', 'salt_value'),
         ('fat_value', 'energy_value'), ('carbohydrates_value', 'energy_value'), ('salt_value', 'nutri_value'),
         ('energy_value', 'nutri_value'), ('saturated-fat_value', 'nutri_value'), ('sugars_value', 'nutri_value')])
    model.fit(new_food_df, estimator=BayesianEstimator, prior_type="BDeu")
    model_infer = VariableElimination(model)
    q = model_infer.query(variables=['nutri_value'],
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


if __name__ == "__main__":
    food_df = readCSV('../data/food_dataset_final.csv', ',')
    values = {'energy_value': 500, 'fat_value': 30, 'carbohydrates_value': 20,
              'proteins_value': 6}
    print(bayesianNetwork(food_df, values))
