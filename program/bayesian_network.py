import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from sklearn.preprocessing import MinMaxScaler

from program.data_manager import readCSV

features = ['energy', 'fat', 'saturated-fat', 'carbohydrates', 'sugars', 'proteins', 'salt', 'nutriscore']
features_ext = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g',
                'salt_100g', 'nutriscore_score']


def bayesianNetwork(food_df):
    new_food_df = food_df[features_ext]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(new_food_df)
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

    model = BayesianModel(
        [('fat_value', 'saturated-fat_value'), ('carbohydrates_value', 'sugars_value'), ('proteins_value', 'salt_value'),
         ('fat_value', 'energy_value'), ('carbohydrates_value', 'energy_value'), ('salt_value', 'nutri_value'),
         ('energy_value', 'nutri_value'), ('saturated-fat_value', 'nutri_value'), ('sugars_value', 'nutri_value')])
    model.fit(new_food_df, estimator=BayesianEstimator, prior_type="BDeu")
    model_infer = VariableElimination(model)

    for x in range(0,5):
        for y in range(0, 5):
            for z in range(0, 5):
                q = model_infer.query(variables=['nutri_value'],
                                      evidence={'proteins_value': x, 'fat_value': y, 'carbohydrates_value': z})
                print(q)
                print('prot: ', x, 'fat: ', y, 'carbs: ', z)
    print("* 0=Alimento Salutare")
    print("* 1=Alimento NON Salutare")


if __name__ == "__main__":
    food_df = readCSV('../data/food_dataset_final.csv', ',')
    bayesianNetwork(food_df)
