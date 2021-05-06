import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


def parseCSV(path):
    '''
    Legge un file .csv e lo incapsula in un dataframe
    :param path: percorso file .csv
    :return: dataframe con i dati estratti
    '''
    column_names = [
        "id",
        "name",
        "serving_size",
        "calories",
        "total_fat",
        "saturated_fat",
        "cholesterol",
        "sodium",
        "choline",
        "folate",
        "folic_acid",
        "niacin",
        "pantothenic_acid",
        "riboflavin",
        "thiamin",
        "vitamin_a",
        "vitamin_a_rae",
        "carotene_alpha",
        "carotene_beta",
        "cryptoxanthin_beta",
        "lutein_zeaxanthin",
        "lucopene",
        "vitamin_b12",
        "vitamin_b6",
        "vitamin_c",
        "vitamin_d",
        "vitamin_e",
        "tocopherol_alpha",
        "vitamin_k",
        "calcium",
        "copper",
        "irom",
        "magnesium",
        "manganese",
        "phosphorous",
        "potassium",
        "selenium",
        "zink",
        "protein",
        "alanine",
        "arginine",
        "aspartic_acid",
        "cystine",
        "glutamic_acid",
        "glycine",
        "histidine",
        "hydroxyproline",
        "isoleucine",
        "leucine",
        "lysine",
        "methionine",
        "phenylalanine",
        "proline",
        "serine",
        "threonine",
        "tryptophan",
        "tyrosine",
        "valine",
        "carbohydrate",
        "fiber",
        "sugars",
        "fructose",
        "galactose",
        "glucose",
        "lactose",
        "maltose",
        "sucrose",
        "fat",
        "saturated_fatty_acids",
        "monounsaturated_fatty_acids",
        "polyunsaturated_fatty_acids",
        "fatty_acids_total_trans",
        "alcohol",
        "ash",
        "caffeine",
        "theobromine",
        "water"
    ]
    food = pd.read_csv(path, header=None, sep=',', error_bad_lines=False, names=column_names)
    return food

def eliminateGrams(df):
    '''
    Elimina grammi, milligrammi e microgrammi dal dataset, convertendo milligrammi e microgrammi in grammi
    :param df: dataframe su cui effettuare la conversione
    :return: nuovo dataframe
    '''
    for name in df.columns:
        if name not in accepted:
            df = df.drop(name, axis=1)
    for col in convert:
        for index in df.index:
            if 'g' in df.loc[index, col]:
                df.loc[index, col] = df.loc[index, col].replace('g', '')
            if ' g' in df.loc[index, col]:
                df.loc[index, col] = df.loc[index, col].replace(' g', '')
            if 'mg' in df.loc[index, col]:
                df.loc[index, col] = df.loc[index, col].replace('mg', '')
                df.loc[index, col] = str(float(df.loc[index, col]) / 1000)
            if ' mg' in df.loc[index, col]:
                df.loc[index, col] = df.loc[index, col].replace(' mg', '')
                df.loc[index, col] = str(float(df.loc[index, col]) / 1000)
            if 'mcg' in df.loc[index, col]:
                df.loc[index, col] = df.loc[index, col].replace('mcg', '')
                df.loc[index, col] = str(float(df.loc[index, col]) / 1000000)
            if ' mcg' in df.loc[index, col]:
                df.loc[index, col] = df.loc[index, col].replace(' mcg', '')
                df.loc[index, col] = str(float(df.loc[index, col]) / 1000000)
    return df

def splittingAndEval(df):
    food_features = ["carbohydrate", "protein", "total_fat"]
    X_food = df[food_features]
    y_food = df['calories']
    print(df.head(10))
    # preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X_food, y_food, random_state=0)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # features normalization (train set)
    X_test_scaled = scaler.transform(X_test)        # features normalization (test set)
    knn = KNeighborsRegressor(n_neighbors=5)        # creo un knnRegressor
    knn.fit(X_train_scaled, y_train)                # fit del knn

    # Checking performance on the training set
    print('Accuracy of K-NN regressor on training set: {:.2f}'
          .format(knn.score(X_train_scaled, y_train)))
    # Checking performance on the test set
    print('Accuracy of K-NN regressor on test set: {:.2f}'
          .format(knn.score(X_test_scaled, y_test)))
    example_food = [[4.97, 1.92, 0.3]]  # inserire qui i valori di carboidrati, proteine e grassi per la predizione
    example_food_scaled = scaler.transform(example_food)
    # Making an prediction based on x values
    print('Predicted food calories for ', example_food, ' is ',
          knn.predict(example_food_scaled)[0] - 1)


accepted = ["name", "calories", "carbohydrate", "protein", "total_fat"]
convert = ["carbohydrate", "protein", "total_fat"]
f_r = parseCSV("nutrition.csv")
f_r = eliminateGrams(f_r)
splittingAndEval(f_r)
