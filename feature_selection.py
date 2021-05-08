import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


N_CLUSTER = 6


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
        df[col] = df[col].astype(str)
        df[col] = df[col].map(lambda x: x.rstrip('IU'))
        df[col] = df[col].map(lambda x: x.rstrip('mcg'))
        df[col] = df[col].map(lambda x: x.rstrip('mg'))
        df[col] = df[col].map(lambda x: x.rstrip('g'))
        df[col] = df[col].astype(float)
    return df


def knnRegressor(df):
    food_features = ["carbohydrate", "protein", "total_fat", "fiber"]
    X_food = df[food_features]
    y_food = df['calories']
    print(df.head(3))
    # preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X_food, y_food, random_state=0)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # features normalization (train set)
    X_test_scaled = scaler.transform(X_test)  # features normalization (test set)
    knn = KNeighborsRegressor(n_neighbors=5)  # creo un knnRegressor
    knn.fit(X_train_scaled, y_train)  # fit del knn

    # Checking performance on the training set
    print('Accuracy of K-NN regressor on training set: {:.2f}'
          .format(knn.score(X_train_scaled, y_train)))
    # Checking performance on the test set
    print('Accuracy of K-NN regressor on test set: {:.2f}'
          .format(knn.score(X_test_scaled, y_test)))
    example_food = [[1.4, 1.3, 0.1, 1.2]]  # inserire qui i valori di carboidrati, proteine e grassi per la predizione
    example_food_scaled = scaler.transform(example_food)
    # Making an prediction based on x values
    print('Predicted food calories for ', example_food, ' is ',
          knn.predict(example_food_scaled)[0])


def kMeansCategorization(df):
    food_features=allFoodFeature()
    scaler = MinMaxScaler()
    scaler.fit(df[food_features])
    df[food_features] = scaler.transform(df[food_features])
    kMeans = KMeans(n_clusters=N_CLUSTER, random_state=0)
    prediction = kMeans.fit_predict(df[food_features])

    transformToLabel = {0: 'Frutta e Verdura', 1: 'Frutta secca e oli', 2: 'Carne-  e pesce', 3: 'Farinacei', 4: 'Dolci', 5: 'Formaggi'}
    prediction = [transformToLabel[pred] for pred in prediction]
    df['Cluster_name'] = prediction
    print(df.head(50))

    # hyperparameter evaluation using silhouette score
    k_to_test = range(2, 5, 1)  # [2,3,4, ..., 24]
    silhouette_scores = {}
    for k in k_to_test:
        kmeans_k = KMeans(n_clusters=k)
        kmeans_k.fit(df[food_features])
        labels_k = kmeans_k.labels_
        score_k = metrics.silhouette_score(df[food_features], labels_k)
        silhouette_scores[k] = score_k
        print("Tested kMeans with k = %d\tSS: %5.4f" % (k, score_k))

def PCAFeatureSelection(df):
    food_features = allFoodFeature()
    scaler = MinMaxScaler()
    scaler.fit(df[food_features])
    df[food_features] = scaler.transform(df[food_features])
    pca = PCA(n_components=4)
    principalComponent = pca.fit_transform(df[food_features])
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='blue')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    print(features)
    plt.show()

def randomForestFeatureSelection(df):
    food_features = allFoodFeature()
    X_food = df[food_features]
    y_food = df['calories']
    X_train, X_test, y_train, y_test = train_test_split(X_food, y_food, random_state=0)
    sel = SelectFromModel(RandomForestClassifier(n_estimators=10))
    sel.fit(X_train, y_train)
    print(sel.get_support())
    selected_feat= X_train.columns[(sel.get_support())]
    print(selected_feat)

def feature_selection_univariate(df):
    array = df.values
    X = array[:, 1:75]
    Y = array[:, 1]
    # feature extraction
    test = SelectKBest(score_func=f_classif, k=10)
    fit = test.fit(X, Y)
    # summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    # summarize selected features
    print(features[0:5, :])
    for val in features[1]:
        print(df.columns[(df == val).iloc[1]])

def feature_selection_recursive_elimination(df):
    array = df.values
    X = array[:, 1:75]
    Y = array[:, 1]
    Y = Y.astype('int')
    # feature extraction
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model)
    fit = rfe.fit(X, Y)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)

def allFoodFeature():
    food_features = ["calories",
                     "total_fat",
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
                     "water"]
    return (food_features)

accepted = ["name",
            "calories",
            "total_fat",
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
            "water"]
convert = ["total_fat",
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
           "water"]
f_r = parseCSV("nutrition.csv")
f_r = eliminateGrams(f_r)
# print(f_r.corr())
# knnRegressor(f_r)
#kMeansCategorization(f_r)
# PCAFeatureSelection(f_r)
# randomForestFeatureSelection(f_r)
feature_selection_univariate(f_r)
#feature_selection_recursive_elimination(f_r)
