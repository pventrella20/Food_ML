import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def changeCommas(df):
    '''
    Elimina grammi, milligrammi e microgrammi dal dataset, convertendo milligrammi e microgrammi in grammi
    :param df: dataframe su cui effettuare la conversione
    :return: nuovo dataframe
    '''
    for col in df:
        if col in food_l:
            df[col] = df[col].astype(str)
            df[col] = df.apply(lambda x: x[col].replace(",", '.'), axis=1)
            df[col] = df[col].astype(float)
    return df


def knnClassifier(df):
    food_features = ["carbohydrates_100g", "proteins_100g", "fat_100g", "fiber_100g", "energy_100g"]
    X_food = df[food_features]
    y_food = df['nutriscore_grade']
    # preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X_food, y_food, test_size=0.15, random_state=0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # features normalization (train set)
    X_test_scaled = scaler.transform(X_test)  # features normalization (test set)
    for i in range(1, 100):
        knn = KNeighborsClassifier(n_neighbors=i)  # creo un knnClassifier
        knn.fit(X_train_scaled, y_train)  # fit del knn
        #y_pred = knn.predict(X_test_scaled)
        print('accuracy for {} neighbors is {:.2f}'.format(i, knn.score(X_test_scaled, y_test)))
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # # Checking performance on the training set
    # print('Accuracy of K-NN classifier on training set: {:.2f}'
    #       .format(knn.score(X_train_scaled, y_train)))
    # # Checking performance on the test set
    # print('Accuracy of K-NN classifier on test set: {:.2f}'
    #       .format(knn.score(X_test_scaled, y_test)))


def parseCSV(path, separ):
    '''
    Legge un file .csv e lo incapsula in un dataframe
    :param path: percorso file .csv
    :return: dataframe con i dati estratti
    '''
    read = pd.read_csv(path, sep=separ, error_bad_lines=False)
    return read


food_l = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'monounsaturated-fat_100g',
          'polyunsaturated-fat_100g', 'omega-3-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
          'proteins_100g', 'salt_100g', 'sodium_100g', 'vitamin-a_100g', 'vitamin-d_100g', 'vitamin-e_100g',
          'vitamin-c_100g', 'vitamin-b1_100g', 'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g',
          'vitamin-b9_100g', 'vitamin-b12_100g', 'pantothenic-acid_100g', 'potassium_100g', 'calcium_100g',
          'phosphorus_100g', 'iron_100g', 'magnesium_100g', 'zinc_100g', 'iodine_100g']
food_df = parseCSV("food_dataset2.csv", ',')
food_df = changeCommas(food_df)
knnClassifier(food_df)
