import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from numpy import mean, std
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def cmatrix_display(accuracy, confusion_matrix, names, figsize=(10, 7)):
    """
    visualizza a video la matrice di confusione
    :param spk_names: lista dei parlatori per cui il modello è addestrato
    :param accuracy: accuracy della predizione
    :param confusion_matrix_spk: matrice di confusione
    :param figsize: grandezza dell'immagine
    :return:
    """
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in names],
                         columns=[i for i in names])
    plt.figure(figsize=figsize)
    plt.title('Confusion Matrix\n Accuracy: {0:.3f}'.format(accuracy))
    sn.heatmap(df_cm, cmap='rocket_r', annot=True, fmt='g')
    plt.ylabel('True speaker')
    plt.xlabel('Predicted speaker')
    plt.subplots_adjust(bottom=0.155, right=0.924)
    plt.show()


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


def kFold_cross_validation(X, y, splits=20):
    kf = KFold(n_splits=splits, random_state=0, shuffle=True)
    results = []
    print('Cross validation via K-Fold [{} splits]'.format(splits))
    for i in range(1, 101):
        knn_2 = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(knn_2, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
        results.append(mean(scores))
        print('Accuracy: %.3f (%.3f), nn = %d' % (mean(scores), std(scores), i))
    print('best NN number is {} ({:.2f}%)'.format(results.index(max(results)), max(results)))


def knnClassifier(df):
    food_features = ["energy_100g", "carbohydrates_100g", "proteins_100g", "fat_100g", "fiber_100g", "salt_100g",
                     "sugars_100g", "sodium_100g"]
    X_food = df[food_features]
    y_food = df['nutriscore_grade']
    # cross-validation
    kFold_cross_validation(X_food, y_food)
    # preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X_food, y_food, test_size=0.5, random_state=200)
    # normalizing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # features normalization (train set)
    X_test_scaled = scaler.transform(X_test)  # features normalization (test set)
    knn = KNeighborsClassifier(n_neighbors=14)  # creo un knnClassifier
    knn.fit(X_train_scaled, y_train)  # fit del knn
    y_pred = knn.predict(X_test_scaled)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # Checking performance on the training set
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(X_train_scaled, y_train)))
    # Checking performance on the test set
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(knn.score(X_test_scaled, y_test)))

    # print confusion matrix
    names = ["a", "b", "c", "d", "e"]
    cmatrix_display(knn.score(X_test_scaled, y_test), confusion_matrix(y_test, y_pred), names)


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

