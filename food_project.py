import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from numpy import mean, std
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def cmatrix_display(accuracy, confusion_matrix, names, figsize=(10, 7)):
    """
    visualizza a video la matrice di confusione
    :param names: lista delle categorie di nutriscore (a, b, c, d, e)
    :param accuracy: accuracy della predizione
    :param confusion_matrix: matrice di confusione
    :param figsize: grandezza dell'immagine
    :return:
    """
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in names],
                         columns=[i for i in names])
    plt.figure(figsize=figsize)
    plt.title('Confusion Matrix\n Accuracy: {0:.3f}'.format(accuracy))
    sn.heatmap(df_cm, cmap='rocket_r', annot=True, fmt='g')
    plt.ylabel('True letter')
    plt.xlabel('Predicted letter')
    plt.subplots_adjust(bottom=0.155, right=0.924)
    plt.show()

def feature_selection_recursive_elimination(df, n_features):
    array = df.values
    X = array[:, 1:29]
    y = array[:, 1]
    y = y.astype('int')
    # feature extraction
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector = selector.fit(X, y)
    print("Num Features: %d" % selector.n_features_)
    count = 0
    features_to_select = []
    for value in selector.support_:
        if value:
            features_to_select.append(food_l[count])
        count += 1
    print("Selected Features: %s" % features_to_select)
    print("Feature Ranking: %s" % selector.ranking_)

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


def kFold_cross_validation(X, y, model_type, splits=20):
    kf = KFold(n_splits=splits, shuffle=True)
    results = []
    print('Cross validation via K-Fold [{} splits]'.format(splits))
    if model_type == 'classifier':
        for i in range(1, 101):
            knn_2 = KNeighborsClassifier(n_neighbors=i)
            scores = cross_val_score(knn_2, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
            results.append(mean(scores))
            print('Accuracy: %.3f (%.3f), nn = %d' % (mean(scores), std(scores), i))
    if model_type == 'regressor':
        for i in range(1, 101):
            knn_2 = KNeighborsRegressor(n_neighbors=i)
            scores = cross_val_score(knn_2, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
            results.append(mean(scores))
            print('Accuracy: %.3f (%.3f), nn = %d' % (mean(scores), std(scores), i))
    print('best NN number is {} ({:.2f}%)'.format(results.index(max(results)), max(results)))


def knnClassifier(df):
    X_food = df[food_l]
    y_food = df['nutriscore_grade']
    # hyperparameter optimization
    hyper_opt(X_food, y_food)
    # cross-validation
    kFold_cross_validation(X_food, y_food, 'classifier', 10)
    # preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X_food, y_food, test_size=0.2, random_state=200)
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

def knnRegressor(df):
    X_food = df[food_l]
    y_food = df['nutriscore_score']
    # cross-validation
    kFold_cross_validation(X_food, y_food, 'regressor', 10)
    X_train, X_test, y_train, y_test = train_test_split(X_food, y_food, test_size=0.1, random_state=200)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # features normalization (train set)
    X_test_scaled = scaler.transform(X_test)  # features normalization (test set)
    knn = KNeighborsRegressor(n_neighbors=14)  # creo un knnRegressor
    knn.fit(X_train_scaled, y_train)  # fit del knn
    print('Accuracy of K-NN regressor on test set: {:.2f}'
          .format(knn.score(X_test_scaled, y_test)))
    acc = knn.score(X_test_scaled, y_test)

    scores = knn.predict(X_test_scaled)
    # converto i numeri predetti in lettere
    y_pred = score_converter(scores)
    y_test_2 = score_converter(y_test)
    print(confusion_matrix(y_test_2, y_pred))
    print(classification_report(y_test_2, y_pred))
    names = ["a", "b", "c", "d", "e"]
    cmatrix_display(acc, confusion_matrix(y_test_2, y_pred), names)

def hyper_opt(X, y):
    leaf_size = list(range(1, 50))
    n_neighbors = list(range(1, 30))
    p = [1, 2]

    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    knn_2 = KNeighborsClassifier()
    clf = GridSearchCV(knn_2, hyperparameters, cv=10)

    # Fit the model
    best_model = clf.fit(X, y)
    # Print The value of best Hyperparameters
    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

def score_converter(y):
    scores = []
    for score in y:
        if score <= -1:
            scores.append('a')
        elif score <= 2:
            scores.append('b')
        elif score <= 10:
            scores.append('c')
        elif score <= 18:
            scores.append('d')
        else:
            scores.append('e')
    return scores

def data_analisys(df):
    f, axes = plt.subplots(1, 2, figsize=(20, 4))
    sn.histplot(data=df['energy_100g'], ax=axes[0])
    sn.boxplot(data=df['energy_100g'], ax=axes[1])
    plt.show()

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
data_analisys(food_df)
#feature_selection_recursive_elimination(food_df[food_l], 20)
knnClassifier(food_df)
#knnRegressor(food_df)

