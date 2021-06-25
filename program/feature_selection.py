from sklearn.feature_selection import RFE
from sklearn.svm import SVR


def feature_selection_recursive_elimination(df, n_features, col_list):
    """
    seleziona le features più rilevanti in un dataframe
    :param df: dataframe in input
    :param n_features: numero totale di features
    :param col_list: nomi delle features
    :return:
    """
    array = df[col_list].values
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
            features_to_select.append(col_list[count])
        count += 1
    print("Selected Features: %s" % features_to_select)
    print("Feature Ranking: %s" % selector.ranking_)
