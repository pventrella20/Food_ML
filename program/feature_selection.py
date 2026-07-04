from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from program.config import DATASET_PATH, FEATURES_KNN
from program.data_manager import readCSV
def feature_selection_recursive_elimination(df, n_features, col_list):
    """
    Select the most relevant features in a DataFrame.
    :param df: input DataFrame
    :param n_features: number of features to select
    :param col_list: feature names
    """
    array = df[col_list].values
    X = array[:, 1:29]
    y = array[:, 1].astype("int")
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector = selector.fit(X, y)
    print("Num Features: %d" % selector.n_features_)
    features_to_select = [col_list[i] for i, keep in enumerate(selector.support_) if keep]
    print("Selected Features: %s" % features_to_select)
    print("Feature Ranking: %s" % selector.ranking_)
if __name__ == "__main__":
    food_df = readCSV(DATASET_PATH, ",")
    feature_selection_recursive_elimination(food_df, 20, FEATURES_KNN)
