from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from program.data_manager import readCSV


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

if __name__ == "__main__":
    food_l = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'monounsaturated-fat_100g',
              'polyunsaturated-fat_100g', 'omega-3-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
              'proteins_100g', 'salt_100g', 'vitamin-a_100g', 'vitamin-d_100g', 'vitamin-e_100g',
              'vitamin-c_100g', 'vitamin-b1_100g', 'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g',
              'vitamin-b9_100g', 'vitamin-b12_100g', 'pantothenic-acid_100g', 'potassium_100g', 'calcium_100g',
              'phosphorus_100g', 'iron_100g', 'magnesium_100g', 'zinc_100g', 'iodine_100g']
    food_df = readCSV('../data/food_dataset_final.csv', ',')
    feature_selection_recursive_elimination(food_df, 20, food_l)
