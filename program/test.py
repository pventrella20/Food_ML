from program.bayesian_network import bayesianTest
from program.config import DATASET_PATH, FEATURES_CLUSTER, FEATURES_KNN
from program.data_manager import readCSV
from program.nutriscore_classification import knn_classifier, rf_classifier

food_l = FEATURES_KNN
food_b = FEATURES_CLUSTER

if __name__ == "__main__":
    food_df = readCSV(DATASET_PATH, ',')
    print(food_df[food_b].describe())
    print('>>> Training and testing KNN classifier...')
    print('> Optimize hyperparameters? [y/n]')
    answ = input()
    print('> Enter number of folds for cross-validation:')
    folds = int(input())
    if answ == 'y' or answ == 'Y':
        print('mean accuracy: ', knn_classifier(food_df, food_l, hyp_opt=True, folds=folds))
    else:
        print('mean accuracy: ', knn_classifier(food_df, food_l, hyp_opt=False, folds=folds))

    print('>>> Training and testing RF classifier...')
    print('> Optimize hyperparameters? [y/n]')
    answ = input()
    if answ == 'y' or answ == 'Y':
        print('mean accuracy: ', rf_classifier(food_df, food_l, hyp_opt=True, folds=folds))
    else:
        print('mean accuracy: ', rf_classifier(food_df, food_l, hyp_opt=False, folds=folds))

    print('>>> Training and testing Bayesian network...')
    print('> Enter number of folds for cross-validation:')
    folds = int(input())
    print('mean accuracy: ', bayesianTest(food_df, folds))
