from program.bayesian_network import bayesianTest
from program.config import DATASET_PATH, FEATURES_CLUSTER, FEATURES_KNN
from program.data_manager import readCSV
from program.nutriscore_classification import knn_classifier, rf_classifier

food_l = FEATURES_KNN
food_b = FEATURES_CLUSTER

if __name__ == "__main__":
    food_df = readCSV(DATASET_PATH, ',')
    print(food_df[food_b].describe())
    print('>>> Training e testing del classificatore KNN...')
    print('> Ottimizzare gli iperparametri? [y/n]')
    answ = input()
    print('> Inserire il numero di fold per la cross-validation:')
    folds = int(input())
    if answ == 'y' or answ == 'Y':
        print('accuracy media: ', knn_classifier(food_df, food_l, hyp_opt=True, folds=folds))
    else:
        print('accuracy media: ', knn_classifier(food_df, food_l, hyp_opt=False, folds=folds))

    print('>>> Training e testing del classificatore RF...')
    print('> Ottimizzare gli iperparametri? [y/n]')
    answ = input()
    if answ == 'y' or answ == 'Y':
        print('accuracy media: ', rf_classifier(food_df, food_l, hyp_opt=True, folds=folds))
    else:
        print('accuracy media: ', rf_classifier(food_df, food_l, hyp_opt=False, folds=folds))

    print('>>> Training e testing della rete Bayesiana...')
    print('> Inserire il numero di fold per la cross-validation:')
    folds = int(input())
    print('accuracy media: ', bayesianTest(food_df, folds))
