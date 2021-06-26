from program.bayesian_network import bayesianTest
from program.data_manager import readCSV
from program.nutriscore_classification import knn_classifier, rf_classifier

food_l = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'monounsaturated-fat_100g',
          'polyunsaturated-fat_100g', 'omega-3-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
          'proteins_100g', 'salt_100g', 'sodium_100g', 'vitamin-a_100g', 'vitamin-d_100g', 'vitamin-e_100g',
          'vitamin-c_100g', 'vitamin-b1_100g', 'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g',
          'vitamin-b9_100g', 'vitamin-b12_100g', 'pantothenic-acid_100g', 'potassium_100g', 'calcium_100g',
          'phosphorus_100g', 'iron_100g', 'magnesium_100g', 'zinc_100g', 'iodine_100g']
food_b = ['product_name', 'energy_100g', 'fat_100g', 'proteins_100g', 'carbohydrates_100g']

if __name__ == "__main__":
    food_df = readCSV('../data/food_dataset_final.csv', ',')
    print(food_df[food_b].describe())
    print('>>> Training e testing del classificatore KNN...')
    print('> Ottimizzare gli iperparametri? [y/n]')
    answ = input()
    print('> Inserire il numero di fold per la cross-validation:')
    folds = int(input())
    if answ == 'y' or answ == 'Y':
        print('accuracy media: ', knn_classifier(food_df, food_l, True, folds=folds))
    else:
        print('accuracy media: ', knn_classifier(food_df, food_l, folds=folds))

    print('>>> Training e testing del classificatore RF...')
    print('> Ottimizzare gli iperparametri? [y/n]')
    answ = input()
    if answ == 'y' or answ == 'Y':
        print('accuracy media: ', rf_classifier(food_df, food_l, True, folds=folds))
    else:
        print('accuracy media: ', rf_classifier(food_df, food_l, folds=folds))

    print('>>> Training e testing della rete Bayesiana...')
    print('> Inserire il numero di fold per la cross-validation:')
    folds = int(input())
    print('accuracy media: ', bayesianTest(food_df, folds))
