from feature_selection import feature_selection_recursive_elimination
from program.data_manager import readCSV, data_analisys
from program.knn_nutriscore import knn_classifier, knn_regressor


food_l = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'monounsaturated-fat_100g',
          'polyunsaturated-fat_100g', 'omega-3-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
          'proteins_100g', 'salt_100g', 'sodium_100g', 'vitamin-a_100g', 'vitamin-d_100g', 'vitamin-e_100g',
          'vitamin-c_100g', 'vitamin-b1_100g', 'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g',
          'vitamin-b9_100g', 'vitamin-b12_100g', 'pantothenic-acid_100g', 'potassium_100g', 'calcium_100g',
          'phosphorus_100g', 'iron_100g', 'magnesium_100g', 'zinc_100g', 'iodine_100g']

food_df = readCSV('../data/food_dataset_final.csv', ',')

data_analisys(food_df, 'sodium_100g')

#feature_selection_recursive_elimination(food_df, 20, food_l)

#knn_classifier(food_df, food_l, True)

knn_regressor(food_df, food_l, False)
