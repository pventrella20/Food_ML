import pandas as pd


features = ['product_name', 'categoria', 'packaging_tags', 'categories_tags', 'labels_tags', 'allergens_tags',
            'additives_n', 'additives_tags', 'nutriscore_score', 'nutriscore_grade', 'nova_group', 'pnns_groups_1',
            'pnns_groups_2', 'main_category', 'energy_100g', 'fat_100g', 'saturated-fat_100g',
            'monounsaturated-fat_100g', 'polyunsaturated-fat_100g',
            'omega-3-fat_100g', 'omega-6-fat_100g',
            'trans-fat_100g', 'carbohydrates_100g', 'sugars_100g',
            'fiber_100g',
            'proteins_100g', 'casein_100g', 'serum-proteins_100g', 'nucleotides_100g',
            'salt_100g', 'sodium_100g',
            'alcohol_100g',
            'vitamin-a_100g', 'beta-carotene_100g', 'vitamin-d_100g', 'vitamin-e_100g', 'vitamin-k_100g',
            'vitamin-c_100g', 'vitamin-b1_100g', 'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g',
            'vitamin-b9_100g', 'folates_100g', 'vitamin-b12_100g', 'biotin_100g', 'pantothenic-acid_100g',
            'silica_100g', 'bicarbonate_100g', 'potassium_100g', 'chloride_100g', 'calcium_100g',
            'phosphorus_100g', 'iron_100g', 'magnesium_100g', 'zinc_100g', 'copper_100g', 'manganese_100g',
            'fluoride_100g', 'selenium_100g', 'chromium_100g', 'molybdenum_100g', 'iodine_100g',
            'caffeine_100g', 'taurine_100g', 'ph_100g']

def parseCSV(path, separ):
    '''
    Legge un file .csv e lo incapsula in un dataframe
    :param path: percorso file .csv
    :return: dataframe con i dati estratti
    '''
    read = pd.read_csv(path, sep=separ, error_bad_lines=False)
    return read


# food_df = parseCSV("allFoodModify.csv", ';')
# food_df2 = parseCSV("allFoods.CSV", ';')
# food_df['product_name'] = food_df['product_name'].str.lower()
# food_df.sort_values("product_name", inplace=True)
# food_df2['product_name'] = food_df['product_name'].str.lower()
# food_df2.sort_values("product_name", inplace=True)
# col_one_list = food_df['product_name'].tolist()
# food_df2 = food_df2[food_df2['product_name'].isin(col_one_list)]
# print(food_df2.isnull().any())
# food_df2 = food_df.fillna(0)
# food_df2 = food_df2[features]
# food_df2 = food_df2.loc[:, (food_df2 != 0).any(axis=0)]
#food_df2.to_csv('food_dataset.csv', index=False)
food_df = parseCSV("food_dataset.csv", ',')
food_df.sort_values("product_name", inplace=True)
food_df.to_csv('food_dataset2.csv', index=False)


