import pandas as pd

N_CLUSTER = 3


def parseCSV(path):
    '''
    Legge un file .csv e lo incapsula in un dataframe
    :param path: percorso file .csv
    :return: dataframe con i dati estratti
    '''
    food = pd.read_csv(path, sep='\t', error_bad_lines=False)
    return food


def selectFeatures(df):
    to_try = ['product_name', 'categories', 'categories_tags',
              'allergens', 'allergens_tags',
              'additives_n', 'additives_tags',
              'ingredients_from_palm_oil_n', 'ingredients_from_palm_oil', 'ingredients_from_palm_oil_tags',
              'nutriscore_score', 'nutriscore_grade',
              'pnns_groups_1', 'pnns_groups_2',
              'main_category',
              'energy-kj_100g', 'energy-kcal_100g', 'energy_100g', 'energy-from-fat_100g',
              'fat_100g', 'saturated-fat_100g',
              '-butyric-acid_100g', '-caproic-acid_100g', '-caprylic-acid_100g', '-capric-acid_100g', '-lauric-acid_100g',
              '-myristic-acid_100g', '-palmitic-acid_100g', '-stearic-acid_100g', '-arachidic-acid_100g', '-behenic-acid_100g',
              '-lignoceric-acid_100g', '-cerotic-acid_100g', '-montanic-acid_100g', '-melissic-acid_100g',
              'monounsaturated-fat_100g', 'polyunsaturated-fat_100g',
              'omega-3-fat_100g',
              '-alpha-linolenic-acid_100g', '-eicosapentaenoic-acid_100g', '-docosahexaenoic-acid_100g',
              'omega-6-fat_100g',
              '-linoleic-acid_100g', '-arachidonic-acid_100g', '-gamma-linolenic-acid_100g', '-dihomo-gamma-linolenic-acid_100g',
              'omega-9-fat_100g',
              '-oleic-acid_100g', '-elaidic-acid_100g', '-gondoic-acid_100g', '-mead-acid_100g', '-erucic-acid_100g', '-nervonic-acid_100g',
              'trans-fat_100g', 'cholesterol_100g',
              'carbohydrates_100g', 'sugars_100g',
              '-sucrose_100g', '-glucose_100g', '-fructose_100g', '-lactose_100g', '-maltose_100g', '-maltodextrins_100g',
              'starch_100g', 'polyols_100g',
              'fiber_100g',
              '-soluble-fiber_100g', '-insoluble-fiber_100g',
              'proteins_100g',
              'casein_100g', 'serum-proteins_100g',
              'nucleotides_100g',
              'salt_100g', 'sodium_100g',
              'alcohol_100g',
              'vitamin-a_100g', 'beta-carotene_100g', 'vitamin-d_100g', 'vitamin-e_100g', 'vitamin-k_100g', 'vitamin-c_100g', 'vitamin-b1_100g',
              'vitamin-b2_100g', 'vitamin-pp_100g', 'vitamin-b6_100g', 'vitamin-b9_100g', 'folates_100g', 'vitamin-b12_100g',
              'biotin_100g', 'pantothenic-acid_100g', 'silica_100g', 'bicarbonate_100g',
              'potassium_100g', 'chloride_100g', 'calcium_100g', 'phosphorus_100g', 'iron_100g', 'magnesium_100g', 'zinc_100g', 'copper_100g', 'manganese_100g',
              'fluoride_100g', 'selenium_100g', 'chromium_100g', 'molybdenum_100g', 'iodine_100g',
              'caffeine_100g']
    to_drop = ['code', 'url', 'creator', 'created_t', 'last_modified_t', 'abbreviated_product_name', 'generic_name',
               'quantity', 'packaging', 'packaging_tags', 'packaging_text', 'brands', 'brands_tags',
               'origins', 'origins_tags', 'manufacturing_places', 'manufacturing_places_tags', 'labels',
               'labels_tags', 'emb_codes',
               'emb_codes_tags', 'cities', 'cities_tags', 'purchase_places', 'stores', 'countries',
               'ingredients_text', 'traces', 'traces_tags', 'serving_size', 'serving_quantity',
               'ingredients_that_may_be_from_palm_oil_n',
               'ingredients_that_may_be_from_palm_oil', 'ingredients_that_may_be_from_palm_oil_tags',
               'nova_group',
               'states', 'brand_owner', 'ecoscore_score_fr', 'ecoscore_grade_fr',
               'image_url',
               'image_small_url', 'image_front_url', 'image_front_small_url', 'image_ingredients_url',
               'image_ingredients_small_url', 'image_nutrition_url',
               'image_nutrition_small_url',
               'fruits-vegetables-nuts_100g', 'fruits-vegetables-nuts-dried_100g',
               'fruits-vegetables-nuts-estimate_100g', 'collagen-meat-protein-ratio_100g', 'cocoa_100g',
               'chlorophyl_100g', 'carbon-footprint_100g',
               'carbon-footprint-from-meat-or-fish_100g', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g']
    df = df[to_try]
    return df


f_r = parseCSV("openfoodfacts.csv")
f_r = selectFeatures(f_r)
print(f_r.isnull().any())
f_r = f_r.fillna(0)
