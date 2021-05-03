import pandas as pd


def parseCSV(path):
    column_names = [
        "id",
        "name",
        "serving_size",
        "calories",
        "total_fat",
        "saturated_fat",
        "cholesterol",
        "sodium",
        "choline",
        "folate",
        "folic_acid",
        "niacin",
        "pantothenic_acid",
        "riboflavin",
        "thiamin",
        "vitamin_a",
        "vitamin_a_rae",
        "carotene_alpha",
        "carotene_beta",
        "cryptoxanthin_beta",
        "lutein_zeaxanthin",
        "lucopene",
        "vitamin_b12",
        "vitamin_b6",
        "vitamin_c",
        "vitamin_d",
        "vitamin_e",
        "tocopherol_alpha",
        "vitamin_k",
        "calcium",
        "copper",
        "irom",
        "magnesium",
        "manganese",
        "phosphorous",
        "potassium",
        "selenium",
        "zink",
        "protein",
        "alanine",
        "arginine",
        "aspartic_acid",
        "cystine",
        "glutamic_acid",
        "glycine",
        "histidine",
        "hydroxyproline",
        "isoleucine",
        "leucine",
        "lysine",
        "methionine",
        "phenylalanine",
        "proline",
        "serine",
        "threonine",
        "tryptophan",
        "tyrosine",
        "valine",
        "carbohydrate",
        "fiber",
        "sugars",
        "fructose",
        "galactose",
        "glucose",
        "lactose",
        "maltose",
        "sucrose",
        "fat",
        "saturated_fatty_acids",
        "monounsaturated_fatty_acids",
        "polyunsaturated_fatty_acids",
        "fatty_acids_total_trans",
        "alcohol",
        "ash",
        "caffeine",
        "theobromine",
        "water"
    ]
    food = pd.read_csv(path, header=None, sep=',', error_bad_lines=False, names=column_names)
    return food


accepted = ["name", "calories", "carbohydrate", "protein", "total_fat"]
convert = ["carbohydrate", "protein", "total_fat"]
f_r = parseCSV("nutrition.csv")
for name in f_r.columns:
    if name not in accepted:
        f_r = f_r.drop(name, axis=1)
for col in convert:
    for index in f_r.index:
        if 'g' in f_r.loc[index, col]:
            f_r.loc[index, col] = f_r.loc[index, col].replace('g', '')
        if ' g' in f_r.loc[index, col]:
            f_r.loc[index, col] = f_r.loc[index, col].replace(' g', '')
        if 'mg' in f_r.loc[index, col]:
            f_r.loc[index, col] = f_r.loc[index, col].replace('mg', '')
            f_r.loc[index, col] = float(f_r.loc[index, col]) * 1000
        if ' mg' in f_r.loc[index, col]:
            f_r.loc[index, col] = f_r.loc[index, col].replace(' mg', '')
            f_r.loc[index, col] = float(f_r.loc[index, col]) * 1000
        if 'mcg' in f_r.loc[index, col]:
            f_r.loc[index, col] = f_r.loc[index, col].replace('mcg', '')
            f_r.loc[index, col] = float(f_r.loc[index, col]) * 1000000
        if ' mcg' in f_r.loc[index, col]:
            f_r.loc[index, col] = f_r.loc[index, col].replace(' mcg', '')
            f_r.loc[index, col] = float(f_r.loc[index, col]) * 1000000

# f_r["calories"].hist(bins=15)
# plt.show()
correlation_matrix = f_r.corr()
correlation_matrix["calories"]
