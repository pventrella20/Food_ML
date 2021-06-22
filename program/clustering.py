import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

N_CLUSTER = 150


def kMeansCategorization(data, col_list):
    output = ""
    # scaler = StandardScaler()
    # scaler.fit(df[col_list])
    # df[col_list] = scaler.transform(df[col_list])
    clist = []
    for elem in col_list:
        clist.append(elem)
    clist.remove('product_name')
    kMeans_model = KMeans(n_clusters=N_CLUSTER, random_state=0)
    data['cluster'] = kMeans_model.fit_predict(data[clist])
    records = data[data['cluster'] == data['cluster'].iloc[-1]]
    for index, row in records.iterrows():
        if index != -1:
            output += row['product_name'] + '\n('
            output += 'kCal ' + str(round(row['energy_100g'] / 4.184, 2)) + ', '
            output += 'prot ' + str(round(row['proteins_100g'], 2)) + ', '
            output += 'fats ' + str(round(row['fat_100g'], 2)) + ', '
            output += 'carb ' + str(round(row['carbohydrates_100g'], 2)) + ')'
            output += '\n-------------\n'
    return output


def kMeansCluster(df, col_list, values):
    new_col = []
    for col in col_list:
        if col in values:
            new_col.append(values[col])
        else:
            new_col.append(np.nan)
    X = df[col_list]
    X.loc[-1] = new_col
    output = kMeansCategorization(X, col_list)
    return output
