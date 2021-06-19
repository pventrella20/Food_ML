from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

N_CLUSTER = 150


def kMeansCategorization(df, col_list):
    # scaler = StandardScaler()
    # scaler.fit(df[col_list])
    # df[col_list] = scaler.transform(df[col_list])
    kMeans = KMeans(n_clusters=N_CLUSTER, random_state=0)
    df['cluster'] = kMeans.fit_predict(df[col_list])
    for i in range(0, N_CLUSTER):
        records = df[df['cluster'] == i]
        for index, row in records.iterrows():
            print(row['cluster'],
                  'kCal ' + str(round(row['energy_100g'], 2)),
                  'prot ' + str(round(row['proteins_100g'], 2)),
                  'fats ' + str(round(row['fat_100g'], 2)),
                  'carb ' + str(round(row['carbohydrates_100g'], 2)),
                  'fibe ' + str(round(row['fiber_100g'], 2)),
                  'salt ' + str(round(row['salt_100g'], 2)),
                  row['product_name'])
        print('------------------')
