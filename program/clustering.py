import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from program.config import KCAL_TO_KJ, N_CLUSTERS

N_CLUSTER = N_CLUSTERS  # retro-compatibilita'


class FoodClustering:
    """
    Clustering KMeans addestrato UNA sola volta.

    Il modello viene fittato nel costruttore; per ogni input si predice il
    cluster di appartenenza e si restituiscono gli alimenti dello stesso cluster
    (in precedenza il KMeans veniva ri-addestrato ad ogni predizione).
    """

    def __init__(self, df, col_list, n_clusters: int = N_CLUSTERS):
        self.col_list = list(col_list)
        self.numeric_cols = [c for c in self.col_list if c != 'product_name']
        self.model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        self.data = df[self.col_list].copy()
        self.data['cluster'] = self.model.fit_predict(self.data[self.numeric_cols])

    def similar_foods(self, values: dict) -> pd.DataFrame:
        """Restituisce un DataFrame con gli alimenti del cluster predetto."""
        row = [[values.get(c, np.nan) for c in self.numeric_cols]]
        cluster = int(self.model.predict(row)[0])
        members = self.data[self.data['cluster'] == cluster].copy()
        # energia in kcal per una lettura piu' naturale
        if 'energy_100g' in members.columns:
            members['energy_kcal'] = (members['energy_100g'] / KCAL_TO_KJ).round(2)
        return members

    def similar_foods_text(self, values: dict) -> str:
        """Versione testuale (retro-compatibile con la vecchia GUI Tkinter)."""
        output = ""
        for _, row in self.similar_foods(values).iterrows():
            output += row['product_name'] + '\n('
            output += 'kCal ' + str(round(row['energy_100g'] / KCAL_TO_KJ, 2)) + ', '
            output += 'prot ' + str(round(row['proteins_100g'], 2)) + ', '
            output += 'fats ' + str(round(row['fat_100g'], 2)) + ', '
            output += 'carb ' + str(round(row['carbohydrates_100g'], 2)) + ')'
            output += '\n-------------\n'
        return output


def kMeansCluster(df, col_list, values):
    """
    Wrapper retro-compatibile: costruisce un ``FoodClustering`` e restituisce il testo.
    Preferire l'uso diretto della classe per evitare ri-addestramenti.
    :return: stringa contenente gli alimenti del cluster predetto
    """
    return FoodClustering(df, col_list).similar_foods_text(values)
