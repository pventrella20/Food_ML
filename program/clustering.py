import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from program.config import KCAL_TO_KJ, N_CLUSTERS

N_CLUSTER = N_CLUSTERS  # backward compatibility


class FoodClustering:
    """
    KMeans clustering trained ONCE.

    The model is fitted in the constructor; for each input we predict the
    assigned cluster and return foods from that cluster
    (previously KMeans was retrained for each prediction).
    """

    def __init__(self, df, col_list, n_clusters: int = N_CLUSTERS):
        self.col_list = list(col_list)
        self.numeric_cols = [c for c in self.col_list if c != 'product_name']
        self.model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        self.data = df[self.col_list].copy()
        self.data['cluster'] = self.model.fit_predict(self.data[self.numeric_cols])

    def similar_foods(self, values: dict) -> pd.DataFrame:
        """Return a DataFrame with foods in the predicted cluster."""
        row = [[values.get(c, np.nan) for c in self.numeric_cols]]
        cluster = int(self.model.predict(row)[0])
        members = self.data[self.data['cluster'] == cluster].copy()
        # Energy in kcal for more intuitive display.
        if 'energy_100g' in members.columns:
            members['energy_kcal'] = (members['energy_100g'] / KCAL_TO_KJ).round(2)
        return members

    def similar_foods_text(self, values: dict) -> str:
        """Text output (backward-compatible with legacy Tkinter GUI)."""
        output = ""
        for _, row in self.similar_foods(values).iterrows():
            output += row['product_name'] + '\n('
            output += 'kCal ' + str(round(row['energy_100g'] / KCAL_TO_KJ, 2)) + ', '
            output += 'protein ' + str(round(row['proteins_100g'], 2)) + ', '
            output += 'fats ' + str(round(row['fat_100g'], 2)) + ', '
            output += 'carb ' + str(round(row['carbohydrates_100g'], 2)) + ')'
            output += '\n-------------\n'
        return output


def kMeansCluster(df, col_list, values):
    """
    Backward-compatible wrapper: build ``FoodClustering`` and return text.
    Prefer direct class usage to avoid retraining.
    :return: string with foods in the predicted cluster
    """
    return FoodClustering(df, col_list).similar_foods_text(values)
