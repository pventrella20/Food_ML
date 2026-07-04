"""
Service layer che orchestra i tre modelli di ML del progetto.

Espone un'unica classe ``FoodPredictor`` che addestra una sola volta:
  * il classificatore KNN del Nutriscore;
  * la rete bayesiana per la salubrita';
  * il clustering KMeans per gli alimenti simili.

In questo modo l'interfaccia utente puo' effettuare piu' predizioni senza
ri-addestrare i modelli ad ogni richiesta.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from program.bayesian_network import BayesianHealthModel
from program.clustering import FoodClustering
from program.config import (
    FEATURES_CLUSTER, FEATURES_KNN, KCAL_TO_KJ, KNN_HYPERS, N_CLUSTERS,
)
from program.data_manager import readCSV
from program.nutriscore_classification import NutriScoreKNN


@dataclass
class PredictionResult:
    """Contenitore tipizzato del risultato di una predizione."""
    nutriscore: str
    health: str
    similar_foods: pd.DataFrame

    @property
    def is_healthy(self) -> bool:
        return "salutare" in self.health.lower()


class FoodPredictor:
    """Orchestratore dei modelli, addestrati una sola volta alla costruzione."""

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df if df is not None else readCSV()
        self.knn = NutriScoreKNN(self.df, FEATURES_KNN, KNN_HYPERS)
        self.bayes = BayesianHealthModel(self.df)
        self.cluster = FoodClustering(self.df, FEATURES_CLUSTER, N_CLUSTERS)

    @staticmethod
    def build_values(energy_kcal: float, carbs: float, proteins: float, fats: float) -> dict:
        """Costruisce il dizionario di feature convertendo le kcal in kJ."""
        return {
            "energy_100g": energy_kcal * KCAL_TO_KJ,
            "carbohydrates_100g": carbs,
            "proteins_100g": proteins,
            "fat_100g": fats,
        }

    def predict(self, values: dict) -> PredictionResult:
        """Esegue le tre predizioni e restituisce un ``PredictionResult``."""
        return PredictionResult(
            nutriscore=self.knn.predict(values),
            health=self.bayes.predict(values),
            similar_foods=self.cluster.similar_foods(values),
        )

