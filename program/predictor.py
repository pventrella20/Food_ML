"""
Service layer that orchestrates the project's three ML models.

Exposes a single ``FoodPredictor`` class that trains once:
  * KNN classifier for Nutri-Score;
  * Bayesian network for healthiness;
  * KMeans clustering for similar foods.

This allows the UI to run multiple predictions without retraining models
on every request.
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
    """Typed container for prediction output."""
    nutriscore: str
    health: str
    similar_foods: pd.DataFrame

    @property
    def is_healthy(self) -> bool:
        return "healthy" in self.health.lower() and "unhealthy" not in self.health.lower()


class FoodPredictor:
    """Model orchestrator, trained once at construction time."""

    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df if df is not None else readCSV()
        self.knn = NutriScoreKNN(self.df, FEATURES_KNN, KNN_HYPERS)
        self.bayes = BayesianHealthModel(self.df)
        self.cluster = FoodClustering(self.df, FEATURES_CLUSTER, N_CLUSTERS)

    @staticmethod
    def build_values(energy_kcal: float, carbs: float, proteins: float, fats: float) -> dict:
        """Build the feature dictionary converting kcal to kJ."""
        return {
            "energy_100g": energy_kcal * KCAL_TO_KJ,
            "carbohydrates_100g": carbs,
            "proteins_100g": proteins,
            "fat_100g": fats,
        }

    def predict(self, values: dict) -> PredictionResult:
        """Run all three predictions and return a ``PredictionResult``."""
        return PredictionResult(
            nutriscore=self.knn.predict(values),
            health=self.bayes.predict(values),
            similar_foods=self.cluster.similar_foods(values),
        )

