"""
Configurazione centralizzata del progetto Food_ML.

Raccoglie in un unico punto percorsi, liste di feature, iperparametri e
costanti condivise, eliminando la duplicazione presente nei vari moduli.
"""
from __future__ import annotations

from pathlib import Path

# --- Percorsi (robusti, indipendenti dalla working directory) ---------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
DATASET_PATH: Path = DATA_DIR / "food_dataset_final.csv"
CSV_SEPARATOR: str = ","

# --- Costanti di dominio -----------------------------------------------------
# Il dataset esprime l'energia in kJ, mentre l'utente ragiona in kcal.
KCAL_TO_KJ: float = 4.184

# Target delle predizioni
TARGET_GRADE: str = "nutriscore_grade"
TARGET_SCORE: str = "nutriscore_score"

# --- Liste di feature --------------------------------------------------------
# Feature complete usate dal classificatore KNN (ex "food_l")
FEATURES_KNN: list[str] = [
    "energy_100g", "fat_100g", "saturated-fat_100g", "monounsaturated-fat_100g",
    "polyunsaturated-fat_100g", "omega-3-fat_100g", "carbohydrates_100g", "sugars_100g",
    "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g", "vitamin-a_100g",
    "vitamin-d_100g", "vitamin-e_100g", "vitamin-c_100g", "vitamin-b1_100g",
    "vitamin-b2_100g", "vitamin-pp_100g", "vitamin-b6_100g", "vitamin-b9_100g",
    "vitamin-b12_100g", "pantothenic-acid_100g", "potassium_100g", "calcium_100g",
    "phosphorus_100g", "iron_100g", "magnesium_100g", "zinc_100g", "iodine_100g",
]

# Feature usate per il clustering (ex "food_b"); include il nome prodotto
FEATURES_CLUSTER: list[str] = [
    "product_name", "energy_100g", "fat_100g", "proteins_100g", "carbohydrates_100g",
]

# Feature nutrizionali inserite dall'utente (in unità "utente": kcal e grammi)
USER_INPUT_FEATURES: list[str] = [
    "energy_100g", "carbohydrates_100g", "proteins_100g", "fat_100g",
]

# --- Iperparametri -----------------------------------------------------------
KNN_HYPERS: dict = {"leaf_size": 1, "n_neighbors": 7, "p": 1, "weights": "distance"}
KNN_IMPUTER_NEIGHBORS: int = 7
N_CLUSTERS: int = 150

