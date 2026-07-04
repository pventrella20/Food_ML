# Food Machine Learning

Machine Learning application that predicts, from a food's nutritional values
(**kcal**, **carbohydrates**, **proteins**, **fats** per 100 g):

- **Nutri-Score** `[A-E]` (using a **KNN** classifier);
- whether the food is mostly **healthy/unhealthy** (using a **Bayesian Network**);
- **similar foods** with close nutritional values (using **KMeans** clustering).

## Requirements

- Python 3.10+
- Dependencies listed in [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

## Run the web interface (recommended)

The modern interface is built with **Streamlit**:

```bash
streamlit run app.py
```

This opens a browser app with a validated input form, color-coded Nutri-Score
badge, health verdict, and interactive table of similar foods.

> The old Tkinter GUI is still available for compatibility via
> `python -m program.main` (run from project root).

## Architecture

Code is organized into modules in the `program/` directory:

| Module | Role |
| --- | --- |
| `config.py` | **Centralized configuration**: paths, feature lists, hyperparameters, constants. |
| `predictor.py` | **Service layer**: `FoodPredictor` trains models **once** and orchestrates predictions. |
| `nutriscore_classification.py` | KNN Nutri-Score classifier (`NutriScoreKNN`, *train-once* pattern). |
| `bayesian_network.py` | Bayesian network for healthiness (`BayesianHealthModel`, *train-once*). |
| `clustering.py` | KMeans clustering for similar foods (`FoodClustering`, *train-once*). |
| `hyperparameter_optimization.py` | Hyperparameter optimization (GridSearch) for KNN and Random Forest. |
| `cross_validation.py` | K-fold cross-validation for models. |
| `data_manager.py` | Data loading and exploratory analysis utilities. |
| `feature_selection.py` | Feature selection (RFE). |
| `test.py` | CLI testing for supervised models with cross-validation. |
| `main.py` | Legacy Tkinter GUI. |

The dataset is in `data/food_dataset_final.csv`; `docs/` contains Nutri-Score
reference material and analysis images.

## Main improvements over the previous version

- **UX**: modern Streamlit web interface replacing old Tkinter-first workflow.
- **Performance**: models are trained **once** (instead of retraining per prediction) and cached.
- **Architecture**: centralized configuration (`config.py`), service layer (`predictor.py`),
  and reduced duplication of feature lists and hardcoded paths.
- **Compatibility**: removed/reworked deprecated APIs
  (`pandas.read_csv(error_bad_lines=...)`, `pgmpy.BayesianModel`,
  `RandomForest max_features='auto'`).
