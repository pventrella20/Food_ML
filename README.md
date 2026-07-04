# Food Machine Learning

Applicazione di Machine Learning che, a partire dai valori nutrizionali di un
alimento (**kcal**, **carboidrati**, **proteine**, **grassi** su 100 g),
predice:

- il **Nutriscore** `[A-E]` (classificatore **KNN**);
- se l'alimento è prevalentemente **sano/nocivo** (**Rete Bayesiana**);
- gli **alimenti con valori nutrizionali simili** (**clustering KMeans**).

## Requisiti

- Python 3.10+
- Dipendenze in [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

## Avvio dell'interfaccia web (consigliato)

L'interfaccia moderna è realizzata con **Streamlit**:

```bash
streamlit run app.py
```

Si aprirà nel browser un'app con form validato, badge Nutriscore colorato,
verdetto di salubrità e tabella interattiva degli alimenti simili.

> La vecchia GUI Tkinter resta disponibile per compatibilità con
> `python -m program.main` (eseguire dalla root del progetto).

## Architettura

Il codice è organizzato in moduli nella directory `program/`:

| Modulo | Ruolo |
| --- | --- |
| `config.py` | **Configurazione centralizzata**: percorsi, liste di feature, iperparametri, costanti. |
| `predictor.py` | **Service layer**: la classe `FoodPredictor` addestra i modelli **una sola volta** e orchestra le predizioni. |
| `nutriscore_classification.py` | Classificatore KNN del Nutriscore (`NutriScoreKNN`, pattern *train-once*). |
| `bayesian_network.py` | Rete bayesiana per la salubrità (`BayesianHealthModel`, *train-once*). |
| `clustering.py` | Clustering KMeans per gli alimenti simili (`FoodClustering`, *train-once*). |
| `hyperparameter_optimization.py` | Ottimizzazione iperparametri (GridSearch) per KNN e Random Forest. |
| `cross_validation.py` | K-fold cross-validation per i modelli. |
| `data_manager.py` | Caricamento e analisi esplorativa dei dati. |
| `feature_selection.py` | Selezione delle feature (RFE). |
| `test.py` | Testing dei modelli supervisionati con cross-validation. |
| `main.py` | GUI Tkinter *legacy*. |

Il dataset è in `data/food_dataset_final.csv`; in `docs/` informazioni sul
Nutriscore e grafici dell'analisi.

## Principali migliorie rispetto alla versione precedente

- **UX**: nuova interfaccia web Streamlit al posto della GUI Tkinter.
- **Performance**: i modelli vengono addestrati **una sola volta** (prima
  venivano ri-addestrati ad ogni predizione) e messi in cache.
- **Architettura**: configurazione centralizzata (`config.py`), service layer
  (`predictor.py`), eliminazione della duplicazione delle liste di feature e
  dei percorsi hardcoded.
- **Compatibilità**: rimozione di API deprecate/rimosse
  (`pandas.read_csv(error_bad_lines=...)`, `pgmpy.BayesianModel`,
  `RandomForest max_features='auto'`).
