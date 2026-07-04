import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from program.config import CSV_SEPARATOR, DATASET_PATH
def readCSV(path=DATASET_PATH, separ=CSV_SEPARATOR):
    """
    legge un file .csv e lo incapsula in un dataframe pandas
    :param path: percorso file .csv da leggere (default: dataset del progetto)
    :param separ: separatore (carattere)
    :return: dataframe contenente i dati estratti
    """
    # 'on_bad_lines' sostituisce 'error_bad_lines' rimosso in pandas >= 2.0
    return pd.read_csv(path, sep=separ, on_bad_lines="skip")
def data_analisys(df, feature):
    """analisi dei dati (istogramma + boxplot) per una feature"""
    f, axes = plt.subplots(1, 2, figsize=(20, 4))
    sns.histplot(data=df[feature], ax=axes[0])
    sns.boxplot(data=df[feature], ax=axes[1])
    plt.show()
def column_analisys(df, col):
    """stampa il numero di valori per la feature"""
    print(df[col].value_counts())
def correlation_matrix(df):
    """stampa la matrice di correlazione del dataframe"""
    plt.figure(figsize=(15, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.show()
def pair_plot(df):
    """stampa il pair plot del dataframe"""
    sns.pairplot(df)
    plt.show()
def box_plot(df):
    """stampa il box plot del dataframe"""
    plt.figure(figsize=(15, 4))
    sns.boxplot(data=df, orient="h")
    plt.show()
