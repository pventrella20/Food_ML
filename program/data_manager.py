import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from program.config import CSV_SEPARATOR, DATASET_PATH
def readCSV(path=DATASET_PATH, separ=CSV_SEPARATOR):
    """
    Read a .csv file and return a pandas DataFrame.
    :param path: .csv file path to read (default: project dataset)
    :param separ: separator character
    :return: DataFrame containing extracted data
    """
    # 'on_bad_lines' replaces 'error_bad_lines' removed in pandas >= 2.0
    return pd.read_csv(path, sep=separ, on_bad_lines="skip")
def data_analysis(df, feature):
    """Data analysis (histogram + boxplot) for a feature."""
    f, axes = plt.subplots(1, 2, figsize=(20, 4))
    sns.histplot(data=df[feature], ax=axes[0])
    sns.boxplot(data=df[feature], ax=axes[1])
    plt.show()
def column_analysis(df, col):
    """Print the value count for a feature."""
    print(df[col].value_counts())
def correlation_matrix(df):
    """Display the DataFrame correlation matrix."""
    plt.figure(figsize=(15, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.show()
def pair_plot(df):
    """Display the DataFrame pair plot."""
    sns.pairplot(df)
    plt.show()
def box_plot(df):
    """Display the DataFrame box plot."""
    plt.figure(figsize=(15, 4))
    sns.boxplot(data=df, orient="h")
    plt.show()


# Backward-compatible aliases
data_analisys = data_analysis
column_analisys = column_analysis

