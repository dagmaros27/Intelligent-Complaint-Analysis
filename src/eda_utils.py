import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def load_data(filepath):
    """Load dataset from CSV or similar."""
    return pd.read_csv(filepath)



def dataset_overview(df):
    print("Shape:", df.shape)
    print("\nColumns and Data Types:\n", df.dtypes)
    print("\nFirst 5 Rows:\n", df.head())

def summary_statistics(df):
    print("\nSummary Statistics:\n", df.describe())