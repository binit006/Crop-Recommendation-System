import pandas as pd

def load_data():
    df = pd.read_csv("Crop_recommendation.csv")

    print("First 5 rows:\n", df.head())
    print("\nDataset Info:\n")
    print(df.info())

    print("\nMissing Values:\n", df.isnull().sum())

    return df
