import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):

    # Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.drop("label", axis=1).corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Histogram
    df.hist(figsize=(12,10), bins=20)
    plt.suptitle("Feature Distribution")
    plt.show()

    # Boxplot
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df.drop("label", axis=1))
    plt.title("Outlier Detection")
    plt.xticks(rotation=45)
    plt.show()

    # Scatter Plot
    plt.figure(figsize=(8,5))
    plt.scatter(df["temperature"], df["humidity"], alpha=0.5)
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.title("Temperature vs Humidity")
    plt.show()

    # Count Plot
    plt.figure(figsize=(12,6))
    sns.countplot(x="label", data=df)
    plt.xticks(rotation=90)
    plt.title("Crop Distribution")
    plt.show()
