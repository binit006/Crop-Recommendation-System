# =====================================
# 1. IMPORT LIBRARIES
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================
# 2. LOAD DATASET
# =====================================
df = pd.read_csv(r"C:\Users\binit\Downloads\Crop_recommendation.csv")

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

# =====================================
# 3. DATA VISUALIZATION
# =====================================

# 🔥 1. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.drop("label", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 🔥 2. Histogram
df.hist(figsize=(12,10), bins=20)
plt.suptitle("Feature Distribution")
plt.show()

# 🔥 3. Boxplot (Outliers)
plt.figure(figsize=(12,6))
sns.boxplot(data=df.drop("label", axis=1))
plt.title("Outlier Detection using Boxplot")
plt.xticks(rotation=45)
plt.show()

# 🔥 4. Outlier Pie Chart
outlier_count = 0
normal_count = 0

for col in df.columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    
    outlier_count += len(outliers)
    normal_count += len(df) - len(outliers)

plt.pie([outlier_count, normal_count],
        labels=["Outliers", "Normal"],
        autopct="%1.1f%%",
        colors=["red", "green"])
plt.title("Outliers vs Normal Data")
plt.show()

# 🔥 5. Pairplot (sample for speed)
sns.pairplot(df.sample(500), hue="label")
plt.show()

# 🔥 6. Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(df["temperature"], df["humidity"], alpha=0.5)
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Temperature vs Humidity")
plt.show()

# 🔥 7. Count Plot (Crop distribution)
plt.figure(figsize=(12,6))
sns.countplot(x="label", data=df)
plt.xticks(rotation=90)
plt.title("Crop Distribution")
plt.show()

# =====================================
# 4. PREPARE DATA
# =====================================
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 5. TRAIN MODEL
# =====================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =====================================
# 6. EVALUATION
# =====================================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix with Labels")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# =====================================
# 7. FEATURE IMPORTANCE
# =====================================
importance = model.feature_importances_
features = X.columns

sorted_idx = np.argsort(importance)

plt.figure(figsize=(8,5))
plt.barh(range(len(sorted_idx)), importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), features[sorted_idx])
plt.title("Feature Importance")
plt.show()
# =====================================
# 8. USER INPUT PREDICTION
# =====================================
def predict_crop():
    print("\nEnter values:")

    N = float(input("Nitrogen: "))
    P = float(input("Phosphorus: "))
    K = float(input("Potassium: "))
    temp = float(input("Temperature: "))
    humidity = float(input("Humidity: "))
    ph = float(input("pH: "))
    rainfall = float(input("Rainfall: "))

    data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    prediction = model.predict(data)

    print("\n🌱 Recommended Crop:", prediction[0])

# Run prediction
predict_crop()


