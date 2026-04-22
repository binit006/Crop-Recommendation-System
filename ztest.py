import pandas as pd
from load import load_data
from eda import perform_eda
from ml import train_model

def predict_crop(model, columns):
    print("\nEnter values:")

    N = float(input("Nitrogen: "))
    P = float(input("Phosphorus: "))
    K = float(input("Potassium: "))
    temp = float(input("Temperature: "))
    humidity = float(input("Humidity: "))
    ph = float(input("pH: "))
    rainfall = float(input("Rainfall: "))

    data = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall]],
                        columns=columns)

    prediction = model.predict(data)

    print("\n🌱 Recommended Crop:", prediction[0])


if __name__ == "__main__":
    df = load_data()
    perform_eda(df)
    model, columns = train_model(df)
    predict_crop(model, columns)
