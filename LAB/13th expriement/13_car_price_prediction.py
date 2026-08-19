# ============================================================
# PROGRAM 13
# CAR PRICE PREDICTION USING LINEAR REGRESSION
# ============================================================

import sys
from pathlib import Path

# ============================================================
# AUTOMATIC OUTPUT FOLDER
# ============================================================

# Get the folder where this Python file is located
current_folder = Path(__file__).parent

# Create OUTPUT folder automatically
output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

# Text output file
output_file = output_folder / "13_car_price_prediction_output.txt"

# Graph output file
graph_file = output_folder / "13_car_price_prediction_graph.png"


# ============================================================
# SAVE OUTPUT TO TERMINAL AND TEXT FILE
# ============================================================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


# Open output file
file = open(output_file, "w", encoding="utf-8")

# Display output in terminal AND save to text file
sys.stdout = Tee(sys.__stdout__, file)


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# CREATE DATASET
# ============================================================

data = {
    "Age": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        2, 4, 6, 8, 3
    ],

    "Mileage": [
        10000, 20000, 30000, 40000, 50000,
        60000, 70000, 80000, 90000, 100000,
        18000, 42000, 65000, 85000, 28000
    ],

    "EngineSize": [
        2.0, 2.0, 1.8, 1.8, 1.6,
        1.6, 1.5, 1.5, 1.4, 1.4,
        2.0, 1.8, 1.6, 1.5, 2.0
    ],

    "Price": [
        1800000, 1650000, 1500000, 1350000, 1200000,
        1050000, 900000, 780000, 650000, 550000,
        1700000, 1400000, 1000000, 700000, 1550000
    ]
}


# Create DataFrame
df = pd.DataFrame(data)


# ============================================================
# DISPLAY DATASET
# ============================================================

print("============================================================")
print("CAR PRICE PREDICTION USING LINEAR REGRESSION")
print("============================================================")

print("\nDataset:")
print(df)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Age",
        "Mileage",
        "EngineSize"
    ]
]

y = df["Price"]


# ============================================================
# SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n============================================================")
print("DATASET SPLIT")
print("============================================================")

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n============================================================")
print("TRAINING MODEL")
print("============================================================")

model.fit(
    X_train,
    y_train
)

print("Linear Regression model trained successfully.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# MODEL EVALUATION
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n============================================================")
print("MODEL PERFORMANCE")
print("============================================================")

print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

print("\nMean Squared Error:")
print(mse)

print("\nR2 Score:")
print(r2)

print("\nR2 Score Percentage:")
print(round(r2 * 100, 2), "%")


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

print("\n============================================================")
print("MODEL COEFFICIENTS")
print("============================================================")

print("\nAge Coefficient:")
print(model.coef_[0])

print("\nMileage Coefficient:")
print(model.coef_[1])

print("\nEngine Size Coefficient:")
print(model.coef_[2])

print("\nIntercept:")
print(model.intercept_)


# ============================================================
# GRAPH
# ============================================================

print("\n============================================================")
print("GENERATING GRAPH")
print("============================================================")

plt.figure(figsize=(10, 6))

# Actual prices
plt.scatter(
    range(len(y_test)),
    y_test,
    label="Actual Price"
)

# Predicted prices
plt.scatter(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted Price"
)

plt.xlabel("Test Car Number")

plt.ylabel("Car Price")

plt.title(
    "Car Price Prediction using Linear Regression"
)

plt.legend()

plt.grid(True)


# Save graph
plt.savefig(
    graph_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nGraph saved successfully!")

print("Graph location:")
print(graph_file)


# Display graph
plt.show()

plt.close()


# ============================================================
# PREDICT PRICE OF A NEW CAR
# ============================================================

print("\n============================================================")
print("NEW CAR PRICE PREDICTION")
print("============================================================")

age = float(
    input("\nEnter car age in years: ")
)

mileage = float(
    input("Enter mileage in km: ")
)

engine_size = float(
    input("Enter engine size in litres: ")
)


# Create new car data
new_car = [[
    age,
    mileage,
    engine_size
]]


# Predict price
prediction = model.predict(
    new_car
)


predicted_price = prediction[0]


print("\nPredicted Car Price:")
print("₹", round(predicted_price, 2))


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n============================================================")
print("OUTPUT SAVED SUCCESSFULLY")
print("============================================================")

print("\nText output:")
print(output_file)

print("\nGraph output:")
print(graph_file)

print("\nProgram completed successfully.")


# ============================================================
# CLOSE OUTPUT FILE
# ============================================================

file.close()