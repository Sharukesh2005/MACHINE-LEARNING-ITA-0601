# ============================================================
# PROGRAM 14
# HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ============================================================

import sys
from pathlib import Path

# ============================================================
# AUTOMATIC OUTPUT FOLDER
# ============================================================

# Get the folder containing this Python file
current_folder = Path(__file__).parent

# Create OUTPUT folder automatically
output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

# Text output file
output_file = output_folder / "14_house_price_prediction_output.txt"

# Graph output file
graph_file = output_folder / "14_house_price_prediction_graph.png"


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
    "Area": [
        800, 1000, 1200, 1400, 1600,
        1800, 2000, 2200, 2400, 2600,
        1100, 1300, 1500, 1900, 2300
    ],

    "Bedrooms": [
        2, 2, 3, 3, 3,
        3, 4, 4, 4, 5,
        2, 3, 3, 4, 4
    ],

    "Age": [
        20, 18, 15, 12, 10,
        8, 6, 5, 4, 2,
        17, 14, 11, 7, 5
    ],

    "Price": [
        2500000, 3000000, 3600000, 4200000, 4800000,
        5400000, 6200000, 6800000, 7500000, 8500000,
        3300000, 3900000, 4500000, 5800000, 7200000
    ]
}


# Create DataFrame
df = pd.DataFrame(data)


# ============================================================
# DISPLAY DATASET
# ============================================================

print("============================================================")
print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
print("============================================================")

print("\nDataset:")
print(df)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Area",
        "Bedrooms",
        "Age"
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
# CREATE MODEL
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

print("\nActual House Prices:")
print(y_test.values)

print("\nPredicted House Prices:")
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

print("\nArea Coefficient:")
print(model.coef_[0])

print("\nBedrooms Coefficient:")
print(model.coef_[1])

print("\nAge Coefficient:")
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

# Connect actual and predicted values
plt.plot(
    range(len(y_test)),
    y_test,
    linestyle="--",
    label="Actual Trend"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    linestyle="--",
    label="Predicted Trend"
)

plt.xlabel("Test House Number")

plt.ylabel("House Price")

plt.title(
    "House Price Prediction using Linear Regression"
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
# PREDICT NEW HOUSE PRICE
# ============================================================

print("\n============================================================")
print("NEW HOUSE PRICE PREDICTION")
print("============================================================")

area = float(
    input("\nEnter house area in square feet: ")
)

bedrooms = int(
    input("Enter number of bedrooms: ")
)

age = float(
    input("Enter house age in years: ")
)


# Create new house data
new_house = [[
    area,
    bedrooms,
    age
]]


# Predict house price
prediction = model.predict(
    new_house
)


predicted_price = prediction[0]


print("\nPredicted House Price:")
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