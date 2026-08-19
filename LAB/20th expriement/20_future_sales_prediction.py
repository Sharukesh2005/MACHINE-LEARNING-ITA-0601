# ============================================================
# PROGRAM 20
# FUTURE SALES PREDICTION USING LINEAR REGRESSION
# ============================================================

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# AUTOMATIC OUTPUT FOLDER
# ============================================================

current_folder = Path(__file__).parent

output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "20_future_sales_prediction_output.txt"

graph_file = output_folder / "20_future_sales_prediction_graph.png"


# ============================================================
# SAVE OUTPUT TO TERMINAL AND TEXT FILE
# ============================================================

class Tee:

    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for output in self.files:
            output.write(text)
            output.flush()

    def flush(self):
        for output in self.files:
            output.flush()


file = open(
    output_file,
    "w",
    encoding="utf-8"
)

sys.stdout = Tee(
    sys.__stdout__,
    file
)


# ============================================================
# CREATE SALES DATASET
# ============================================================

data = {
    "Month": [
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24
    ],

    "Advertising": [
        20, 22, 25, 28, 30, 32,
        35, 38, 40, 42, 45, 48,
        50, 52, 55, 58, 60, 62,
        65, 68, 70, 72, 75, 78
    ],

    "PreviousSales": [
        100, 105, 112, 118, 125, 132,
        140, 148, 155, 163, 170, 178,
        185, 193, 201, 210, 218, 226,
        235, 243, 252, 260, 269, 278
    ],

    "Sales": [
        105, 112, 118, 125, 132, 140,
        148, 155, 163, 170, 178, 185,
        193, 201, 210, 218, 226, 235,
        243, 252, 260, 269, 278, 288
    ]
}


# Create DataFrame
df = pd.DataFrame(data)


# ============================================================
# DISPLAY DATASET
# ============================================================

print("============================================================")
print("FUTURE SALES PREDICTION USING LINEAR REGRESSION")
print("============================================================")

print("\nDataset:")
print(df)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Month",
        "Advertising",
        "PreviousSales"
    ]
]

y = df["Sales"]


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

print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ============================================================
# CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n============================================================")
print("TRAINING LINEAR REGRESSION MODEL")
print("============================================================")

model.fit(
    X_train,
    y_train
)

print("Linear Regression model trained successfully.")


# ============================================================
# PREDICT TEST DATA
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

print("\nActual Sales:")
print(y_test.values)

print("\nPredicted Sales:")
print(y_pred)

print("\nMean Squared Error:")
print(mse)

print("\nR2 Score:")
print(r2)

print("\nR2 Score Percentage:")
print(
    round(
        r2 * 100,
        2
    ),
    "%"
)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

print("\n============================================================")
print("MODEL COEFFICIENTS")
print("============================================================")

print("\nMonth Coefficient:")
print(model.coef_[0])

print("\nAdvertising Coefficient:")
print(model.coef_[1])

print("\nPrevious Sales Coefficient:")
print(model.coef_[2])

print("\nIntercept:")
print(model.intercept_)


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

print("\n============================================================")
print("GENERATING SALES PREDICTION GRAPH")
print("============================================================")

# Sort test data by month for a cleaner graph
test_results = pd.DataFrame({
    "Month": X_test["Month"].values,
    "Actual": y_test.values,
    "Predicted": y_pred
})

test_results = test_results.sort_values(
    by="Month"
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    test_results["Month"],
    test_results["Actual"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    test_results["Month"],
    test_results["Predicted"],
    marker="x",
    linestyle="--",
    label="Predicted Sales"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Sales"
)

plt.title(
    "Future Sales Prediction using Linear Regression"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


# Save graph
plt.savefig(
    graph_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nGraph saved successfully!")

print("Graph location:")
print(graph_file)


plt.show()

plt.close()


# ============================================================
# PREDICT FUTURE SALES
# ============================================================

print("\n============================================================")
print("FUTURE SALES PREDICTION")
print("============================================================")

future_month = int(
    input(
        "\nEnter future month number: "
    )
)

future_advertising = float(
    input(
        "Enter expected advertising expenditure: "
    )
)

future_previous_sales = float(
    input(
        "Enter previous month's sales: "
    )
)


# Create future data
future_data = [[
    future_month,
    future_advertising,
    future_previous_sales
]]


# Predict future sales
future_prediction = model.predict(
    future_data
)


predicted_sales = future_prediction[0]


print("\nPredicted Future Sales:")
print(
    round(
        predicted_sales,
        2
    )
)


# ============================================================
# SALES FORECAST INTERPRETATION
# ============================================================

print("\n============================================================")
print("SALES FORECAST")
print("============================================================")

print(
    "For Month:",
    future_month
)

print(
    "Expected Advertising:",
    future_advertising
)

print(
    "Previous Sales:",
    future_previous_sales
)

print(
    "Predicted Sales:",
    round(
        predicted_sales,
        2
    )
)


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
# RESTORE NORMAL TERMINAL OUTPUT
# ============================================================

sys.stdout = sys.__stdout__

file.close()