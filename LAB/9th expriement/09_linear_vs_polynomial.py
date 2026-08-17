# ============================================================
# PROGRAM 9
# COMPARE LINEAR AND POLYNOMIAL REGRESSION USING PYTHON
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
output_file = output_folder / "09_linear_vs_polynomial_output.txt"

# Graph output file
graph_file = output_folder / "09_linear_vs_polynomial_graph.png"


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

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# DATASET
# ============================================================

# Input values
X = np.array([
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
]).reshape(-1, 1)

# Output values
y = np.array([
    2,
    5,
    8,
    18,
    25,
    40,
    55,
    72,
    90,
    115
])


# ============================================================
# DISPLAY DATASET
# ============================================================

print("============================================================")
print("COMPARE LINEAR AND POLYNOMIAL REGRESSION")
print("============================================================")

print("\nDataset")
print("-------")

print("X values:")
print(X.flatten())

print("\nY values:")
print(y)


# ============================================================
# LINEAR REGRESSION
# ============================================================

print("\n============================================================")
print("LINEAR REGRESSION")
print("============================================================")

# Create Linear Regression model
linear_model = LinearRegression()

# Train model
linear_model.fit(X, y)

# Predict values
linear_prediction = linear_model.predict(X)

# Calculate Mean Squared Error
linear_mse = mean_squared_error(
    y,
    linear_prediction
)

# Calculate R2 score
linear_r2 = r2_score(
    y,
    linear_prediction
)

# Display results
print("\nSlope:")
print(linear_model.coef_[0])

print("\nIntercept:")
print(linear_model.intercept_)

print("\nPredicted Values:")
print(linear_prediction)

print("\nMean Squared Error:")
print(linear_mse)

print("\nR2 Score:")
print(linear_r2)


# ============================================================
# POLYNOMIAL REGRESSION
# ============================================================

print("\n============================================================")
print("POLYNOMIAL REGRESSION")
print("============================================================")

# Create polynomial features
poly = PolynomialFeatures(
    degree=2
)

# Transform X into polynomial features
X_poly = poly.fit_transform(X)

# Create Linear Regression model
poly_model = LinearRegression()

# Train Polynomial Regression model
poly_model.fit(
    X_poly,
    y
)

# Predict values
poly_prediction = poly_model.predict(
    X_poly
)

# Calculate Mean Squared Error
poly_mse = mean_squared_error(
    y,
    poly_prediction
)

# Calculate R2 score
poly_r2 = r2_score(
    y,
    poly_prediction
)

# Display results
print("\nPolynomial Coefficients:")
print(poly_model.coef_)

print("\nIntercept:")
print(poly_model.intercept_)

print("\nPredicted Values:")
print(poly_prediction)

print("\nMean Squared Error:")
print(poly_mse)

print("\nR2 Score:")
print(poly_r2)


# ============================================================
# COMPARISON
# ============================================================

print("\n============================================================")
print("PERFORMANCE COMPARISON")
print("============================================================")

print("\nLinear Regression:")
print("MSE     =", linear_mse)
print("R2 Score =", linear_r2)

print("\nPolynomial Regression:")
print("MSE     =", poly_mse)
print("R2 Score =", poly_r2)


# ============================================================
# DETERMINE BETTER MODEL
# ============================================================

print("\n============================================================")
print("BEST MODEL")
print("============================================================")

if poly_r2 > linear_r2 and poly_mse < linear_mse:

    print("Polynomial Regression performs better.")

elif linear_r2 > poly_r2 and linear_mse < poly_mse:

    print("Linear Regression performs better.")

else:

    print("Both models have similar performance.")


# ============================================================
# GRAPH
# ============================================================

print("\n============================================================")
print("GENERATING GRAPH")
print("============================================================")

# Create figure
plt.figure(figsize=(9, 6))

# Plot actual data
plt.scatter(
    X,
    y,
    label="Actual Data"
)

# Plot Linear Regression
plt.plot(
    X,
    linear_prediction,
    label="Linear Regression"
)

# Plot Polynomial Regression
plt.plot(
    X,
    poly_prediction,
    label="Polynomial Regression"
)

# Labels
plt.xlabel("X")

plt.ylabel("Y")

plt.title(
    "Comparison of Linear and Polynomial Regression"
)

# Legend
plt.legend()

# Grid
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

# Close graph
plt.close()


# ============================================================
# FINAL OUTPUT INFORMATION
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