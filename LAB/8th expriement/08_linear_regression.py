import sys
from pathlib import Path

# ---------------------------------------
# AUTOMATIC OUTPUT SAVING
# ---------------------------------------

# Get the folder containing this Python file
current_folder = Path(__file__).parent

# Create OUTPUT folder automatically
output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

# Output file
output_file = output_folder / "08_linear_regression_output.txt"


# Display output in terminal AND save it
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

sys.stdout = Tee(sys.__stdout__, file)


# ---------------------------------------
# LINEAR REGRESSION
# ---------------------------------------

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Dataset
# Hours studied
X = np.array([
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10
]).reshape(-1, 1)

# Marks obtained
y = np.array([
    35, 40, 50, 55, 65,
    70, 75, 80, 90, 95
])


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Linear Regression model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# ---------------------------------------
# DISPLAY RESULTS
# ---------------------------------------

print("Linear Regression")
print("-----------------")

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nActual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)


# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)


# ---------------------------------------
# PREDICT NEW VALUE
# ---------------------------------------

print("\n---------------------------------------")

hours = float(input("Enter number of study hours: "))

prediction = model.predict([[hours]])

print("Predicted Marks:", prediction[0])


# ---------------------------------------
# OUTPUT LOCATION
# ---------------------------------------

print("\n---------------------------------------")
print("OUTPUT SAVED SUCCESSFULLY")
print("---------------------------------------")
print("File:", output_file)


# Close output file
file.close()