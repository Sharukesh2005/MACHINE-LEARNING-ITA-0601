# ============================================================
# PROGRAM 18
# PERCEPTRON BASED CLASSIFICATION
# ============================================================

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# AUTOMATIC OUTPUT FOLDER
# ============================================================

current_folder = Path(__file__).parent

output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "18_perceptron_output.txt"
graph_file = output_folder / "18_perceptron_graph.png"


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
# DATASET
# ============================================================
# Features:
# X1 = Study Hours
# X2 = Attendance Percentage
#
# Class:
# 0 = Fail
# 1 = Pass
# ============================================================

X = np.array([
    [1, 50],
    [2, 55],
    [2, 60],
    [3, 58],
    [3, 65],
    [4, 60],
    [4, 70],
    [5, 65],
    [5, 75],
    [6, 70],
    [6, 80],
    [7, 75],
    [7, 85],
    [8, 80],
    [8, 90],
    [9, 85],
    [9, 95],
    [10, 90]
], dtype=float)

y = np.array([
    0, 0, 0, 0, 0, 0,
    1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1
])


# ============================================================
# DISPLAY DATASET
# ============================================================

print("============================================================")
print("PERCEPTRON BASED CLASSIFICATION")
print("============================================================")

print("\nDataset:")
print("Study Hours | Attendance | Result")
print("-----------------------------------")

for i in range(len(X)):

    result = "Pass" if y[i] == 1 else "Fail"

    print(
        f"{X[i][0]:11.0f} | "
        f"{X[i][1]:10.0f}% | "
        f"{result}"
    )


# ============================================================
# NORMALIZE DATA
# ============================================================

X_scaled = X.copy()

X_scaled[:, 0] = X_scaled[:, 0] / 10
X_scaled[:, 1] = X_scaled[:, 1] / 100


# ============================================================
# PERCEPTRON PARAMETERS
# ============================================================

learning_rate = 0.1
epochs = 20

weights = np.zeros(
    X_scaled.shape[1]
)

bias = 0


# ============================================================
# STEP ACTIVATION FUNCTION
# ============================================================

def step_function(value):

    if value >= 0:
        return 1

    return 0


# ============================================================
# TRAIN PERCEPTRON
# ============================================================

print("\n============================================================")
print("PERCEPTRON TRAINING")
print("============================================================")

print("\nLearning Rate:")
print(learning_rate)

print("\nNumber of Epochs:")
print(epochs)


for epoch in range(epochs):

    total_errors = 0

    for i in range(len(X_scaled)):

        weighted_sum = np.dot(
            X_scaled[i],
            weights
        ) + bias

        prediction = step_function(
            weighted_sum
        )

        error = y[i] - prediction

        weights = weights + (
            learning_rate *
            error *
            X_scaled[i]
        )

        bias = bias + (
            learning_rate *
            error
        )

        if error != 0:
            total_errors += 1

    print(
        "Epoch",
        epoch + 1,
        "Errors:",
        total_errors
    )


# ============================================================
# FINAL PARAMETERS
# ============================================================

print("\n============================================================")
print("FINAL PERCEPTRON PARAMETERS")
print("============================================================")

print("\nWeights:")
print(weights)

print("\nBias:")
print(bias)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(X_data):

    predictions = []

    for sample in X_data:

        weighted_sum = np.dot(
            sample,
            weights
        ) + bias

        prediction = step_function(
            weighted_sum
        )

        predictions.append(prediction)

    return np.array(predictions)


# ============================================================
# PREDICT DATASET
# ============================================================

y_pred = predict(X_scaled)


# ============================================================
# ACCURACY
# ============================================================

correct = np.sum(
    y_pred == y
)

total = len(y)

accuracy = (
    correct / total
) * 100


print("\n============================================================")
print("MODEL PERFORMANCE")
print("============================================================")

print("\nActual Classes:")
print(y)

print("\nPredicted Classes:")
print(y_pred)

print("\nCorrect Predictions:")
print(correct)

print("\nTotal Predictions:")
print(total)

print("\nAccuracy:")
print(
    round(
        accuracy,
        2
    ),
    "%"
)


# ============================================================
# PREDICTION RESULTS
# ============================================================

print("\n============================================================")
print("PREDICTION RESULTS")
print("============================================================")

print(
    "\nStudy Hours | Attendance | Actual | Predicted"
)

print("----------------------------------------------")

for i in range(len(X)):

    actual = (
        "Pass"
        if y[i] == 1
        else "Fail"
    )

    predicted = (
        "Pass"
        if y_pred[i] == 1
        else "Fail"
    )

    print(
        f"{X[i][0]:11.0f} | "
        f"{X[i][1]:10.0f}% | "
        f"{actual:6} | "
        f"{predicted}"
    )


# ============================================================
# GRAPH
# ============================================================

print("\n============================================================")
print("GENERATING GRAPH")
print("============================================================")

plt.figure(
    figsize=(9, 6)
)

fail = y == 0

plt.scatter(
    X[fail, 0],
    X[fail, 1],
    label="Fail"
)

passed = y == 1

plt.scatter(
    X[passed, 0],
    X[passed, 1],
    label="Pass"
)


# Decision boundary

x_values = np.linspace(
    0,
    11,
    100
)

w1 = weights[0]
w2 = weights[1]

b = bias

if w2 != 0:

    x_scaled = x_values / 10

    y_scaled = -(
        w1 * x_scaled + b
    ) / w2

    y_values = y_scaled * 100

    plt.plot(
        x_values,
        y_values,
        label="Decision Boundary"
    )


plt.xlabel(
    "Study Hours"
)

plt.ylabel(
    "Attendance Percentage"
)

plt.title(
    "Perceptron Based Classification"
)

plt.legend()

plt.grid(True)

plt.xlim(
    0,
    11
)

plt.ylim(
    40,
    100
)


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
# NEW STUDENT PREDICTION
# ============================================================

print("\n============================================================")
print("NEW STUDENT PREDICTION")
print("============================================================")

study_hours = float(
    input(
        "\nEnter study hours: "
    )
)

attendance = float(
    input(
        "Enter attendance percentage: "
    )
)


# Normalize input

new_student = np.array([
    study_hours / 10,
    attendance / 100
])


# Calculate prediction

weighted_sum = np.dot(
    new_student,
    weights
) + bias

prediction = step_function(
    weighted_sum
)


if prediction == 1:
    result = "PASS"
else:
    result = "FAIL"


print("\nPredicted Result:")
print(result)


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