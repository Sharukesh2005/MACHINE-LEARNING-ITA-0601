# ============================================================
# PROGRAM 15
# IRIS FLOWER CLASSIFICATION USING NAIVE BAYES
# ============================================================

import sys
from pathlib import Path

# ============================================================
# AUTOMATIC OUTPUT FOLDER
# ============================================================

current_folder = Path(__file__).parent

output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

# Output files
output_file = output_folder / "15_iris_naive_bayes_output.txt"

confusion_matrix_file = (
    output_folder / "15_iris_naive_bayes_confusion_matrix.png"
)

accuracy_graph_file = (
    output_folder / "15_iris_naive_bayes_accuracy_graph.png"
)


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
# IMPORT LIBRARIES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# LOAD IRIS DATASET
# ============================================================

iris = load_iris()

X = iris.data

y = iris.target

target_names = iris.target_names


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

print("============================================================")
print("IRIS FLOWER CLASSIFICATION USING NAIVE BAYES")
print("============================================================")

print("\nDataset loaded successfully.")

print("\nNumber of samples:")
print(len(X))

print("\nNumber of features:")
print(X.shape[1])

print("\nFeature names:")

for feature in iris.feature_names:
    print("-", feature)

print("\nFlower classes:")

for i, name in enumerate(target_names):
    print(i, "=", name)


# ============================================================
# SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n============================================================")
print("DATASET SPLIT")
print("============================================================")

print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# CREATE NAIVE BAYES MODEL
# ============================================================

model = GaussianNB()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n============================================================")
print("TRAINING NAIVE BAYES MODEL")
print("============================================================")

model.fit(
    X_train_scaled,
    y_train
)

print("Naive Bayes model trained successfully.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n============================================================")
print("MODEL PERFORMANCE")
print("============================================================")

print("\nActual Labels:")
print(y_test)

print("\nPredicted Labels:")
print(y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n============================================================")
print("CLASSIFICATION REPORT")
print("============================================================")

report = classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n============================================================")
print("CONFUSION MATRIX")
print("============================================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# Create confusion matrix display
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

disp.plot(
    ax=ax
)

ax.set_title(
    "Iris Classification using Naive Bayes"
)

plt.tight_layout()


# Save confusion matrix
plt.savefig(
    confusion_matrix_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion matrix saved to:")
print(confusion_matrix_file)


# Display confusion matrix
plt.show()

plt.close()


# ============================================================
# ACCURACY GRAPH
# ============================================================

print("\n============================================================")
print("GENERATING ACCURACY GRAPH")
print("============================================================")

plt.figure(
    figsize=(7, 5)
)

plt.bar(
    ["Naive Bayes"],
    [accuracy * 100]
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Classification Algorithm"
)

plt.title(
    "Naive Bayes Classification Accuracy"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y"
)


# Save accuracy graph
plt.savefig(
    accuracy_graph_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nAccuracy graph saved to:")
print(accuracy_graph_file)


# Display graph
plt.show()

plt.close()


# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_test_scaled
)

print("\n============================================================")
print("PREDICTION PROBABILITIES")
print("============================================================")

print("\nFirst 10 test samples:")

for i in range(
    min(
        10,
        len(probabilities)
    )
):

    print(
        "Sample",
        i + 1,
        ":",
        probabilities[i]
    )


# ============================================================
# NEW FLOWER PREDICTION
# ============================================================

print("\n============================================================")
print("NEW IRIS FLOWER PREDICTION")
print("============================================================")

print("\nEnter flower measurements:")

sepal_length = float(
    input(
        "Enter sepal length (cm): "
    )
)

sepal_width = float(
    input(
        "Enter sepal width (cm): "
    )
)

petal_length = float(
    input(
        "Enter petal length (cm): "
    )
)

petal_width = float(
    input(
        "Enter petal width (cm): "
    )
)


# Create new flower
new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]


# Scale new flower
new_flower_scaled = scaler.transform(
    new_flower
)


# Predict flower
prediction = model.predict(
    new_flower_scaled
)


# Get flower name
predicted_flower = target_names[
    prediction[0]
]


print("\nPredicted Flower:")
print(predicted_flower)


# ============================================================
# NEW FLOWER PROBABILITIES
# ============================================================

new_probability = model.predict_proba(
    new_flower_scaled
)


print("\nPrediction Probabilities:")

for i, probability in enumerate(
    new_probability[0]
):

    print(
        target_names[i],
        ":",
        round(
            probability * 100,
            2
        ),
        "%"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n============================================================")
print("OUTPUT SAVED SUCCESSFULLY")
print("============================================================")

print("\nText output:")
print(output_file)

print("\nConfusion matrix:")
print(confusion_matrix_file)

print("\nAccuracy graph:")
print(accuracy_graph_file)

print("\nProgram completed successfully.")


# ============================================================
# CLOSE OUTPUT FILE
# ============================================================

file.close()