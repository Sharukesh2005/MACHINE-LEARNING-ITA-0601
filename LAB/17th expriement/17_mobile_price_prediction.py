# ============================================================
# PROGRAM 17
# MOBILE PRICE PREDICTION USING RANDOM FOREST CLASSIFIER
# ============================================================

import sys
from pathlib import Path

# ============================================================
# AUTOMATIC OUTPUT FOLDER
# ============================================================

current_folder = Path(__file__).parent

output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "17_mobile_price_prediction_output.txt"

accuracy_graph_file = (
    output_folder / "17_mobile_price_prediction_accuracy.png"
)

confusion_matrix_file = (
    output_folder / "17_mobile_price_prediction_confusion_matrix.png"
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

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CREATE MOBILE DATASET
# ============================================================

data = {
    "RAM": [
        2, 2, 3, 3, 4,
        4, 4, 6, 6, 8,
        8, 8, 12, 12, 16,
        2, 3, 4, 6, 8
    ],

    "Storage": [
        32, 64, 64, 128, 64,
        128, 256, 128, 256, 128,
        256, 512, 256, 512, 512,
        32, 64, 128, 256, 512
    ],

    "Battery": [
        3000, 3500, 4000, 4000, 4500,
        4500, 5000, 4500, 5000, 4500,
        5000, 5500, 5000, 5500, 6000,
        2800, 4000, 4500, 5000, 5500
    ],

    "Camera": [
        12, 13, 16, 16, 20,
        24, 32, 32, 48, 48,
        50, 64, 64, 108, 108,
        12, 16, 20, 48, 64
    ],

    "PriceRange": [
        0, 0, 0, 1, 1,
        1, 1, 1, 2, 2,
        2, 2, 3, 3, 3,
        0, 0, 1, 2, 3
    ]
}


# Create DataFrame
df = pd.DataFrame(data)


# ============================================================
# PRICE RANGE INFORMATION
# ============================================================

price_labels = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Premium"
}


# ============================================================
# DISPLAY DATASET
# ============================================================

print("============================================================")
print("MOBILE PRICE PREDICTION")
print("============================================================")

print("\nDataset:")
print(df)


print("\nPrice Range:")
print("0 = Low")
print("1 = Medium")
print("2 = High")
print("3 = Premium")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[
    [
        "RAM",
        "Storage",
        "Battery",
        "Camera"
    ]
]

y = df["PriceRange"]


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
# CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n============================================================")
print("TRAINING RANDOM FOREST MODEL")
print("============================================================")

model.fit(
    X_train_scaled,
    y_train
)

print("Model trained successfully.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n============================================================")
print("MODEL PERFORMANCE")
print("============================================================")

print("\nActual Price Ranges:")
print(y_test.values)

print("\nPredicted Price Ranges:")
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
    labels=[0, 1, 2, 3],
    target_names=[
        "Low",
        "Medium",
        "High",
        "Premium"
    ],
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
    y_pred,
    labels=[0, 1, 2, 3]
)

print(cm)


# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Low",
        "Medium",
        "High",
        "Premium"
    ]
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

disp.plot(
    ax=ax
)

ax.set_title(
    "Mobile Price Prediction - Confusion Matrix"
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


# Display graph
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
    ["Random Forest"],
    [accuracy * 100]
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Algorithm"
)

plt.title(
    "Mobile Price Prediction Accuracy"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y"
)


# Display accuracy value
plt.text(
    0,
    accuracy * 100 + 2,
    f"{accuracy * 100:.2f}%",
    ha="center"
)

plt.tight_layout()


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
# NEW MOBILE PRICE PREDICTION
# ============================================================

print("\n============================================================")
print("NEW MOBILE PRICE PREDICTION")
print("============================================================")

ram = float(
    input("\nEnter RAM in GB: ")
)

storage = float(
    input("Enter storage in GB: ")
)

battery = float(
    input("Enter battery capacity in mAh: ")
)

camera = float(
    input("Enter main camera resolution in MP: ")
)


# Create new mobile data
new_mobile = [[
    ram,
    storage,
    battery,
    camera
]]


# Scale new mobile
new_mobile_scaled = scaler.transform(
    new_mobile
)


# Predict price range
prediction = model.predict(
    new_mobile_scaled
)


predicted_range = price_labels[
    prediction[0]
]


print("\nPredicted Mobile Price Range:")
print(predicted_range)


# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    new_mobile_scaled
)

print("\nPrediction Probabilities:")

for class_number, probability in zip(
    model.classes_,
    probabilities[0]
):

    print(
        price_labels[class_number],
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

print("\nAccuracy graph:")
print(accuracy_graph_file)

print("\nConfusion matrix:")
print(confusion_matrix_file)

print("\nProgram completed successfully.")


# ============================================================
# CLOSE OUTPUT FILE
# ============================================================

file.close()