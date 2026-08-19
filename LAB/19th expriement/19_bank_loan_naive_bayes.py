# ============================================================
# PROGRAM 19
# NAIVE BAYES CLASSIFICATION FOR BANK LOAN PREDICTION
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# AUTOMATIC OUTPUT FOLDER
# ============================================================

current_folder = Path(__file__).parent

output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "19_bank_loan_naive_bayes_output.txt"

confusion_matrix_file = (
    output_folder /
    "19_bank_loan_confusion_matrix.png"
)

accuracy_graph_file = (
    output_folder /
    "19_bank_loan_accuracy.png"
)


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
# CREATE BANK LOAN DATASET
# ============================================================

data = {
    "Age": [
        22, 25, 28, 30, 32,
        35, 38, 40, 42, 45,
        23, 27, 31, 34, 37,
        41, 44, 29, 33, 39,
        48, 50, 26, 36, 43,
        24, 30, 35, 46, 52
    ],

    "Income": [
        25000, 30000, 35000, 40000, 45000,
        50000, 55000, 60000, 65000, 70000,
        28000, 33000, 42000, 48000, 52000,
        58000, 68000, 37000, 46000, 62000,
        75000, 80000, 32000, 54000, 72000,
        27000, 43000, 51000, 78000, 85000
    ],

    "LoanAmount": [
        150000, 180000, 200000, 220000, 250000,
        280000, 300000, 320000, 350000, 380000,
        160000, 190000, 230000, 260000, 290000,
        310000, 360000, 210000, 270000, 330000,
        400000, 420000, 170000, 300000, 370000,
        140000, 240000, 280000, 390000, 450000
    ],

    "CreditScore": [
        550, 580, 600, 620, 650,
        680, 700, 720, 740, 760,
        560, 590, 630, 660, 690,
        710, 750, 610, 670, 730,
        780, 800, 570, 700, 770,
        540, 640, 680, 790, 810
    ],

    "ExistingLoans": [
        3, 3, 2, 2, 2,
        1, 1, 1, 0, 0,
        3, 2, 2, 2, 1,
        1, 0, 3, 2, 1,
        0, 0, 3, 1, 0,
        3, 2, 1, 0, 0
    ],

    "LoanTerm": [
        60, 60, 48, 48, 48,
        36, 36, 36, 36, 36,
        60, 48, 48, 48, 36,
        36, 36, 60, 48, 36,
        36, 36, 60, 36, 36,
        60, 48, 36, 36, 36
    ],

    # 0 = Not Eligible
    # 1 = Eligible
    "LoanStatus": [
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 1, 1, 1,
        1, 1, 0, 1, 1,
        1, 1, 0, 1, 1,
        0, 1, 1, 1, 1
    ]
}


# Create DataFrame
df = pd.DataFrame(data)


# ============================================================
# DISPLAY DATASET
# ============================================================

print("============================================================")
print("NAIVE BAYES CLASSIFICATION FOR BANK LOAN PREDICTION")
print("============================================================")

print("\nDataset:")
print(df)


print("\nLoan Status:")
print("0 = Not Eligible")
print("1 = Eligible")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "ExistingLoans",
        "LoanTerm"
    ]
]

y = df["LoanStatus"]


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
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n============================================================")
print("MODEL PERFORMANCE")
print("============================================================")

print("\nActual Loan Status:")
print(y_test.values)

print("\nPredicted Loan Status:")
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
    labels=[0, 1],
    target_names=[
        "Not Eligible",
        "Eligible"
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
    labels=[0, 1]
)

print(cm)


# Create confusion matrix graph
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Eligible",
        "Eligible"
    ]
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

disp.plot(
    ax=ax
)

ax.set_title(
    "Bank Loan Prediction - Naive Bayes"
)

plt.tight_layout()


# Save graph
plt.savefig(
    confusion_matrix_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion matrix saved to:")
print(confusion_matrix_file)

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
    "Algorithm"
)

plt.title(
    "Bank Loan Prediction Accuracy"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y"
)

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

for i in range(len(probabilities)):

    print(
        "Test Sample",
        i + 1,
        ":",
        "Not Eligible =",
        round(probabilities[i][0] * 100, 2),
        "%",
        "| Eligible =",
        round(probabilities[i][1] * 100, 2),
        "%"
    )


# ============================================================
# NEW CUSTOMER LOAN PREDICTION
# ============================================================

print("\n============================================================")
print("NEW CUSTOMER LOAN PREDICTION")
print("============================================================")

age = float(
    input("\nEnter customer age: ")
)

income = float(
    input("Enter monthly income: ")
)

loan_amount = float(
    input("Enter requested loan amount: ")
)

credit_score = float(
    input("Enter credit score: ")
)

existing_loans = float(
    input("Enter number of existing loans: ")
)

loan_term = float(
    input("Enter loan term in months: ")
)


# Create new customer
new_customer = [[
    age,
    income,
    loan_amount,
    credit_score,
    existing_loans,
    loan_term
]]


# Scale new customer
new_customer_scaled = scaler.transform(
    new_customer
)


# Predict
prediction = model.predict(
    new_customer_scaled
)


if prediction[0] == 1:
    loan_result = "ELIGIBLE"
else:
    loan_result = "NOT ELIGIBLE"


print("\nLoan Prediction:")
print(loan_result)


# ============================================================
# NEW CUSTOMER PROBABILITY
# ============================================================

new_probability = model.predict_proba(
    new_customer_scaled
)


print("\nPrediction Probability:")

print(
    "Not Eligible:",
    round(
        new_probability[0][0] * 100,
        2
    ),
    "%"
)

print(
    "Eligible:",
    round(
        new_probability[0][1] * 100,
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
# RESTORE NORMAL TERMINAL OUTPUT
# ============================================================

sys.stdout = sys.__stdout__

file.close()