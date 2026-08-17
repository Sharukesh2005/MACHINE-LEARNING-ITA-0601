# ============================================================
# PROGRAM 11
# CREDIT SCORE CLASSIFICATION USING MACHINE LEARNING
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
output_file = output_folder / "11_credit_score_classification_output.txt"

# Graph output file
graph_file = output_folder / "11_credit_score_classification_graph.png"


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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CREATE DATASET
# ============================================================

data = {
    "Income": [
        25000, 30000, 45000, 50000, 60000,
        75000, 80000, 90000, 100000, 120000,
        35000, 40000, 55000, 70000, 85000
    ],

    "LoanAmount": [
        20000, 25000, 30000, 35000, 30000,
        25000, 20000, 25000, 30000, 35000,
        40000, 45000, 35000, 30000, 20000
    ],

    "Age": [
        22, 25, 28, 30, 32,
        35, 38, 40, 42, 45,
        27, 29, 33, 37, 41
    ],

    "CreditHistory": [
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 1, 1, 1
    ],

    "CreditScore": [
        "Poor",
        "Poor",
        "Average",
        "Average",
        "Good",
        "Good",
        "Good",
        "Excellent",
        "Excellent",
        "Excellent",
        "Poor",
        "Poor",
        "Average",
        "Good",
        "Good"
    ]
}


# Create DataFrame
df = pd.DataFrame(data)


# ============================================================
# DISPLAY DATASET
# ============================================================

print("============================================================")
print("CREDIT SCORE CLASSIFICATION")
print("============================================================")

print("\nDataset:")
print(df)


# ============================================================
# CONVERT CREDIT SCORE TO NUMERICAL VALUES
# ============================================================

score_mapping = {
    "Poor": 0,
    "Average": 1,
    "Good": 2,
    "Excellent": 3
}

df["CreditScore"] = df["CreditScore"].map(score_mapping)


print("\nNumerical Credit Score Labels:")
print(df["CreditScore"].values)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Income",
        "LoanAmount",
        "Age",
        "CreditHistory"
    ]
]

y = df["CreditScore"]


# ============================================================
# SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# CREATE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n============================================================")
print("TRAINING MODEL")
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
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n============================================================")
print("MODEL PERFORMANCE")
print("============================================================")

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(accuracy * 100, "%")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CREDIT SCORE DISTRIBUTION GRAPH
# ============================================================

print("\n============================================================")
print("GENERATING GRAPH")
print("============================================================")

# Create score names for graph
score_names = [
    "Poor",
    "Average",
    "Good",
    "Excellent"
]

score_counts = [
    sum(df["CreditScore"] == 0),
    sum(df["CreditScore"] == 1),
    sum(df["CreditScore"] == 2),
    sum(df["CreditScore"] == 3)
]


plt.figure(figsize=(8, 6))

plt.bar(
    score_names,
    score_counts
)

plt.xlabel("Credit Score Category")

plt.ylabel("Number of Customers")

plt.title(
    "Credit Score Classification Dataset"
)

plt.grid(
    axis="y"
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


# Display graph
plt.show()

plt.close()


# ============================================================
# PREDICT NEW CUSTOMER
# ============================================================

print("\n============================================================")
print("NEW CUSTOMER CREDIT SCORE PREDICTION")
print("============================================================")

income = float(
    input("\nEnter customer income: ")
)

loan = float(
    input("Enter loan amount: ")
)

age = float(
    input("Enter customer age: ")
)

history = int(
    input("Enter credit history (1 = Good, 0 = Bad): ")
)


# Create new customer data
new_customer = [[
    income,
    loan,
    age,
    history
]]


# Scale new data
new_customer_scaled = scaler.transform(
    new_customer
)


# Predict
prediction = model.predict(
    new_customer_scaled
)


# Convert prediction back to category
labels = {
    0: "Poor",
    1: "Average",
    2: "Good",
    3: "Excellent"
}


predicted_score = labels[
    prediction[0]
]


print("\nPredicted Credit Score:")
print(predicted_score)


# ============================================================
# PREDICTION PROBABILITY
# ============================================================

probabilities = model.predict_proba(
    new_customer_scaled
)

print("\nPrediction Probabilities:")

for i, probability in enumerate(probabilities[0]):
    print(
        labels[i],
        ":",
        round(probability * 100, 2),
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

print("\nGraph output:")
print(graph_file)

print("\nProgram completed successfully.")


# ============================================================
# CLOSE OUTPUT FILE
# ============================================================

file.close()