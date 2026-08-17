import sys
from pathlib import Path

# ---------------------------------------
# AUTOMATIC OUTPUT SAVING
# ---------------------------------------

# Get the folder where this Python file is located
current_folder = Path(__file__).parent

# Create output folder automatically
output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

# Output file
output_file = output_folder / "07_logistic_regression_output.txt"


# Class to display output in terminal AND save it to file
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

# Display in terminal and save to file
sys.stdout = Tee(sys.__stdout__, file)


# ---------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Logistic Regression model
model = LogisticRegression(max_iter=10000)


# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)


# Display results
print("Logistic Regression")
print("-------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n---------------------------------------")
print("OUTPUT SAVED SUCCESSFULLY")
print("---------------------------------------")
print("File:", output_file)


# Close output file
file.close()