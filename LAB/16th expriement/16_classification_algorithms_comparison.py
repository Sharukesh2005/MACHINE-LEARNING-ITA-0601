# ============================================================
# PROGRAM 16
# COMPARISON OF DIFFERENT CLASSIFICATION ALGORITHMS
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
output_file = output_folder / "16_classification_algorithms_output.txt"
graph_file = output_folder / "16_classification_algorithms_accuracy_graph.png"


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

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report
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
print("COMPARISON OF DIFFERENT CLASSIFICATION ALGORITHMS")
print("============================================================")

print("\nDataset: Iris Flower Dataset")

print("\nNumber of samples:")
print(len(X))

print("\nNumber of features:")
print(X.shape[1])

print("\nFeatures:")

for feature in iris.feature_names:
    print("-", feature)

print("\nClasses:")

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
# CREATE CLASSIFICATION MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC()
}


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

predictions = {}


print("\n============================================================")
print("TRAINING CLASSIFICATION MODELS")
print("============================================================")


for name, model in models.items():

    print("\n--------------------------------------------")
    print(name)
    print("--------------------------------------------")

    # Train model
    model.fit(
        X_train_scaled,
        y_train
    )

    print("Model trained successfully.")

    # Prediction
    y_pred = model.predict(
        X_test_scaled
    )

    predictions[name] = y_pred

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    results[name] = accuracy

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

    # Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            zero_division=0
        )
    )


# ============================================================
# DISPLAY COMPARISON TABLE
# ============================================================

print("\n============================================================")
print("CLASSIFICATION ALGORITHM COMPARISON")
print("============================================================")

print(
    "\n{:<25} {:<15}".format(
        "Algorithm",
        "Accuracy"
    )
)

print("-" * 40)

for name, accuracy in results.items():

    print(
        "{:<25} {:.2f}%".format(
            name,
            accuracy * 100
        )
    )


# ============================================================
# FIND BEST ALGORITHM
# ============================================================

best_algorithm = max(
    results,
    key=results.get
)

best_accuracy = results[
    best_algorithm
]


print("\n============================================================")
print("BEST CLASSIFICATION ALGORITHM")
print("============================================================")

print("\nBest Algorithm:")
print(best_algorithm)

print("\nBest Accuracy:")
print(
    round(
        best_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

print("\n============================================================")
print("GENERATING ACCURACY COMPARISON GRAPH")
print("============================================================")

algorithm_names = list(
    results.keys()
)

accuracies = [
    accuracy * 100
    for accuracy in results.values()
]


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    algorithm_names,
    accuracies
)

plt.xlabel(
    "Classification Algorithm"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Comparison of Classification Algorithms"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=20
)

plt.grid(
    axis="y"
)


# Display accuracy values above bars

for i, accuracy in enumerate(accuracies):

    plt.text(
        i,
        accuracy + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )


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


# Display graph
plt.show()

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL SUMMARY")
print("============================================================")

for name, accuracy in results.items():

    print(
        name,
        ":",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )


print("\nBest performing algorithm:")
print(best_algorithm)

print(
    "with accuracy of",
    round(
        best_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# OUTPUT INFORMATION
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