# ============================================================
# PROGRAM 12
# IRIS FLOWER CLASSIFICATION USING K-NEAREST NEIGHBOUR (KNN)
# ============================================================

import sys
from pathlib import Path

# ============================================================
# OUTPUT FOLDER
# ============================================================

current_folder = Path(__file__).parent

output_folder = current_folder / "OUTPUT"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "12_iris_knn_output.txt"
graph_file = output_folder / "12_iris_knn_graph.png"


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


file = open(output_file, "w", encoding="utf-8")

sys.stdout = Tee(sys.__stdout__, file)


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# LOAD IRIS DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

target_names = iris.target_names


print("============================================================")
print("IRIS FLOWER CLASSIFICATION USING KNN")
print("============================================================")

print("\nDataset loaded successfully.")

print("Number of samples:", len(X))
print("Number of features:", X.shape[1])

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


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# GET K VALUE
# ============================================================

print("\n============================================================")
print("KNN CLASSIFIER")
print("============================================================")

k = int(input("\nEnter the value of K: "))

if k <= 0:
    print("ERROR: K must be greater than 0.")
    file.close()
    sys.exit()

if k > len(X_train):
    print("ERROR: K is too large.")
    file.close()
    sys.exit()


print("Selected K:", k)


# ============================================================
# CREATE KNN MODEL
# ============================================================

model = KNeighborsClassifier(
    n_neighbors=k
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining KNN model...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training completed successfully.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# EVALUATION
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
print(round(accuracy * 100, 2), "%")

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
# GRAPH
# ============================================================

print("\n============================================================")
print("GENERATING GRAPH")
print("============================================================")

plt.figure(figsize=(10, 6))

sample_numbers = range(1, len(y_test) + 1)

plt.scatter(
    sample_numbers,
    y_test,
    label="Actual Class"
)

plt.scatter(
    sample_numbers,
    y_pred,
    marker="x",
    label="Predicted Class"
)

plt.xlabel("Test Sample Number")

plt.ylabel("Flower Class")

plt.title("Iris Flower Classification using KNN")

plt.yticks(
    [0, 1, 2],
    target_names
)

plt.legend()

plt.grid(True)

# Save graph
plt.savefig(
    graph_file,
    dpi=300,
    bbox_inches="tight"
)

print("Graph saved to:")
print(graph_file)

# Display graph
plt.show()

plt.close()


# ============================================================
# NEW FLOWER PREDICTION
# ============================================================

print("\n============================================================")
print("NEW IRIS FLOWER PREDICTION")
print("============================================================")

sepal_length = float(
    input("Enter sepal length (cm): ")
)

sepal_width = float(
    input("Enter sepal width (cm): ")
)

petal_length = float(
    input("Enter petal length (cm): ")
)

petal_width = float(
    input("Enter petal width (cm): ")
)


new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]


new_flower_scaled = scaler.transform(
    new_flower
)


prediction = model.predict(
    new_flower_scaled
)


predicted_flower = target_names[
    prediction[0]
]


print("\nPredicted Flower:")
print(predicted_flower)


# ============================================================
# PROBABILITY
# ============================================================

probabilities = model.predict_proba(
    new_flower_scaled
)

print("\nPrediction Probabilities:")

for i, probability in enumerate(probabilities[0]):

    print(
        target_names[i],
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
# CLOSE FILE
# ============================================================

file.close()