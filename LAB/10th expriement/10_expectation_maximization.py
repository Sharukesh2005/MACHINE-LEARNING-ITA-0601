# ============================================================
# PROGRAM 10
# IMPLEMENTATION OF EXPECTATION-MAXIMIZATION ALGORITHM
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
output_file = output_folder / "10_expectation_maximization_output.txt"

# Graph output file
graph_file = output_folder / "10_expectation_maximization_graph.png"


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

from sklearn.mixture import GaussianMixture


# ============================================================
# CREATE DATASET
# ============================================================

np.random.seed(42)

# First group of data
data1 = np.random.normal(
    loc=5,
    scale=1,
    size=100
)

# Second group of data
data2 = np.random.normal(
    loc=10,
    scale=1,
    size=100
)

# Combine both groups
X = np.concatenate(
    [data1, data2]
).reshape(-1, 1)


# ============================================================
# DISPLAY PROGRAM INFORMATION
# ============================================================

print("============================================================")
print("EXPECTATION-MAXIMIZATION ALGORITHM")
print("============================================================")

print("\nDataset created successfully.")

print("\nNumber of data points:")
print(len(X))

print("\nNumber of clusters:")
print(2)


# ============================================================
# CREATE GAUSSIAN MIXTURE MODEL
# ============================================================

print("\n============================================================")
print("GAUSSIAN MIXTURE MODEL")
print("============================================================")

# Create GMM
gmm = GaussianMixture(
    n_components=2,
    random_state=42
)


# ============================================================
# TRAIN MODEL USING EM ALGORITHM
# ============================================================

print("\nTraining model using Expectation-Maximization...")

gmm.fit(X)

print("Training completed successfully.")


# ============================================================
# PREDICT CLUSTERS
# ============================================================

labels = gmm.predict(X)


# ============================================================
# DISPLAY MODEL PARAMETERS
# ============================================================

print("\n============================================================")
print("MODEL PARAMETERS")
print("============================================================")

# Means
print("\nCluster Means:")
print(gmm.means_.flatten())

# Covariance
print("\nCluster Covariances:")
print(gmm.covariances_.flatten())

# Weights
print("\nCluster Weights:")
print(gmm.weights_)


# ============================================================
# DISPLAY CLUSTER ASSIGNMENTS
# ============================================================

print("\n============================================================")
print("CLUSTER ASSIGNMENTS")
print("============================================================")

print("\nFirst 20 data points and their cluster labels:")

for i in range(20):
    print(
        "Data Point:",
        round(X[i][0], 3),
        "-> Cluster:",
        labels[i]
    )


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

probabilities = gmm.predict_proba(X)

print("\n============================================================")
print("CLUSTER PROBABILITIES")
print("============================================================")

print("\nFirst 10 data points:")
print(probabilities[:10])


# ============================================================
# CALCULATE LOG-LIKELIHOOD
# ============================================================

log_likelihood = gmm.score(X)

print("\nAverage Log-Likelihood:")
print(log_likelihood)


# ============================================================
# GRAPH
# ============================================================

print("\n============================================================")
print("GENERATING GRAPH")
print("============================================================")

plt.figure(figsize=(10, 6))

# Plot cluster 0
plt.scatter(
    X[labels == 0],
    np.zeros(sum(labels == 0)),
    label="Cluster 0"
)

# Plot cluster 1
plt.scatter(
    X[labels == 1],
    np.zeros(sum(labels == 1)),
    label="Cluster 1"
)

# Plot cluster means
means = gmm.means_.flatten()

for i, mean in enumerate(means):
    plt.axvline(
        mean,
        linestyle="--",
        label=f"Cluster {i} Mean"
    )

# Labels
plt.xlabel("Data Values")

plt.ylabel("Cluster")

plt.title(
    "Expectation-Maximization Algorithm using Gaussian Mixture Model"
)

plt.legend()

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