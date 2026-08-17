import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Create Output folder
os.makedirs("Output", exist_ok=True)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=iris.target_names)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.savefig("Output/confusion_matrix.png")
plt.show()

# Save output to text file
with open("Output/output.txt", "w") as f:
    f.write("========== NAIVE BAYES ==========\n\n")
    f.write("Actual Values:\n")
    f.write(str(y_test))
    f.write("\n\nPredicted Values:\n")
    f.write(str(y_pred))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nAccuracy: {:.2f}%".format(accuracy * 100))

# Save accuracy separately
with open("Output/accuracy.txt", "w") as f:
    f.write("Accuracy = {:.2f}%".format(accuracy * 100))

print("\nAccuracy :", round(accuracy * 100, 2), "%")
print("\nOutput saved inside Output folder.")
print("Program Executed Successfully!")