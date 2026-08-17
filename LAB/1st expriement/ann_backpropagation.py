# ANN using Backpropagation Algorithm
# Dataset: Breast Cancer Wisconsin Dataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------------------------
# Load Dataset
# ------------------------------------

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

print("Dataset Loaded Successfully")
print("Features Shape :", X.shape)
print("Target Shape :", y.shape)

# ------------------------------------
# Split Dataset
# ------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------
# Feature Scaling
# ------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------
# Build ANN
# ------------------------------------

model = Sequential()

model.add(Dense(
    units=16,
    activation='relu',
    input_shape=(X_train.shape[1],)
))

model.add(Dense(
    units=8,
    activation='relu'
))

model.add(Dense(
    units=1,
    activation='sigmoid'
))

# ------------------------------------
# Compile Model
# ------------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

print("\nTraining Completed Successfully!")

# ------------------------------------
# Prediction
# ------------------------------------

print("\nTesting the Model...\n")

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("      ARTIFICIAL NEURAL NETWORK USING BACKPROPAGATION")
print("=" * 60)

print(f"\nDataset              : Breast Cancer Wisconsin")
print(f"Training Samples     : {len(X_train)}")
print(f"Testing Samples      : {len(X_test)}")
print(f"Input Features       : {X.shape[1]}")
print(f"Hidden Neurons       : 16, 8")
print(f"Epochs               : 50")
print(f"Batch Size           : 16")
print(f"\nModel Accuracy       : {accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("=" * 60)
print("First 10 Predictions")
print("=" * 60)

for i in range(10):
    print(f"Sample {i+1}")
    print(f"Actual Value    : {y_test[i]}")
    print(f"Predicted Value : {int(y_pred[i])}")
    print("-"*30)

print("\nProgram Executed Successfully!")

# ------------------------------------
# Accuracy Graph
# ------------------------------------

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("Accuracy_Graph.png")
plt.show()

# ------------------------------------
# Loss Graph
# ------------------------------------

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("Loss_Graph.png")
plt.show()