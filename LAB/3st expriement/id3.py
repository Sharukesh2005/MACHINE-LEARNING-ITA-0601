import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# Load dataset
data = pd.read_csv("play_tennis.csv")

# Encode categorical values
le = LabelEncoder()

for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Features and Target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Build Decision Tree using ID3 (Entropy)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# Display Decision Tree
print("\nDecision Tree:\n")
print(export_text(model, feature_names=list(X.columns)))

# Classify a New Sample
# Example: Outlook=Sunny, Temperature=Cool, Humidity=High, Wind=Strong

new_sample = [[2, 0, 0, 0]]

prediction = model.predict(new_sample)

if prediction[0] == 1:
    print("\nPrediction: Play Tennis = Yes")
else:
    print("\nPrediction: Play Tennis = No")