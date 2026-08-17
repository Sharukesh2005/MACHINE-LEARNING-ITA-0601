import csv

# Read CSV file
with open('trainingdata.csv', 'r') as file:
    data = list(csv.reader(file))

# Remove header
header = data[0]
data = data[1:]

# Initialize hypothesis
hypothesis = None

print("Training Examples:\n")
for row in data:
    print(row)

# FIND-S Algorithm
for row in data:
    if row[-1].lower() == "yes":
        if hypothesis is None:
            hypothesis = row[:-1]
        else:
            for i in range(len(hypothesis)):
                if hypothesis[i] != row[i]:
                    hypothesis[i] = '?'

print("\nMost Specific Hypothesis:")
print(hypothesis)