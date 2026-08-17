import csv

# Function to load CSV data
def load_csv(filename):
    with open(filename, 'r') as f:
        data = list(csv.reader(f))
    return data

# Candidate Elimination Algorithm
def candidate_elimination(data):
    concepts = [row[:-1] for row in data]
    target = [row[-1] for row in data]

    # Initialize Specific and General Hypotheses
    S = concepts[0].copy()
    G = [["?" for _ in range(len(S))]]

    print("Initial Specific Hypothesis (S):", S)
    print("Initial General Hypothesis (G):", G)

    for i, h in enumerate(concepts):
        if target[i].lower() == "yes":   # Positive Example
            for x in range(len(S)):
                if h[x] != S[x]:
                    S[x] = "?"
            G = [g for g in G if all(g[j] == "?" or g[j] == S[j] for j in range(len(S)))]

        elif target[i].lower() == "no":  # Negative Example
            new_G = []
            for g in G:
                for x in range(len(S)):
                    if S[x] != "?":
                        if h[x] != S[x]:
                            new_h = g.copy()
                            new_h[x] = S[x]
                            new_G.append(new_h)
                    else:
                        if h[x] != "?":
                            new_h = g.copy()
                            new_h[x] = "?"
                            new_G.append(new_h)
            G = new_G

        print("\nAfter Example", i + 1)
        print("S =", S)
        print("G =", G)

    print("\nFinal Specific Hypothesis:")
    print(S)

    print("\nFinal General Hypothesis:")
    for g in G:
        print(g)

# Main Program
filename = "trainingdata.csv"
data = load_csv(filename)

# Remove header row
data = data[1:]

candidate_elimination(data)