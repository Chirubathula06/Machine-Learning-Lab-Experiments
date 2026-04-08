import csv

def candidate_elimination(data):
    num_attributes = len(data[0]) - 1

    # Initialize S and G
    S = ['0'] * num_attributes
    G = [['?'] * num_attributes]

    for instance in data:
        attributes = instance[:-1]
        label = instance[-1].lower()

        # POSITIVE EXAMPLE
        if label == 'yes':
            for i in range(num_attributes):
                if S[i] == '0':
                    S[i] = attributes[i]
                elif S[i] != attributes[i]:
                    S[i] = '?'

            # Remove hypotheses from G inconsistent with positive example
            G = [g for g in G if all(g[i] == '?' or g[i] == attributes[i]
                                     for i in range(num_attributes))]

        # NEGATIVE EXAMPLE
        else:
            new_G = []
            for g in G:
                for i in range(num_attributes):
                    if g[i] == '?':
                        if S[i] != attributes[i] and S[i]!='?':
                            new_hypothesis = g.copy()
                            new_hypothesis[i] = S[i]
                            if new_hypothesis not in new_G:
                                new_G.append(new_hypothesis)
            G = new_G

    return S, G


# Read CSV file
training_data = []
with open("training_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        training_data.append(row)

# Run Candidate Elimination
S_final, G_final = candidate_elimination(training_data)

print("Final Specific Boundary (S):")
print(S_final)

print("\nFinal General Boundary (G):")
for g in G_final:
    print(g)
