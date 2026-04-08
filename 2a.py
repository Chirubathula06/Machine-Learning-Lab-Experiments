def candidate_elimination(concepts, target):
    # Initialize S and G
    n_attributes = len(concepts[0])
    S = ['0'] * n_attributes
    G = [['?'] * n_attributes]

    for i, instance in enumerate(concepts):
        if target[i] == 'Yes':
            # Update S
            for j in range(n_attributes):
                if S[j] == '0':
                    S[j] = instance[j]
                elif S[j] != instance[j]:
                    S[j] = '?'

            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(n_attributes))]

        else:  # Negative example
            new_G = []
            for g in G:
                for j in range(n_attributes):
                    if g[j] == '?' and S[j] != instance[j]:
                        new_hypothesis = g.copy()
                        new_hypothesis[j] = S[j]
                        new_G.append(new_hypothesis)
            G = new_G

    return S, G
concepts = [
    ['Yes', 'No', 'Yes'],
    ['Yes', 'No', 'No'],
    ['No', 'Yes', 'No'],
    ['Yes', 'No', 'Yes']
]

target = ['Yes', 'Yes', 'No', 'Yes']

S, G = candidate_elimination(concepts, target)

print("Email Spam Filtering")
print("Final S:", S)
print("Final G:", G)
concepts = [
    ['High', 'Yes', 'Positive'],
    ['High', 'Yes', 'Positive'],
    ['Normal', 'No', 'Negative'],
    ['High', 'Yes', 'Positive']
]

target = ['Yes', 'Yes', 'No', 'Yes']

S, G = candidate_elimination(concepts, target)

print("\nMedical Diagnosis")
print("Final S:", S)
print("Final G:", G)
concepts = [
    ['Electronics', 'High', 'Yes'],
    ['Electronics', 'High', 'Yes'],
    ['Clothing', 'Low', 'No'],
    ['Electronics', 'High', 'Yes']
]

target = ['Yes', 'Yes', 'No', 'Yes']

S, G = candidate_elimination(concepts, target)

print("\nProduct Recommendation")
print("Final S:", S)
print("Final G:", G)
concepts = [
    ['Hot', 'High', 'Weak'],
    ['Hot', 'High', 'Weak'],
    ['Cold', 'Normal', 'Strong'],
    ['Hot', 'High', 'Weak']
]

target = ['Yes', 'Yes', 'No', 'Yes']

S, G = candidate_elimination(concepts, target)

print("\nWeather Prediction")
print("Final S:", S)
print("Final G:", G)
concepts = [
    ['High', 'Abroad', 'Unknown'],
    ['High', 'Abroad', 'Unknown'],
    ['Low', 'Local', 'Known'],
    ['High', 'Abroad', 'Unknown']
]

target = ['Yes', 'Yes', 'No', 'Yes']

S, G = candidate_elimination(concepts, target)

print("\nFraud Detection")
print("Final S:", S)
print("Final G:", G)
