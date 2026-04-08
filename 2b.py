import pandas as pd
import numpy as np
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Temperature': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'Normal'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Weak'],
    'Water': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Forecast': ['Same', 'Same', 'Change', 'Same'],
    'PlaySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print(df)
def candidate_elimination(df):
    concepts = np.array(df.iloc[:, :-1])   # attributes
    target = np.array(df.iloc[:, -1])      # class labels

    n_attributes = concepts.shape[1]

    # Initialize S and G
    S = ['0'] * n_attributes
    G = [['?'] * n_attributes]

    for i, instance in enumerate(concepts):
        if target[i] == 'Yes':   # Positive example
            for j in range(n_attributes):
                if S[j] == '0':
                    S[j] = instance[j]
                elif S[j] != instance[j]:
                    S[j] = '?'

            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(n_attributes))]

        else:   # Negative example
            new_G = []
            for g in G:
                for j in range(n_attributes):
                    if g[j] == '?' and S[j] != instance[j]:
                        h = g.copy()
                        h[j] = S[j]
                        new_G.append(h)
            G = new_G

    return S, G
S, G = candidate_elimination(df)

print("\nFinal Specific Boundary (S):")
print(S)

print("\nFinal General Boundary (G):")
for hypothesis in G:
    print(hypothesis)
