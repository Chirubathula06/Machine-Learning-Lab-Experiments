import pandas as pd
import numpy as np
import random
def generate_dataset(n):
    Sky = ['Sunny', 'Rainy']
    Temp = ['Warm', 'Cold']
    Humidity = ['High', 'Normal']
    Wind = ['Strong', 'Weak']
    Water = ['Warm', 'Cold']
    Forecast = ['Same', 'Change']

    data = []
    for _ in range(n):
        instance = [
            random.choice(Sky),
            random.choice(Temp),
            random.choice(Humidity),
            random.choice(Wind),
            random.choice(Water),
            random.choice(Forecast)
        ]

        # Target concept rule (hidden)
        if instance[0] == 'Sunny' and instance[1] == 'Warm':
            label = 'Yes'
        else:
            label = 'No'

        data.append(instance + [label])

    columns = ['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast', 'Play']
    return pd.DataFrame(data, columns=columns)
def candidate_elimination(df):
    concepts = np.array(df.iloc[:, :-1])
    target = np.array(df.iloc[:, -1])

    n_attr = concepts.shape[1]

    S = ['0'] * n_attr
    G = [['?'] * n_attr]

    for i, instance in enumerate(concepts):
        if target[i] == 'Yes':
            for j in range(n_attr):
                if S[j] == '0':
                    S[j] = instance[j]
                elif S[j] != instance[j]:
                    S[j] = '?'

            G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(n_attr))]

        else:  # Negative example
            new_G = []
            for g in G:
                for j in range(n_attr):
                    if g[j] == '?' and S[j] != instance[j]:
                        h = g.copy()
                        h[j] = S[j]
                        new_G.append(h)
            G = new_G

    return S, G
df_10 = generate_dataset(10)
S10, G10 = candidate_elimination(df_10)

print("DATASET SIZE: 10")
print("Final S:", S10)
print("Final G:", G10)
print("Version Space Size:", len(G10))
df_100 = generate_dataset(100)
S100, G100 = candidate_elimination(df_100)

print("\nDATASET SIZE: 100")
print("Final S:", S100)
print("Final G:", G100)
print("Version Space Size:", len(G100))
df_300 = generate_dataset(300)
S300, G300 = candidate_elimination(df_300)

print("\nDATASET SIZE: 300")
print("Final S:", S300)
print("Final G:", G300)
print("Version Space Size:", len(G300))
