import pandas as pd
import numpy as np

# Load dataset
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Weak'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Initialize hypothesis
hypothesis = ['Ø'] * len(X[0])

# Find-S Algorithm
for i in range(len(X)):
    if y[i] == 'Yes':
        for j in range(len(hypothesis)):
            if hypothesis[j] == 'Ø':
                hypothesis[j] = X[i][j]
            elif hypothesis[j] != X[i][j]:
                hypothesis[j] = '?'

print("Final Hypothesis:", hypothesis)
