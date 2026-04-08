import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
def generate_dataset(n):
    Marks = ['High', 'Medium', 'Low']
    Attendance = ['Good', 'Average', 'Poor']
    Internship = ['Yes', 'No']

    data = []

    for _ in range(n):
        m = random.choice(Marks)
        a = random.choice(Attendance)
        i = random.choice(Internship)

        if m == 'High' and a == 'Good':
            result = 'Pass'
        else:
            result = 'Fail'

        data.append([m, a, i, result])

    return pd.DataFrame(data, columns=['Marks', 'Attendance', 'Internship', 'Result'])
def train_evaluate_id3(df):
    X = df.drop('Result', axis=1)
    y = df['Result']

    encoders = {}
    for col in X.columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col])

    y_enc = LabelEncoder()
    y = y_enc.fit_transform(y)

    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    errors = len(y) - sum(y == predictions)

    print("\nDecision Tree Rules:")
    print(export_text(model, feature_names=list(X.columns)))

    print("Accuracy:", accuracy)
    print("Number of Errors:", errors)
df_10 = generate_dataset(10)
print("\n========== DATASET SIZE: 10 ==========")
train_evaluate_id3(df_10)
df_100 = generate_dataset(100)
print("\n========== DATASET SIZE: 100 ==========")
train_evaluate_id3(df_100)
df_300 = generate_dataset(300)
print("\n========== DATASET SIZE: 300 ==========")
train_evaluate_id3(df_300)
