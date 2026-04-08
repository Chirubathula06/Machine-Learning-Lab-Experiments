import pandas as pd
import random
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Possible categorical values
outlook = ['Sunny', 'Overcast', 'Rain']
temp = ['Hot', 'Mild', 'Cool']
humidity = ['High', 'Normal']
windy = ['True', 'False']
play = ['Yes', 'No']

# Encode categories to numbers
encode_map = {
    'Sunny':0, 'Overcast':1, 'Rain':2,
    'Hot':0, 'Mild':1, 'Cool':2,
    'High':0, 'Normal':1,
    'True':1, 'False':0,
    'Yes':1, 'No':0
}

# Function to create dataset
def generate_dataset(size):
    data = []
    for _ in range(size):
        row = [
            random.choice(outlook),
            random.choice(temp),
            random.choice(humidity),
            random.choice(windy),
            random.choice(play)
        ]
        data.append(row)

    df = pd.DataFrame(data, columns=['Outlook','Temp','Humidity','Windy','Play'])

    # Encode categorical values
    for col in df.columns:
        df[col] = df[col].map(encode_map)

    return df

# Function to train and test
def run_naive_bayes(df):
    X = df[['Outlook','Temp','Humidity','Windy']]
    y = df['Play']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model = CategoricalNB()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return acc

# Run for 10, 100, 300 records
for size in [10, 100, 300]:
    dataset = generate_dataset(size)
    accuracy = run_naive_bayes(dataset)
    print(f"Dataset Size: {size} -> Accuracy: {accuracy:.2f}")
