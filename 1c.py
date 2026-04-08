def find_s(dataset):
    hypothesis = ['Ø'] * (len(dataset[0]) - 1)

    for row in dataset:
        if row[-1] == "Yes":  # Positive example
            for i in range(len(hypothesis)):
                if hypothesis[i] == 'Ø':
                    hypothesis[i] = row[i]
                elif hypothesis[i] != row[i]:
                    hypothesis[i] = '?'
    return hypothesis
import random

def generate_dataset(n):
    data = []
    for _ in range(n):
        sky = random.choice(["Sunny", "Rainy", "Cloudy"])
        air = random.choice(["Warm", "Cold"])
        humidity = random.choice(["High", "Normal"])
        wind = random.choice(["Strong", "Weak"])
        water = random.choice(["Warm", "Cool"])
        forecast = random.choice(["Same", "Change"])

        # Target rule (hidden concept)
        enjoy = "Yes" if sky == "Sunny" and air == "Warm" else "No"

        data.append([sky, air, humidity, wind, water, forecast, enjoy])
    return data
data_10 = generate_dataset(10)
hyp_10 = find_s(data_10)

print("Dataset size: 10")
print("Final Hypothesis:", hyp_10)
data_100 = generate_dataset(100)
hyp_100 = find_s(data_100)

print("\nDataset size: 100")
print("Final Hypothesis:", hyp_100)
data_300 = generate_dataset(300)
hyp_300 = find_s(data_300)

print("\nDataset size: 300")
print("Final Hypothesis:", hyp_300)
