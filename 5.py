import csv
from collections import defaultdict

# Load CSV
def load_csv(filename):
    with open(filename, 'r') as f:
        data = list(csv.reader(f))
    return data[1:]  # skip header

# Train Naive Bayes
def train(data):
    class_count = defaultdict(int)
    feature_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for row in data:
        label = row[-1]
        class_count[label] += 1
        for i in range(len(row)-1):
            feature_count[i][row[i]][label] += 1

    return class_count, feature_count

# Predict class
def predict(row, class_count, feature_count, total):
    probs = {}
    for cls in class_count:
        prob = class_count[cls] / total
        for i in range(len(row)-1):
            count = feature_count[i][row[i]][cls]
            prob *= (count + 1) / (class_count[cls] + 2)  # Laplace smoothing
        probs[cls] = prob
    return max(probs, key=probs.get)

# Calculate accuracy
def accuracy(test, class_count, feature_count):
    correct = 0
    total = len(test)
    total_train = sum(class_count.values())

    for row in test:
        actual = row[-1]
        pred = predict(row, class_count, feature_count, total_train)
        if pred == actual:
            correct += 1

    return correct / total

# Main
train_data = load_csv('train.csv')
test_data = load_csv('test.csv')

class_count, feature_count = train(train_data)
acc = accuracy(test_data, class_count, feature_count)

print("Accuracy of Naive Bayes Classifier:", acc)
