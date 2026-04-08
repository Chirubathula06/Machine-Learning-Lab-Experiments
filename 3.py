import csv
import math
from collections import Counter, defaultdict

# Read CSV data
def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [row for row in reader]
    return headers, data

# Calculate entropy
def entropy(data):
    labels = [row[-1] for row in data]
    total = len(labels)
    counts = Counter(labels)

    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

# Information Gain
def information_gain(data, attr_index):
    total_entropy = entropy(data)
    values = defaultdict(list)

    for row in data:
        values[row[attr_index]].append(row)

    weighted_entropy = 0
    total = len(data)

    for subset in values.values():
        weighted_entropy += (len(subset) / total) * entropy(subset)

    return total_entropy - weighted_entropy

# ID3 Algorithm
def id3(data, headers):
    labels = [row[-1] for row in data]

    # If all examples have same label
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    # If no attributes left
    if len(headers) == 1:
        return Counter(labels).most_common(1)[0][0]

    # Choose best attribute
    gains = [information_gain(data, i) for i in range(len(headers) - 1)]
    best_attr = gains.index(max(gains))

    tree = {headers[best_attr]: {}}

    attr_values = set(row[best_attr] for row in data)

    for value in attr_values:
        subset = [row[:best_attr] + row[best_attr+1:] 
                  for row in data if row[best_attr] == value]

        new_headers = headers[:best_attr] + headers[best_attr+1:]
        tree[headers[best_attr]][value] = id3(subset, new_headers)

    return tree

# Classification of new sample
def classify(tree, headers, sample):
    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))
    attr_index = headers.index(attribute)
    value = sample[attr_index]

    return classify(tree[attribute][value], headers, sample)


# ---------------- MAIN ----------------
headers, data = load_data("play_tennis.csv")
decision_tree = id3(data, headers[:-1])

print("Decision Tree:")
print(decision_tree)

# New sample classification
new_sample = ['Sunny', 'Cool', 'High', 'Strong']
result = classify(decision_tree, headers[:-1], new_sample)

print("\nNew Sample:", new_sample)
print("Classification Result:", result)
