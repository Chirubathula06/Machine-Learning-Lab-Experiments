import csv

def find_s_algorithm(training_data):
    hypothesis = None

    for instance in training_data:
        if instance[-1].lower() == "yes":  # Positive example
            if hypothesis is None:
                hypothesis = instance[:-1]  # Initialize hypothesis
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != instance[i]:
                        hypothesis[i] = '?'
    return hypothesis


# Read CSV file
training_data = [] 
with open("training_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        training_data.append(row)

# Apply FIND-S
final_hypothesis = find_s_algorithm(training_data)

print("Final Most Specific Hypothesis:")
print(final_hypothesis)
