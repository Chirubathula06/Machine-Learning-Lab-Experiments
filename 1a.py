def find_s(dataset):
    hypothesis = ['Ø'] * (len(dataset[0]) - 1)

    for row in dataset:
        if row[-1] == "Yes":        # Positive example
            for i in range(len(hypothesis)):
                if hypothesis[i] == 'Ø':
                    hypothesis[i] = row[i]
                elif hypothesis[i] != row[i]:
                    hypothesis[i] = '?'
    return hypothesis
email_spam = [
    ["Yes", "Yes", "Yes", "Yes", "Yes"],
    ["Yes", "No",  "Yes", "Yes", "Yes"],
    ["Yes", "Yes", "No",  "Yes", "Yes"],
    ["No",  "No",  "No",  "No",  "No"],
    ["No",  "Yes", "Yes", "No",  "No"]
]

print("Email Spam Hypothesis:")
print(find_s(email_spam))
medical = [
    ["Yes", "Yes", "Yes", "Yes", "Yes"],
    ["Yes", "Yes", "No",  "Yes", "Yes"],
    ["Yes", "Yes", "Yes", "No",  "Yes"],
    ["No",  "Yes", "Yes", "Yes", "No"],
    ["Yes", "No",  "Yes", "Yes", "No"]
]

print("\nMedical Diagnosis Hypothesis:")
print(find_s(medical))
product = [
    ["Electronics", "Low", "BrandA", "Yes", "Yes"],
    ["Electronics", "Low", "BrandB", "No",  "Yes"],
    ["Electronics", "Low", "BrandA", "No",  "Yes"],
    ["Clothing",    "High","BrandA", "Yes", "No"],
    ["Electronics", "High","BrandA", "No",  "No"]
]

print("\nProduct Recommendation Hypothesis:")
print(find_s(product))
weather = [
    ["Low",  "High",   "Weak",  "No",  "Yes"],
    ["Low",  "Medium", "Weak",  "No",  "Yes"],
    ["Low",  "High",   "Strong","No",  "Yes"],
    ["High", "Low",    "Strong","Yes", "No"],
    ["High", "Medium", "Weak",  "Yes", "No"]
]

print("\nWeather Prediction Hypothesis:")
print(find_s(weather))
fraud = [
    ["High", "Yes", "Yes", "Night", "Yes"],
    ["High", "Yes", "No",  "Night", "Yes"],
    ["High", "Yes", "Yes", "Night", "Yes"],
    ["Low",  "No",  "No",  "Day",   "No"],
    ["Medium","No", "Yes", "Day",   "No"]
]

print("\nFraud Detection Hypothesis:")
print(find_s(fraud))

