import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load dataset
data = pd.read_csv("heart_disease.csv")

# Select important columns
data = data[['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level',
             'Smoking', 'Diabetes', 'Heart Disease Status']]

# Convert target to binary
data['Heart Disease Status'] = data['Heart Disease Status'].map({'Yes': 1, 'No': 0})

# Convert categorical columns
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Smoking'] = data['Smoking'].map({'Yes': 1, 'No': 0})
data['Diabetes'] = data['Diabetes'].map({'Yes': 1, 'No': 0})

# Discretize numeric columns
data[['Age', 'Blood Pressure', 'Cholesterol Level']] = data[['Age', 'Blood Pressure', 'Cholesterol Level']].apply(
    lambda x: pd.cut(x, bins=3, labels=False)
)

# Rename target
data = data.rename(columns={'Heart Disease Status': 'target'})

# Define model
model = DiscreteBayesianNetwork([
    ('Age', 'target'),
    ('Gender', 'target'),
    ('Blood Pressure', 'target'),
    ('Cholesterol Level', 'target'),
    ('Smoking', 'target'),
    ('Diabetes', 'target')
])

# Train
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inference
infer = VariableElimination(model)

# Prediction example
result = infer.query(variables=['target'], evidence={
    'Age': 1,
    'Gender': 1,
    'Blood Pressure': 1,
    'Cholesterol Level': 1,
    'Smoking': 1,
    'Diabetes': 0
})
if result.values[1] > result.values[0]:
    print("Prediction: Patient likely has Heart Disease")
else:
    print("Prediction: Patient likely does NOT have Heart Disease")

print(result)


