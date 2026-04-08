import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text
def train_and_test_id3(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    encoders = {}
    for col in X.columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col])

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)

    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)

    # PRINT TREE RULES (TEXT FORM)
    print("\nDecision Tree Rules:")
    print(export_text(model, feature_names=list(X.columns)))

    # TEST ON SAME DATA (for lab demo)
    predictions = model.predict(X)
    predictions = y_encoder.inverse_transform(predictions)

    print("\nPredictions:")
    print(predictions)

    print("\nActual Output:")
    print(df[target_col].values)
print("\n--- Email Spam Detection ---")

data = {
    'Keywords': ['Yes', 'Yes', 'No', 'No', 'Yes'],
    'SenderDomain': ['Unknown', 'Unknown', 'Known', 'Known', 'Unknown'],
    'Class': ['Spam', 'Spam', 'NotSpam', 'NotSpam', 'Spam']
}

df = pd.DataFrame(data)
train_and_test_id3(df, 'Class')
print("\n--- Medical Diagnosis ---")

data = {
    'Symptoms': ['Severe', 'Mild', 'None', 'Severe', 'Mild'],
    'TestResult': ['Positive', 'Positive', 'Negative', 'Positive', 'Negative'],
    'Disease': ['Present', 'Present', 'Absent', 'Present', 'Absent']
}

df = pd.DataFrame(data)
train_and_test_id3(df, 'Disease')
print("\n--- Student Performance ---")

data = {
    'Marks': ['High', 'Medium', 'Low', 'High', 'Medium'],
    'Attendance': ['Good', 'Average', 'Poor', 'Good', 'Poor'],
    'Internship': ['Yes', 'Yes', 'No', 'Yes', 'No'],
    'Result': ['Pass', 'Pass', 'Fail', 'Pass', 'Fail']
}

df = pd.DataFrame(data)
train_and_test_id3(df, 'Result')
print("\n--- Loan Approval System ---")

data = {
    'Income': ['High', 'Medium', 'Low', 'High', 'Medium'],
    'CreditScore': ['Good', 'Good', 'Poor', 'Good', 'Average'],
    'LoanStatus': ['Approved', 'Approved', 'Rejected', 'Approved', 'Rejected']
}

df = pd.DataFrame(data)
train_and_test_id3(df, 'LoanStatus')
print("\n--- Fraud Detection ---")

data = {
    'Amount': ['High', 'Low', 'Low', 'High', 'High'],
    'Frequency': ['High', 'Low', 'Low', 'High', 'Low'],
    'Transaction': ['Fraud', 'Normal', 'Normal', 'Fraud', 'Fraud']
}

df = pd.DataFrame(data)
train_and_test_id3(df, 'Transaction')
