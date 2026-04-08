# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create k-NN classifier (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict the test data
predictions = knn.predict(X_test)

# Print correct and wrong predictions
for i in range(len(predictions)):
    print("Actual:", y_test[i], " Predicted:", predictions[i])
    
    if y_test[i] == predictions[i]:
        print("Correct Prediction")
    else:
        print("Wrong Prediction")
