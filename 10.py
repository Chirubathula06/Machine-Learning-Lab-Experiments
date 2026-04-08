import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([1,2,3,4,5,6,7,8])
Y = np.array([1.5,1.7,3.2,3.8,5.1,6.2,6.8,8.0])

# Bandwidth parameter
tau = 0.5

# Function for Locally Weighted Regression
def lwlr(x0, X, Y, tau):
    weights = np.exp(-(X - x0)**2 / (2 * tau**2))
    W = np.diag(weights)
    
    X_mat = np.vstack([np.ones(len(X)), X]).T
    theta = np.linalg.inv(X_mat.T @ W @ X_mat) @ X_mat.T @ W @ Y  
    
    return theta[0] + theta[1] * x0

# Predict values
X_test = np.linspace(1,8,100)
Y_pred = [lwlr(x, X, Y, tau) for x in X_test]

# Plot graph
plt.scatter(X, Y, color='red', label='Data Points')
plt.plot(X_test, Y_pred, color='blue', label='LWR Fit')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Locally Weighted Regression")
plt.legend()
plt.show()
