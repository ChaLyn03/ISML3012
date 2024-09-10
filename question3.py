import cvxpy as cvx
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

def svm_train_dual(data_train, label_train, regularisation_para_C):
    samples, features = data_train.shape
    label_train = label_train * 2 - 1  # Convert 0/1 to -1/1

    # Compute the Gram matrix (dot products between training samples)
    X = np.dot(data_train, data_train.T)
    
    # Set up the parameters for the quadratic program
    P = matrix(np.outer(label_train, label_train) * X)
    Q = matrix(-np.ones(samples))
    G = matrix(np.vstack((-np.eye(samples), np.eye(samples))))
    H = matrix(np.hstack((np.zeros(samples), np.ones(samples) * regularisation_para_C / samples)))
    A = matrix(label_train, (1, samples), 'd')
    B = matrix(0.0)

    # Solve the quadratic program using cvxopt
    solution = solvers.qp(P, Q, G, H, A, B)

    # Extract the Lagrange multipliers (alpha) from the solution
    alpha = np.ravel(solution['x'])

    # Return alpha (dual solution)
    return alpha

# Load the training data and test data
train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)
validation_data = train_data[4000:]
train_data = train_data[:4000]
regularisation_para_C = 100

# Separate features and labels for training data
label_train = train_data.iloc[:, 0].values  # Labels are in the first column
data_train = train_data.iloc[:, 1:].values  # Features are in the remaining columns

# Compute the dual solution
alpha = svm_train_dual(data_train, label_train, regularisation_para_C)

# Save the computed alpha to a file for future use
np.save('alpha.npy', alpha)

# Print the sum of alpha as requested
print("Sum of alpha:", np.sum(alpha))
