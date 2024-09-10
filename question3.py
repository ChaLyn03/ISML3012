from cvxopt import matrix, solvers
import numpy as np
import pandas as pd

# Load the training and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate the labels and features
X_train = train_data.iloc[:, 1:].values  # Features from column 1 onwards
y_train = train_data.iloc[:, 0].values   # Labels in the first column
X_test = test_data.iloc[:, 1:].values    # Features from column 1 onwards
y_test = test_data.iloc[:, 0].values     # Labels in the first column

# Convert labels from (0, 1) to (-1, 1) for SVM compatibility
y_train = np.where(y_train == 0, -1, y_train)
y_test = np.where(y_test == 0, -1, y_test)

# Define the SVM training function
def svm_train_dual(data_train, label_train, regularisation_para_C):
    m, n = data_train.shape

    # Compute the Kernel matrix
    K = np.dot(data_train, data_train.T)

    # Ensure all matrices are of type 'd' for double precision
    P = matrix(np.outer(label_train, label_train) * K, tc='d')
    q = matrix(-np.ones(m), tc='d')
    G = matrix(np.vstack((-np.eye(m), np.eye(m))), tc='d')
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * regularisation_para_C)), tc='d')
    A = matrix(label_train, (1, m), tc='d')
    b = matrix(0.0)

    # Solve the quadratic programming problem
    solution = solvers.qp(P, q, G, h, A, b)
    alpha_value = np.ravel(solution['x'])

    # Support vectors have non-zero Lagrange multipliers (alpha > 1e-5)
    support_vectors_idx = np.where(alpha_value > 1e-5)[0]
    support_alphas = alpha_value[support_vectors_idx]
    support_vectors = data_train[support_vectors_idx]
    support_labels = label_train[support_vectors_idx]

    # Compute the weight vector (w)
    w = np.sum(support_alphas[:, None] * support_labels[:, None] * support_vectors, axis=0)

    # Compute the intercept (bias term b)
    b = np.mean(
        [support_labels[i] - np.dot(w, support_vectors[i])
         for i in range(len(support_alphas))]
    )

    # Return the model, containing the alphas, support vectors, labels, bias, and weight vector
    return {
        'alpha': alpha_value,
        'support_vectors': support_vectors,
        'support_labels': support_labels,
        'b': b,
        'w': w,
        'support_alphas': support_alphas
    }

# Train the SVM
C = 100
svm_model = svm_train_dual(X_train, y_train, C)

# Extract the values
w = svm_model['w']
b = svm_model['b']
alpha_sum = np.sum(svm_model['alpha'])

# Output the results
print(f"Weight vector (w): {w}")
print(f"Bias term (b): {b}")
print(f"Sum of alphas: {alpha_sum}")
