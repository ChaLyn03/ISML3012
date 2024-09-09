import numpy as np
import pandas as pd
import cvxopt

def svm_train_dual(data_train, label_train, regularisation_para_C):
    # Data dimensions
    N, d = data_train.shape
    
    # Convert labels from {0, 1} to {-1, 1}
    y = np.where(label_train == 0, -1, 1).reshape(-1, 1)

    # Compute the Gram matrix (Kernel matrix), in this case, linear kernel K(x_i, x_j) = x_i.T * x_j
    K = np.dot(data_train, data_train.T)
    
    # Prepare matrices for cvxopt
    P = cvxopt.matrix(np.outer(y, y) * K)  # Quadratic term: y_i y_j K(x_i, x_j)
    q = cvxopt.matrix(-np.ones((N, 1)))    # Linear term: -1 for all i
    G_std = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))  # Constraints for 0 <= alpha_i <= C
    h_std = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * regularisation_para_C)))  # Constraints vector
    A = cvxopt.matrix(y.T.astype(np.double))  # Equality constraint: sum(alpha_i * y_i) = 0
    b = cvxopt.matrix(0.0)

    # Solve the QP problem using cvxopt
    solution = cvxopt.solvers.qp(P, q, G_std, h_std, A, b)
    alphas = np.array(solution['x']).flatten()

    # Print the sum of alpha values
    sum_of_alphas = np.sum(alphas)
    print(f"Sum of Alphas: {sum_of_alphas}")

    # Get weight vector w and bias term b
    w = np.sum(alphas[:, None] * y * data_train, axis=0)
    
    # Support vectors have non-zero alphas
    sv_indices = np.where(alphas > 1e-5)[0]
    b = np.mean([y[i] - np.dot(w, data_train[i]) for i in sv_indices])
    
    return (w, b)

def svm_predict_dual(data_test, label_test, svm_model):
    w, b = svm_model
    # Predict: sign(w.T * X + b)
    predictions = np.sign(np.dot(data_test, w) + b)
    # Convert {-1, 1} back to {0, 1}
    predictions = np.where(predictions == -1, 0, 1)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == label_test)
    return accuracy

if __name__ == '__main__':
    # Load data without any assumptions about headers
    train_data = pd.read_csv('train.csv', header=None)
    test_data = pd.read_csv('test.csv', header=None)

    # Separate features and labels
    X_train = train_data.iloc[:4000, 1:].values  # Features from 2nd column onwards
    y_train = train_data.iloc[:4000, 0].values   # Labels from 1st column
    X_val = train_data.iloc[4000:, 1:].values
    y_val = train_data.iloc[4000:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    # Train the model
    C = 100
    svm_model_dual = svm_train_dual(X_train, y_train, C)

    # Validate and test the model
    val_accuracy_dual = svm_predict_dual(X_val, y_val, svm_model_dual)
    test_accuracy_dual = svm_predict_dual(X_test, y_test, svm_model_dual)

    print(f"Validation Accuracy (Dual): {val_accuracy_dual * 100:.2f}%")
    print(f"Test Accuracy (Dual): {test_accuracy_dual * 100:.2f}%")

    # Sum of weights for a quick check
    w_sum_dual = np.sum(svm_model_dual[0])
    print(f"Sum of w (Dual): {w_sum_dual}")
    
    # Print the bias term
    print(f"Bias (b, Dual): {svm_model_dual[1]}")