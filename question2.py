import numpy as np
import pandas as pd
import cvxopt

def svm_train_primal(data_train, label_train, regularisation_para_C):
    # Data dimensions
    N, d = data_train.shape

    # Convert labels from {0, 1} to {-1, 1}
    y = np.where(label_train == 0, -1, 1).reshape(-1, 1)

    # Prepare matrices for cvxopt
    P = cvxopt.matrix(np.block([[np.eye(d), np.zeros((d, N + 1))],
                                [np.zeros((N + 1, d + N + 1))]]))  # Quadratic term
    q = cvxopt.matrix(np.hstack([np.zeros(d + 1), regularisation_para_C / N * np.ones(N)]))  # Linear term
    G = cvxopt.matrix(np.block([[-np.diag(y.flatten()) @ data_train, -y, -np.eye(N)],
                                [np.zeros((N, d + 1)), -np.eye(N)]]))  # Constraints matrix
    h = cvxopt.matrix(np.hstack([-np.ones(N), np.zeros(N)]))  # Constraints vector

    # Solve the QP problem using cvxopt
    solution = cvxopt.solvers.qp(P, q, G, h)
    w_b_xi = np.array(solution['x']).flatten()

    # Extract weight vector (w), bias term (b), and slack variables (xi)
    w = w_b_xi[:d]
    b = w_b_xi[d]
    return (w, b)

def svm_predict_primal(data_test, label_test, svm_model):
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
    svm_model = svm_train_primal(X_train, y_train, C)

    # Validate and test the model
    val_accuracy = svm_predict_primal(X_val, y_val, svm_model)
    test_accuracy = svm_predict_primal(X_test, y_test, svm_model)

    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Sum of weights for a quick check
    w_sum = np.sum(svm_model[0])
    print(f"Sum of w: {w_sum}")
    
    # Print the bias term
    print(f"Bias (b): {svm_model[1]}")

    # Extract w and b from primal
    w_primal, b_primal = svm_model

    # Print values for comparison
    print(f"Primal SVM: Sum of w = {np.sum(w_primal)}, b = {b_primal}")