import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler

def svm_train_dual(data_train, label_train, regularisation_para_C):
    # Ensure labels are in {-1, 1} format
    label_train = np.where(label_train == 0, -1, 1)
    m, n = data_train.shape

    # Debug: print shapes and data
    print(f"Training data shape: {data_train.shape}, Labels shape: {label_train.shape}")

    # Compute Kernel matrix (for linear kernel)
    K = np.dot(data_train, data_train.T)
    
    # Debug: print kernel matrix and check diagonal for linearity
    print(f"Kernel matrix shape: {K.shape}, Diagonal values: {np.diag(K)[:5]}")

    # Define matrices for QP
    P = matrix(np.outer(label_train, label_train) * K)
    q = matrix(-np.ones(m))
    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * regularisation_para_C)))
    A = matrix(label_train, (1, m), 'd')
    b = matrix(0.0)

    # Debug: print out QP matrices to ensure correctness
    print(f"P matrix shape: {P.size}")
    print(f"q matrix shape: {q.size}")
    print(f"G matrix shape: {G.size}, h matrix shape: {h.size}")
    print(f"A matrix shape: {A.size}, b matrix value: {b}")

    # Solve QP problem
    solution = solvers.qp(P, q, G, h, A, b)
    alpha = np.ravel(solution['x'])

    # Debug: check the sum and range of alpha values
    print(f"Alpha values: {alpha[:10]}")
    print(f"Sum of alpha: {np.sum(alpha)}")
    print(f"Number of support vectors (alpha > 1e-6): {np.sum(alpha > 1e-6)}")

    # Threshold for support vectors
    support_vectors = alpha > 1e-6
    alpha_sv = alpha[support_vectors]
    sv = data_train[support_vectors]
    sv_y = label_train[support_vectors]

    # Debug: check support vectors
    print(f"Support vector count: {len(sv)}")

    # Compute w and b using support vectors
    w = np.dot((alpha_sv * sv_y).T, sv)

    # Debug: check weight vector w
    print(f"Weight vector w: {w[:5]}")
    print(f"Sum of weights (dual): {np.sum(w)}")

    # Bias calculation using margin support vectors (alpha_sv close to C)
    margin_sv = (alpha_sv < regularisation_para_C - 1e-5) & (alpha_sv > 1e-5)
    b = np.mean(sv_y[margin_sv] - np.dot(sv[margin_sv], w.T))

    # Debug: check bias term b
    print(f"Bias (dual): {b}")

    return {'alpha': alpha, 'w': w, 'b': b}

def svm_predict_dual(data_test, svm_model, data_train, label_train):
    alpha = svm_model['alpha']
    w = svm_model['w']
    b = svm_model['b']
    
    # Compute predictions
    predictions = np.sign(np.dot(data_test, w.T) + b)
    predictions = np.where(predictions == -1, 0, 1)

    # Debug: print first few predictions
    print(f"Predictions (first 10): {predictions[:10]}")
    
    return predictions

def svm_train_primal(data_train, label_train, regularisation_para_C):
    # Data dimensions
    N, d = data_train.shape

    # Convert labels from {0, 1} to {-1, 1}
    y = np.where(label_train == 0, -1, 1).reshape(-1, 1)

    # Prepare matrices for cvxopt
    P = matrix(np.block([[np.eye(d), np.zeros((d, N + 1))],
                         [np.zeros((N + 1, d + N + 1))]]))  # Quadratic term
    q = matrix(np.hstack([np.zeros(d + 1), regularisation_para_C / N * np.ones(N)]))  # Linear term
    G = matrix(np.block([[-np.diag(y.flatten()) @ data_train, -y, -np.eye(N)],
                         [np.zeros((N, d + 1)), -np.eye(N)]]))  # Constraints matrix
    h = matrix(np.hstack([-np.ones(N), np.zeros(N)]))  # Constraints vector

    # Solve the QP problem using cvxopt
    solution = solvers.qp(P, q, G, h)
    w_b_xi = np.array(solution['x']).flatten()

    # Extract weight vector (w), bias term (b), and slack variables (xi)
    w = w_b_xi[:d]
    b = w_b_xi[d]
    
    # Debug: primal weight and bias
    print(f"Primal weight vector (w): {w[:5]}")
    print(f"Primal bias (b): {b}")
    
    return (w, b)

def svm_predict_primal(data_test, label_test, svm_model):
    w, b = svm_model
    # Predict: sign(w.T * X + b)
    predictions = np.sign(np.dot(data_test, w) + b)
    # Convert {-1, 1} back to {0, 1}
    predictions = np.where(predictions == -1, 0, 1)

    # Debug: primal predictions
    print(f"Primal Predictions (first 10): {predictions[:10]}")
    
    # Calculate accuracy
    accuracy = np.mean(predictions == label_test)
    return accuracy

if __name__ == '__main__':
    # Load data
    train_data = pd.read_csv('train.csv', header=None)
    test_data = pd.read_csv('test.csv', header=None)

    # Separate features and labels
    X_train = train_data.iloc[:4000, 1:].values  # Features from 2nd column onwards
    y_train = train_data.iloc[:4000, 0].values   # Labels from 1st column
    X_val = train_data.iloc[4000:, 1:].values
    y_val = train_data.iloc[4000:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Primal model training and evaluation
    C = 100
    svm_model_primal = svm_train_primal(X_train, y_train, C)
    val_accuracy_primal = svm_predict_primal(X_val, y_val, svm_model_primal)
    test_accuracy_primal = svm_predict_primal(X_test, y_test, svm_model_primal)
    print(f"Primal Validation Accuracy: {val_accuracy_primal * 100:.2f}%")
    print(f"Primal Test Accuracy: {test_accuracy_primal * 100:.2f}%")

    # Dual model training and evaluation
    svm_model_dual = svm_train_dual(X_train, y_train, C)
    val_predictions_dual = svm_predict_dual(X_val, svm_model_dual, X_train, y_train)
    test_predictions_dual = svm_predict_dual(X_test, svm_model_dual, X_train, y_train)
    val_accuracy_dual = np.mean(val_predictions_dual == y_val)
    test_accuracy_dual = np.mean(test_predictions_dual == y_test)
    print(f"Dual Validation Accuracy: {val_accuracy_dual * 100:.2f}%")
    print(f"Dual Test Accuracy: {test_accuracy_dual * 100:.2f}%")

    # Compare weights and bias
    w_primal, b_primal = svm_model_primal
    print(f"Primal SVM: Sum of w = {np.sum(w_primal)}, b = {b_primal}")
    print(f"Dual SVM: Sum of alpha = {np.sum(svm_model_dual['alpha'])}, w = {np.sum(svm_model_dual['w'])}, b = {svm_model_dual['b']}")
