import numpy as np
import pandas as pd  # Import pandas
import cvxopt

def svm_train_dual(data_train, label_train, regularisation_para_C):
    # Data dimensions
    N, d = data_train.shape

    # Convert labels from {0, 1} to {-1, 1}
    y = np.where(label_train == 0, -1, 1).astype(float).reshape(-1, 1)

    # Compute the kernel matrix (linear kernel, which is just the dot product)
    K = np.dot(data_train, data_train.T)

    # Prepare matrices for cvxopt
    P = cvxopt.matrix(np.outer(y, y) * K)  # P = (y_i y_j K(x_i, x_j))
    q = cvxopt.matrix(-np.ones((N, 1)))  # q = -1 (because of maximization)
    
    # Inequality constraints: 0 <= alpha_i <= C/N
    G_std = np.diag(np.ones(N) * -1)
    G_slack = np.diag(np.ones(N))
    G = cvxopt.matrix(np.vstack((G_std, G_slack)))

    h_std = np.zeros(N)
    h_slack = np.ones(N) * regularisation_para_C / N
    h = cvxopt.matrix(np.hstack((h_std, h_slack)))

    # Equality constraint: sum(alpha_i * y_i) = 0
    A = cvxopt.matrix(y.T, (1, N), 'd')
    b = cvxopt.matrix(0.0)

    # Solve the QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x']).flatten()

    return alphas, y, K

def find_support_vectors_dual(alphas, epsilon=1e-5):
    # Support vectors are those where alpha is greater than a small threshold
    svs = np.where(alphas > epsilon)[0]
    return svs

def calculate_w_b(alphas, y, X, C, support_indices):
    # Support vector alphas and labels
    support_alphas = alphas[support_indices]  # Shape: (392,)
    support_y = y[support_indices].flatten()  # Shape: (392,)
    support_vectors = X[support_indices]  # Shape: (392, 200)

    # Calculate the weight vector w
    w = np.sum(support_alphas[:, None] * support_y[:, None] * support_vectors, axis=0)  # Shape: (200,)

    # Calculate the bias term b by averaging over support vectors
    b = np.mean(support_y - np.dot(support_vectors, w))  # Shape: scalar

    return w, b

# Function to compute primal from dual
def compute_primal_from_dual(data_train, label_train, alpha_opt):
    # Adjust labels from {0, 1} to {-1, 1}
    label_train = label_train * 2 - 1
    label_train = label_train.reshape(-1, 1)

    # Reshape alpha_opt to ensure it can broadcast correctly
    alpha_opt = alpha_opt.reshape(-1, 1)

    # Compute the primal weight vector w_star
    w_star = np.sum(alpha_opt * label_train * data_train, axis=0)

    # Get the indices of the support vectors
    support_vector_indices = np.where(alpha_opt.flatten() > 1e-5)[0]

    if len(support_vector_indices) == 0:
        raise ValueError("No support vectors found. Adjust the threshold or check the dual solution.")

    # Compute the primal bias term b_star
    b_star = np.mean(label_train[support_vector_indices] - np.dot(data_train[support_vector_indices], w_star))

    # Optionally sum the weights if desired (though not necessary)
    w_star_sum = np.sum(w_star)

    return w_star_sum, b_star


if __name__ == '__main__':
    # Load data
    train_data = pd.read_csv('train.csv', header=None)

    # Separate features and labels
    X_train = train_data.iloc[:4000, 1:].values  # Features from 2nd column onwards
    y_train = train_data.iloc[:4000, 0].values   # Labels from 1st column

    # Train the model
    C = 100  # Use the specified C value
    alphas, y_train, K_train = svm_train_dual(X_train, y_train, C)

    # Report the sum of the optimal alphas
    print(f"Sum of alphas: {np.sum(alphas):.4f}")
    print(f"Support Vectors (Dual): {len(find_support_vectors_dual(alphas))}")
    
    # Calculate and print the decision boundary coefficients (w and b)
    svs_dual = find_support_vectors_dual(alphas)
    w_dual, b_dual = calculate_w_b(alphas, y_train, X_train, C, svs_dual)
    print(f"Dual w (first 5 elements): {w_dual[:5]}")
    print(f"Dual Bias (b): {b_dual:.4f}")

    # Calculate the primal from the dual
    w_primal, b_primal = compute_primal_from_dual(X_train, y_train, alphas)
    print(f"Primal w (sum): {w_primal}")
    print(f"Primal Bias (b): {b_primal:.4f}")
    
    print(f"Shape of alphas: {alphas.shape}")
    print(f"Shape of label_train: {label_train.shape}")
    print(f"Shape of data_train: {data_train.shape}")


    # Compare the two biases
    if np.isclose(b_dual, b_primal, atol=1e-5):
        print("The bias values from the dual and primal forms match.")
    else:
        print("The bias values from the dual and primal forms differ.")

