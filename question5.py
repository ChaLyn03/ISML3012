import cvxpy as cvx
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

def svm_train_dual(data_train, label_train, regularisation_para_C):
    samples, features = data_train.shape
    label_train = label_train * 2 - 1  # Convert 0/1 to -1/1
    X = np.dot(data_train, data_train.T)
    
    P = matrix(np.outer(label_train, label_train) * X)
    Q = matrix(-np.ones(samples))
    G = matrix(np.vstack((-np.eye(samples), np.eye(samples))))
    H = matrix(np.hstack((np.zeros(samples), np.ones(samples) * regularisation_para_C/samples)))
    A = matrix(label_train, (1,samples), 'd')
    B = matrix(0.0)
    solution = solvers.qp(P,Q,G,H,A,B)
    alpha = np.ravel(solution['x'])
    return {'alpha': alpha}

def compute_primal_solution(alpha, data_train, label_train, regularisation_para_C, tolerance=1e-5):
    support_vector_indices = np.where((alpha > tolerance) & (alpha < regularisation_para_C))[0]
    label_train = label_train * 2 - 1 
    
    w_star = np.sum((alpha * label_train).reshape(-1, 1) * data_train, axis=0)
    
    print(f"Number of support vectors: {len(support_vector_indices)}")
    print("Support vector indices:", support_vector_indices)
    
    b_star_values = []
    for idx in support_vector_indices:
        x_s = data_train[idx]
        y_s = label_train[idx]
        b_star_value = y_s - np.dot(w_star, x_s)
        b_star_values.append(b_star_value)
    
    # Write support vectors to file
    with open('question5.txt', 'w') as f:
        f.write(f"Number of support vectors: {len(support_vector_indices)}\n")
        f.write(f"Support vector indices: {support_vector_indices.tolist()}\n")
    
    return w_star, None  # b_star removed

train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)
validation_data = train_data[4000:]
train_data = train_data[:4000]
regularisation_para_C = 100
label_train = train_data.iloc[:, 0].values  
data_train = train_data.iloc[:, 1:].values 
label_test = test_data.iloc[:, 0].values  
data_test = test_data.iloc[:, 1:].values 
optimal = svm_train_dual(data_train, label_train, regularisation_para_C)
np.save('alpha.npy', optimal['alpha'])
alpha = np.load('alpha.npy')
w_star, _ = compute_primal_solution(alpha, data_train, label_train, regularisation_para_C)
