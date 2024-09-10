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
    
    print(len(support_vector_indices))
    b_star_values = []
    for idx in support_vector_indices:
        x_s = data_train[idx]
        y_s = label_train[idx]
        b_star_value = y_s - np.dot(w_star, x_s)
        b_star_values.append(b_star_value)
    
    # Average the bias terms from all support vectors
    b_star = np.mean(b_star_values) if b_star_values else 0
    
    return w_star, b_star


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
w_star, b_star = compute_primal_solution(alpha, data_train, label_train, regularisation_para_C)
w_star = np.sum(w_star)

print("w*:", w_star, "b*:", b_star)



