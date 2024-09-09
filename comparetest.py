# Assuming the primal and dual methods are already run and available as functions
# You can either import them from their respective files or run them here directly
import pandas as pd
import numpy as np
# Import necessary methods from question2.py and question3.py
from question2 import svm_train_primal, svm_predict_primal
from question3 import svm_train_dual  # Assuming svm_train_dual is created from the dual form

# Load the datasets (adjust the path as necessary)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Extract training and test data
X_train = train_data.iloc[:4000, 1:].values
y_train = train_data.iloc[:4000, 0].values
y_train = np.where(y_train == 0, -1, 1)

X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values
y_test = np.where(y_test == 0, -1, 1)

# Train using the primal method (Question 2)
C = 100
w_primal, b_primal = svm_train_primal(X_train, y_train, C)

# Train using the dual method (Question 3)
w_dual, b_dual = svm_train_dual(X_train, y_train, C)

# Now you can compare the results using the previously defined compare_models function

def compare_models(w_primal, b_primal, w_dual, b_dual, X_test, y_test):
    # Compare the weight vectors
    w_diff = np.linalg.norm(w_primal - w_dual)
    print(f"Difference in weight vectors (L2 norm): {w_diff}")

    # Compare the bias terms
    b_diff = abs(b_primal - b_dual)
    print(f"Difference in bias terms: {b_diff}")

    # Predictions from both models
    predictions_primal = np.sign(np.dot(X_test, w_primal) + b_primal)
    predictions_dual = np.sign(np.dot(X_test, w_dual) + b_dual)

    # Accuracy comparison
    test_accuracy_primal = np.mean(predictions_primal == y_test)
    test_accuracy_dual = np.mean(predictions_dual == y_test)
    print(f"Primal Test Accuracy: {test_accuracy_primal * 100:.2f}%")
    print(f"Dual Test Accuracy: {test_accuracy_dual * 100:.2f}%")

    # Compare predictions
    prediction_match = np.mean(predictions_primal == predictions_dual)
    print(f"Percentage of matching predictions: {prediction_match * 100:.2f}%")

# Call the comparison function
compare_models(w_primal, b_primal, w_dual, b_dual, X_test, y_test)
