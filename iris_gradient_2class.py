'''
Logistic regression classification on the "Iris" data set (http://archive.ics.uci.edu/ml/datasets/Iris)
Uses homebrew gradient descent implementation for minimizing the cost function.
Author: AC Grama http://acgrama.blogspot.com
Date: 24.04.2012
'''
import csv, datareader, math, numpy, random

ITERATIONS = 1500
ALPHA = 0.01
LAMBDA1 = 0.1
LAMBDA2 = 0.2

def sigmoid(val):
    return 1.0 / (1.0 + numpy.e ** (-1.0 * val))

def compute_hypothesis(X_row, theta):
    theta.shape = (1, X_row.size)
    h_theta = sigmoid(X_row.dot(theta.T))
    return h_theta[0]

def computeCost(X, y, theta):
    m = y.size
    theta.shape = (1, X.shape[1])
    h = sigmoid(X.dot(theta.T)) # For each sample, a h_theta value
    
    J_part1 = y.T.dot(numpy.log(h))
    J_part2 = (1.0 - y.T).dot(numpy.log(1.0 - h))
    J =  J_part1 + J_part2
    J_reg2 = theta[1:]**2
    J_reg1 = theta[1:]
    cost = (-1.0 / m) * (J.sum()) + LAMBDA2 * J_reg2.sum() + LAMBDA1 * J_reg1.sum()
    print "Cost: ", cost#, " J_part1=", J_part1, ", J_part2=", J_part2
    return cost
    
def gradientDescent(X, y, theta):
    m = len(y)
    J_history = numpy.zeros((ITERATIONS, 1))

    for iter in range(1, ITERATIONS):
        # Perform a single gradient step on the parameter vector
        theta_j_update_values = [0] * len(X[0])
        for j in range(len(X[0])):
            sum = 0
            for i in range(0, m-1):
                h_theta = compute_hypothesis(X[i], theta)
                sum += (h_theta - y[i]) * X[i][j]
            theta_j_update_values[j] = sum

        for j in range(len(theta)):
            theta[j] = theta[j] - ALPHA * (1/float(m)) * theta_j_update_values[j]

        # Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta)
        #print J_history[iter]
    return theta
    
def predict(X_row, theta):
    predicted_y = compute_hypothesis(X_row, theta)
    if predicted_y >= 0.5:
        return 1
    else:
        return 0
    
def check_test_data(test_X, test_y, theta):
    correct = 0
    for i in range(len(test_X)):
        prediction = predict(test_X[i], theta)
        #print "Predicted ", prediction, ", actual ", test_y[i]
        if prediction == test_y[i]:
            correct += 1
    print "Correct predictions: ", correct, "/", len(test_X)
    
if __name__ == "__main__":
    print "Parsing input data..."
    
    input_file = 'iris.data' 
    input_test_file = ''
    custom_delimiter = ',' 
    proportion_factor = float(1)/3
    split = True 
    input_columns = range(4) 
    output_column = 4
    input_literal_columns = [0] * 4
    input_label_mapping = {}
    output_literal = True
    output_label_mapping = {'Iris-versicolor': 0, 'Iris-setosa': 0, 'Iris-virginica': 1}
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, custom_delimiter, 
        proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    print "Parsing complete!\n"

    m = train_y.size
    theta = numpy.zeros((train_X.shape[1], 1))

    # Compute initial cost
    cost = computeCost(train_X, train_y, theta)

    theta = gradientDescent(train_X, train_y, theta)
    print "Final theta: "
    print theta
    
    check_test_data(test_X, test_y, theta)