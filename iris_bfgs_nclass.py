'''
Logistic regression classification of N classes (one-vs-all method). Applied to iris dataset.

Author: AC Grama http://acgrama.blogspot.com
Date: 20.04.2012
'''
import math, numpy, random, scipy, scipy.optimize, datareader

LAMBDA1 = 0.01
LAMBDA2 = 0.02

def sigmoid(val):
    return 1.0 / (1.0 + numpy.e ** (-1.0 * val))
    
def compute_hypothesis(X_row, theta):
    theta.shape = (1, X_row.size)
    h_theta = sigmoid(X_row.dot(theta.T))
    return h_theta[0]

def computeCost(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.T)) # For each sample, a h_theta value
    
    J =  y.T.dot(numpy.log(h)) + (1.0 - y.T).dot(numpy.log(1.0 - h))
    J_reg2 = theta[1:]**2
    J_reg1 = theta[1:]
    cost = (-1.0 / m) * (J.sum()) + LAMBDA2 * J_reg2.sum() + LAMBDA1 * J_reg1.sum()
    return cost
      
def predict(X_row, thetas):
    max_predict = 0
    max_class = 0
    for k in range(len(thetas)):
        predicted_y = compute_hypothesis(X_row, theta)
        if predicted_y > max_predict:
            max_predict = predicted_y
            max_class = k
    return max_class
    
def check_test_data(test_X, test_y, thetas, classes):
    correct = 0
    for i in range(len(test_X)):        
        predicted_class = predict(test_X[i], thetas)
        #print "Predicted: ", predicted_class, ", Actual: ", test_y[i]
        if predicted_class == test_y[i]:
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
    
    print "Optimizing...\n"
    initial_train_y = train_y
    thetas = []
    classes = range(3)
    for target_class in classes:
        train_y = datareader.rename_y(initial_train_y, target_class)
        
        initial_thetas = numpy.zeros((train_X.shape[1], 1))
        myargs = (train_X, train_y)
        theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_thetas, args=myargs)

        print "Final theta for target class ", target_class
        print theta
        thetas.append(theta)
    
    check_test_data(test_X, test_y, thetas, classes)