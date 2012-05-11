'''
Logistic regression classification of N classes (one-vs-all method). Applied to iris dataset.

Author: AC Grama http://acgrama.blogspot.com
Date: 20.04.2012
'''
import math, numpy, random, scipy, scipy.optimize, datareader

def sigmoid(val):
    ''' This method computes the sigmoid function value for the given parameter
    
    Args:
        val: input parameter
        
    Returns:
        The value of the sigmoid function for the given parameter.
    '''
    return 1.0 / (1.0 + numpy.e ** (-1.0 * val))
    
def compute_hypothesis(X_row, theta):
    ''' This method computes the hypothesis for the sample X_row, with the current theta values.
    
    Args:
        X_row: a sample from the data set
        theta: vector containing the theta values
        
    Returns:
        The value of the hypothesis for the given data sample and theta values.
    '''
    theta.shape = (1, X_row.size)
    h_theta = sigmoid(X_row.dot(theta.T))
    return h_theta[0]

def computeCost(theta, X, y):
    ''' This method computes the cost for the data set X and theta values w.r.t. the target values y
    
    Args:
        X: the data set
        theta: vector containing the theta values
        y: vector containing the true cost for the samples in the data set

    Returns:
        A vector containing cost values for each sample in X, for the given theta values and true costs in y.
    '''
    m = y.size
    h = sigmoid(X.dot(theta.T))
    
    J =  y.T.dot(numpy.log(h)) + (1.0 - y.T).dot(numpy.log(1.0 - h))
    cost = (-1.0 / m) * (J.sum())
    return cost
      
def predict(X_row, thetas):
    ''' This method applies the optimized model to the sample X_row, with the theta values found after optimizing the cost function, to predict the result.
    
    Args:
        X_row: a sample from the data set
        theta: vector containing the theta values
        
    Returns:
        The class predicted for the sample in X_row, using the given theta values. We are doing one-vs-all prediction, so each set of thetas will be used to 
        make a prediction for a sample, and the highest prediction will be returned as the most likely one.
    '''
    max_predict = 0
    max_class = 0
    for k in range(len(thetas)):
        predicted_y = compute_hypothesis(X_row, theta)
        if predicted_y > max_predict:
            max_predict = predicted_y
            max_class = k
    return max_class
    
def check_test_data(test_X, test_y, thetas, classes):
    ''' This method applies the optimized model to the test data set, with the theta values found after optimizing the cost function. 
    Prints out the RMSE.
    
    Args:
        test_X: the test data set
        test_y: the test set's true results
        theta: vector containing the theta values
    '''        
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
    output_label_mapping = {'Iris-versicolor': 0, 'Iris-setosa': 1, 'Iris-virginica': 2}
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, custom_delimiter, 
        proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    print "Parsing complete!\n"
    
    print train_y
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