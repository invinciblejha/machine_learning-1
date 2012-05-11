'''
Linear regression on the "Auto MPG" data set (http://archive.ics.uci.edu/ml/datasets/Auto+MPG)
The model is optimized by applying the BFGS method.
The optimized model is applied for each sample in the test set and the RMSE is computed.

Author: AC Grama http://acgrama.blogspot.com
Date: 01.05.2012
'''
import csv, datareader, math, numpy, random, scipy.optimize

def compute_hypothesis(X_row, theta):
    ''' This method computes the hypothesis for the sample X_row, with the current theta values.
    
    Args:
        X_row: a sample from the data set
        theta: vector containing the theta values
        
    Returns:
        The value of the hypothesis for the given data sample and theta values.
    '''
    theta.shape = (1, X_row.size)
    h_theta = X_row.dot(theta.T)
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
    h = X.dot(theta.T)
    J =  (h-y)**2
    cost = (1.0 / 2*m) * (J.sum())
    return cost
      
def predict(X_row, theta):
    ''' This method applies the optimized model to the sample X_row, with the theta values found after optimizing the cost function, to predict the result.
    
    Args:
        X_row: a sample from the data set
        theta: vector containing the theta values
        
    Returns:
        The value predicted for the sample in X_row, using the given theta values.
    '''
    predicted_y = compute_hypothesis(X_row, theta)
    return predicted_y
    
def check_test_data(test_X, test_y, theta):
    ''' This method applies the optimized model to the test data set, with the theta values found after optimizing the cost function. 
    Prints out the RMSE.
    
    Args:
        test_X: the test data set
        test_y: the test set's true results
        theta: vector containing the theta values
    '''        
    sum = 0
    for i in range(len(test_X)):
        prediction = predict(test_X[i], theta)
        sum += (prediction - test_y[i])**2
#        delta = float(math.fabs(prediction - test_y[i]))/100
#        print "Predicted value= %.2f" % prediction, ", actual value= ", test_y[i], ", difference in percents of actual value=%.3f" % delta
    sum /= test_X.shape[0]
    rmse = math.sqrt(sum)
    print "Root Mean Squared Error= %.2f" % rmse
    
if __name__ == "__main__":
    print "Parsing input data..."
    
    input_file = 'auto-mpg.data' 
    input_test_file = ''
    custom_delimiter = ' ' 
    proportion_factor = float(1)/3
    split = True 
    input_columns = range(1, 8) 
    output_column = 0
    input_literal_columns = [0] * 8
    input_label_mapping = {}
    output_literal = False
    output_label_mapping = {}
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, custom_delimiter, 
        proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    print "Parsing complete!\n"

    print "Optimizing...\n"
    initial_thetas = numpy.zeros((train_X.shape[1], 1))
    myargs = (train_X, train_y)
    theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_thetas, args=myargs)
    print "\n\nFinal theta: "
    print theta
    
    print "\n\nApplying model to test data..."
    check_test_data(test_X, test_y, theta)