'''
Linear regression on the "Auto MPG" data set (http://archive.ics.uci.edu/ml/datasets/Auto+MPG)
Author: AC Grama http://acgrama.blogspot.com
Date: 01.05.2012
'''
import csv, datareader, math, numpy, random, scipy.optimize

def compute_hypothesis(X_row, theta):
    theta.shape = (1, X_row.size)
    h_theta = X_row.dot(theta.T)
    return h_theta[0]

def computeCost(theta, X, y):
    m = y.size
    h = X.dot(theta.T)
    J =  (h-y)**2
    cost = (1.0 / 2*m) * (J.sum())
    return cost
      
def predict(X_row, theta):
    predicted_y = compute_hypothesis(X_row, theta)
    return predicted_y
    
def check_test_data(test_X, test_y, theta):
    for i in range(len(test_X)):
        prediction = predict(test_X[i], theta)
        delta = float(math.fabs(prediction - test_y[i]))/100
        print "Predicted value= %.2f" % prediction, ", actual value= ", test_y[i], ", difference in percents of actual value=%.3f" % delta
    
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