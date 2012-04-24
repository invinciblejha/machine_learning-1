'''
Logistic regression classification on the "Internet Classification" data set (http://archive.ics.uci.edu/ml/datasets/Internet+Advertisements)
Author: AC Grama http://acgrama.blogspot.com
Date: 24.04.2012
'''
import math, numpy, random, scipy, datareader_nonvector, scipy.optimize, time

LAMBDA2 = 0.2
LAMBDA1 = 0.1

def sigmoid(val):
    return 1.0 / (1.0 + numpy.e ** (-1.0 * val))
    
def compute_hypothesis(X_row, theta):
    theta = numpy.array(theta)
    X_row = numpy.array(X_row)
    theta.shape = (1, X_row.size)
    h_theta = sigmoid(X_row.dot(theta.T))
    return h_theta[0]

def computeCost(theta, X, y):
    new_theta = numpy.array(theta)
    new_X = numpy.array(X)
    new_y = numpy.array(y)
    m = new_y.size
    h = sigmoid(new_X.dot(new_theta.T)) # For each sample, a h_theta value
    new_h = numpy.array(h)
    J = new_y.T.dot(numpy.log(new_h)) + (1.0 - new_y.T).dot(numpy.log(1.0 - new_h)) # For each sample, a J_cost value
    J_reg2 = new_theta[1:]**2
    J_reg1 = new_theta[1:]
    cost = - 1 *  (1.0 / m) * (J.sum() + LAMBDA2 * J_reg2.sum() + LAMBDA1 * J_reg1.sum())
    print "Cost: ", cost
    return cost
       
def predict(X_row, theta):
    predicted_y = compute_hypothesis(X_row, theta)
    if predicted_y >= 0.5:
        return 1
    else:
        return 0
    
def check_test_data(test_X, test_y, thetas, classes):
    correct = 0
    for i in range(len(test_X)):        
        predicted_class = predict(test_X[i], thetas)
        #print "Predicted: ", classes[predicted_class], ", Actual: ", test_y[i]
        if classes[predicted_class] == test_y[i]:
            correct += 1
    print "Correct predictions: ", correct, "/", len(test_X)
    
if __name__ == "__main__":
    print "Started at: ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    print "Parsing input data..."
    input_file = 'ad.data' 
    input_test_file = ''
    custom_delimiter = ',' 
    proportion_factor = float(1)/3
    split = True 
    input_columns = range(1558) 
    output_column = 1558
    input_literal_columns = [0] * 1558
    input_label_mapping = {}
    output_literal = True
    output_label_mapping = {'ad.':1, 'nonad.':0}
    (train_X, train_y, test_X, test_y) = datareader_nonvector.readInputData(input_file, input_test_file, custom_delimiter, proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)   
    print "Parsing complete!\n"
    
    initial_values = numpy.zeros((len(train_X[0]), 1))
    myargs = (train_X, train_y)
    print "Beginning optimization of cost function..."
    theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_values, args=myargs, maxiter=20, disp=True)
    print "Optimization complete!"
    
    print "Final theta: "
    print theta

    print "Ended at: ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    check_test_data(test_X, test_y, theta)