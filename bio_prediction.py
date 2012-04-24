'''
Logistic regression classification.
Author: AC Grama http://acgrama.blogspot.com
Date: 20.04.2012
'''
import math, numpy, random, scipy, datareader_nonvector, scipy.optimize, time

LAMBDA = 100000
FN_EVALS = 0

def sigmoid(val):
    return 1.0 / (1.0 + numpy.e ** (-1.0 * val))
    
def compute_hypothesis(X_row, theta):
    theta = numpy.array(theta)
    X_row = numpy.array(X_row)
    theta.shape = (1, X_row.size)
    h_theta = sigmoid(X_row.dot(theta.T))
    return h_theta[0]

def computeCost(theta, X, y):
    global  FN_EVALS
    FN_EVALS += 1
    print "Function evaluation #", FN_EVALS
    new_theta = numpy.array(theta)
    new_X = numpy.array(X)
    new_y = numpy.array(y)
    m = new_y.size
    h = sigmoid(new_X.dot(new_theta.T)) # For each sample, a h_theta value
    new_h = numpy.array(h)
    J = new_y.T.dot(numpy.log(new_h)) + (1.0 - new_y.T).dot(numpy.log(1.0 - new_h)) # For each sample, a J_cost value
    J_reg = new_theta[1:]**2
    cost = - 1 *  (1.0 / m) * (J.sum() + LAMBDA * J_reg.sum())
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
    (train_X, train_y, test_X, test_y) = datareader_nonvector.readInputData('train.csv', 'uniform_benchmark.csv', ',', 0, False, range(1777), 0, False, {})
    m = len(train_y)    
    print "Parsing complete!\n"
    
    initial_values = numpy.zeros((len(train_X[0]), 1))
    myargs = (train_X, train_y)
    print "Beginning optimization of cost function..."
    theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_values, args=myargs)
    print "Optimization complete!"
    
    print "Final theta: "
    print theta

    print "Ended at: ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    check_test_data(test_X, test_y, theta)