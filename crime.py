'''
Linear regression classification on the "Communities and Crime" data set (http://archive.ics.uci.edu/ml/machine-learning-databases/00211/)
Author: AC Grama http://acgrama.blogspot.com
Date: 25.04.2012
'''
import math, numpy, random, scipy, datareader_nonvector, scipy.optimize, time
  
def compute_hypothesis(X_row, theta):
    theta = numpy.array(theta)
    X_row = numpy.array(X_row)
    theta.shape = (1, X_row.size)
    h_theta = X_row.dot(theta.T)
    return h_theta[0]

def computeCost(theta, X, y):
    new_theta = numpy.array(theta)
    new_X = numpy.array(X)
    new_y = numpy.array(y)
    m = new_y.size
    h = new_X.dot(new_theta.T)
    new_h = numpy.array(h)
    J = (new_y.T - new_h)**2
    cost = (1.0 / 2*m) * J.sum()
    print "Cost: ", cost
    return cost
       
def predict(X_row, theta):
    predicted_y = compute_hypothesis(X_row, theta)
    return predicted_y
    
def check_test_data(test_X, test_y, thetas, classes):
    correct = 0
    for i in range(len(test_X)):        
        predicted_value = predict(test_X[i], thetas)
        #print "Predicted: ", classes[predicted_class], ", Actual: ", test_y[i]
        if predicted_value == test_y[i]:
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
    global LAMBDA1, LAMBDA2, MAX_ITER, ITER
    MAX_ITER = 200
    
    for LAMBDA1 in range(0, 10, 1):
        for LAMBDA2 in range(0, 10, 1):
            ITER = 0
            print "Beginning optimization of cost function for LAMBDA1=", LAMBDA1, " and LAMBDA2=", LAMBDA2
            theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_values, args=myargs, maxfun=2, full_output=True, retall=True)
            
            print "Optimization complete!"
    
    print "Final theta: "
    print theta

    print "Ended at: ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    check_test_data(test_X, test_y, theta)