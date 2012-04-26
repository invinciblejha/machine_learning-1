'''
Logistic regression classification on the "Internet Classification" data set (http://archive.ics.uci.edu/ml/datasets/Internet+Advertisements)
Author: AC Grama http://acgrama.blogspot.com
Date: 24.04.2012
'''
import datareader, math, numpy, random, scipy, scipy.optimize, time

iter = 0

class TooManyIterationsException(Exception):
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return repr(self.value)

def sigmoid(val):
    return 1.0 / (1.0 + numpy.e ** (-1.0 * val))
    
def compute_hypothesis(X_row, theta):
    theta.shape = (1, X_row.size)
    h_theta = sigmoid(X_row.dot(theta.T))
    return h_theta[0]

def computeCost(theta, X, y):
    global iter
    iter += 1
#    if iter > 10000:
#        raise TooManyIterationsException(iter)
#        
    m = y.size
    h = sigmoid(X.dot(theta.T)) # For each sample, a h_theta value
    
    J_part1 = y.T.dot(numpy.log(h))
    J_part2 = (1.0 - y.T).dot(numpy.log(1.0 - h))
    J =  J_part1 + J_part2
    J_reg2 = theta[1:]**2
    J_reg1 = theta[1:]
    cost = (-1.0 / m) * (J.sum()) + LAMBDA2 * J_reg2.sum() + LAMBDA1 * J_reg1.sum()
    print "Iteration", iter, " - Cost: ", cost#, " J_part1=", J_part1, ", J_part2=", J_part2
    print "Theta", theta[:4]
    print "\n\n"
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
    
def lambdas_range():
    my_range = [0.01]
    current = 0.02
    while current < 20:
        my_range.append(current)
        current *= 2
    return my_range
    
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
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, custom_delimiter, proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)   
    print "Parsing complete!\n"
    
    initial_thetas = numpy.zeros((train_X.shape[1], 1))
    myargs = (train_X, train_y)
    global LAMBDA1, LAMBDA2
    my_range = lambdas_range()
    
#    for LAMBDA1 in my_range:
#        for LAMBDA2 in my_range:
#            try:
#                iter = 0
#                print "Beginning optimization of cost function for LAMBDA1=", LAMBDA1, " and LAMBDA2=", LAMBDA2
#                theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_thetas, args=myargs, maxiter=1)            
#                print "Optimization complete!"
#            except TooManyIterationsException as e:
#                print "\n"
            
    LAMBDA1 = 0.1
    LAMBDA2 = 0.2
    print "Beginning optimization of cost function for LAMBDA1=", LAMBDA1, " and LAMBDA2=", LAMBDA2
    theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_thetas, args=myargs, maxiter=1)            
    print "Optimization complete!"

    print "Final theta: "
    print theta

    print "Ended at: ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    check_test_data(test_X, test_y, theta)