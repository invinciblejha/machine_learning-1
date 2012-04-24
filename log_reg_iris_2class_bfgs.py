import csv, datareader_nonvector, math, numpy, random, scipy.optimize

LAMBDA = 0.01

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
    J_reg = new_theta[1:]**2
    cost = (-1.0 / m) * (J.sum() + LAMBDA * J_reg.sum())
    print "Cost: ", cost
    return cost
      
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
    (train_X, train_y, test_X, test_y) = datareader_nonvector.readInputData('iris.data', '', ',', float(1)/3, True, [0, 1, 2, 3], 4, True, {'Iris-versicolor': 1, 'Iris-setosa': 0, 'Iris-virginica': 0})
   
    initial_values = numpy.zeros((len(train_X[0]), 1))
    myargs = (train_X, train_y)
    theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_values, args=myargs)

    print "Final theta: "
    print theta
    
    check_test_data(test_X, test_y, theta)