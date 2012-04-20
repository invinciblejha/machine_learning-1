'''
Logistic regression classification.
Author: AC Grama http://acgrama.blogspot.com
Date: 20.04.2012
'''
import math, numpy, random, scipy, datareader

ITERATIONS = 1500
ALPHA = 0.01

def func(params, *args):
    x = args[0]
    y = args[1]
    m, b = params
    y_model = m*x+b
    error = y-y_model
    return sum(error**2)
    
def compute_hypothesis(X_row, theta):
#    sum = 0
 #   for i in range(len(theta)):
  #      sum += theta[i]*X_row[i]

    h_theta = float(1)/(1 + math.exp(-numpy.dot(theta, X_row)))
    return h_theta

def computeCost(X, y, theta):
    m = len(y)
    cost = 0
    for i in range(0, m-1):
        h_theta = compute_hypothesis(X[i], theta)
        cost = cost + (float(y[i]) * math.log(h_theta) + float(1.0 - y[i]) * math.log(1 - h_theta))
    cost = float(cost) / (-m)
    return cost
    
def gradientDescent(X, y, theta):
    m = len(y)
    J_history = numpy.zeros((ITERATIONS, 1))

    for iter in range(1, ITERATIONS):
        print "Iteration ", iter
        # Perform a single gradient step on the parameter vector
        theta_j_update_values = [0] * len(X[0])
        for j in range(len(X[0])):
            sum = 0
            for i in range(0, m-1):
                h_theta = compute_hypothesis(X[i], theta)
                sum += (h_theta - y[i]) * X[i][j]
                print "Attribute ", j, "Sample ", i
            theta_j_update_values[j] = sum

        for j in range(len(theta)):
            theta[j] = theta[j] - ALPHA * (1/float(m)) * theta_j_update_values[j]

        # Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta)
        #print J_history[iter]
    return theta
    
def predict(X_row, theta):
    sum = 0
    for j in range(len(theta)):
        sum += theta[j]*X_row[j]
    predicted_y = float(1)/(1 + math.exp(-sum))
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
    (train_X, train_y, test_X, test_y) = datareader.readInputData('train.csv', 'test.csv', ',', 0, False, range(1777), 0)
    m = len(train_y)
    
    theta = [0] * len(train_X[0])
    # Compute initial cost
    cost = computeCost(train_X, train_y, theta)

    theta = gradientDescent(train_X, train_y, theta)
    print "Final theta: "
    print theta
    
    #check_test_data(test_X, test_y, theta)