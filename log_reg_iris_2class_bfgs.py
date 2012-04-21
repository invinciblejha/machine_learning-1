import csv, datareader, math, numpy, random, scipy.optimize

def compute_hypothesis(X_row, theta):
    sum = 0
    for i in range(len(theta)):
        sum += theta[i]*X_row[i]
        
    h_theta = float(1)/(1 + math.exp(-sum))
    return h_theta

def computeCost(theta, X, y):
    m = len(y)
    cost = 0
    for i in range(0, m-1):
        h_theta = compute_hypothesis(X[i], theta)
        cost = cost + (y[i] * math.log(h_theta) + (1.0 - y[i]) * math.log(1.0 - h_theta)) 
    cost = float(cost) / (-m)
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

def func(params, *args):
    X = args[0]
    y = args[1]
    theta = params
    error = 0
    
    for i in range(len(X)):
        y_model = compute_hypothesis(X[i], theta)
        error += float(1)/2*(y[i] - y_model)**2
    return error

    
if __name__ == "__main__":
    (train_X, train_y, test_X, test_y) = datareader.readInputData('iris.data', '', ',', float(1)/5, True, [0, 1, 2, 3], 4, True, {'Iris-versicolor': 0, 'Iris-setosa': 0, 'Iris-virginica': 1})
    m = len(train_y)
   
    initial_values = numpy.zeros((len(train_X[0]), 1))
    myargs = (train_X, train_y)
    theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_values, args=myargs)

    print "Final theta: "
    print theta
    
    check_test_data(test_X, test_y, theta)