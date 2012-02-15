import csv, numpy

ITERATIONS = 1500
ALPHA = 0.01

def readInputData(input_file):
    data_reader = csv.reader(open(input_file, 'rb'), delimiter=',')
    X = []
    y = []
    for row in data_reader:
        X.append(float(row[0]))
        y.append(float(row[1]))
    return (X, y)

def computeCost(X, y, theta):
    m = len(y)
    cost = 0
    for i in range(0, m):
        cost = cost + (theta[0] + theta[1]*X[i] - y[i])**2
    
    cost = cost / (2*m)
    return cost
    
def gradientDescent(X, y, theta):
    m = len(y)
    J_history = numpy.zeros((ITERATIONS, 1))

    for iter in range(1, ITERATIONS):
        # Perform a single gradient step on the parameter vector
        sum0 = 0
        sum1 = 0
        for i in range(1, m):
            sum0 = sum0 + theta[0] + theta[1]*X[i] - y[i]
            sum1 = sum1 + (theta[0] + theta[1]*X[i] - y[i]) * X[i]
        theta[0] = theta[0] - ALPHA * (1/float(m)) * sum0
        theta[1] = theta[1] - ALPHA * (1/float(m)) * sum1

        # Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta)
        #print J_history[iter]
    return theta
    
def predict(X, theta):
    predicted_y = theta[0] + theta[1]*X[i]
    return predicted_y
    
if __name__ == "__main__":
    (X, y) = readInputData('ex1data1.txt')
    m = len(y) 
    
    theta = numpy.zeros((2, 1))

    # Compute initial cost
    cost = computeCost(X, y, theta)

    theta = gradientDescent(X, y, theta)
    print "Final theta: "
    print theta
