import csv, numpy

ITERATIONS = 1500;
ALPHA = 0.01;

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
    
def gradientDescent():
    pass
    
if __name__ == "__main__":
    (X, y) = readInputData('ex1data1.txt')
    m = len(y) 
    
    # Add a column of ones to X
    #X = [numpy.ones((len(X1))), X1]
    theta = numpy.zeros((2, 1))

    # Compute initial cost
    cost = computeCost(X, y, theta)
    print cost

    #theta = gradientDescent(X, y, theta, alpha, iterations);
