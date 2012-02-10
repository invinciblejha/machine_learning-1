import csv, numpy

ITERATIONS = 1500;
ALPHA = 0.01;

def readInputData(input_file):
    data_reader = csv.reader(open(input_file, 'rb'), delimiter=',')
    X = []
    y = []
    for row in data_reader:
        X.append(float(row[0]))
        y.append(row[1])
    return (X, y)

def computeCost(X, y, theta):
    pass
    
def gradientDescent():
    pass
    
if __name__ == "__main__":
    (X1, y) = readInputData('ex1data1.txt')
    m = len(y) 
    
    # Add a column of ones to X
    X = [numpy.ones((len(X1))), X1]
    theta = numpy.zeros((2, 1))

    # Compute initial cost
    computeCost(X, y, theta)

    #theta = gradientDescent(X, y, theta, alpha, iterations);
