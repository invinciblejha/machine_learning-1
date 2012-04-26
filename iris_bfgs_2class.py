'''
Logistic regression classification on the "Iris" data set (http://archive.ics.uci.edu/ml/datasets/Iris)
Uses scipy,optimize.fmin_bfgs for minimizing the cost function.
Author: AC Grama http://acgrama.blogspot.com
Date: 24.04.2012
'''
import csv, datareader, math, mlpy, matplotlib.pyplot as plt, numpy, random, scipy.optimize

LAMBDA1 = 0.01
LAMBDA2 = 0.02

def sigmoid(val):
    return 1.0 / (1.0 + numpy.e ** (-1.0 * val))
    
def compute_hypothesis(X_row, theta):
    theta.shape = (1, X_row.size)
    h_theta = sigmoid(X_row.dot(theta.T))
    return h_theta[0]

def computeCost(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.T)) # For each sample, a h_theta value
    
    J =  y.T.dot(numpy.log(h)) + (1.0 - y.T).dot(numpy.log(1.0 - h))
    J_reg2 = theta[1:]**2
    J_reg1 = theta[1:]
    cost = (-1.0 / m) * (J.sum()) + LAMBDA2 * J_reg2.sum() + LAMBDA1 * J_reg1.sum()
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

def plot_data(X, y):
    pca = mlpy.PCA() # new PCA instance
    pca.learn(X) # learn from data
    z = pca.transform(X, k=2) # embed x into the k=2 dimensional subspace

    plt.set_cmap(plt.cm.Paired)
    fig1 = plt.figure(1)
    title = plt.title("PCA on iris dataset")
    plot = plt.scatter(z[:, 0], z[:, 1], c=y)
    labx = plt.xlabel("First component")
    laby = plt.ylabel("Second component")
    plt.show()
    
if __name__ == "__main__":
    print "Parsing input data..."
    
    input_file = 'iris.data' 
    input_test_file = ''
    custom_delimiter = ',' 
    proportion_factor = float(1)/3
    split = True 
    input_columns = range(4) 
    output_column = 4
    input_literal_columns = [0] * 4
    input_label_mapping = {}
    output_literal = True
    output_label_mapping = {'Iris-versicolor': 0, 'Iris-setosa': 0, 'Iris-virginica': 1}
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, custom_delimiter, 
        proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    print "Parsing complete!\n"
    
    plot_data(train_X, train_y)
        
    print "Optimizing...\n"
    initial_thetas = numpy.zeros((train_X.shape[1], 1))
    myargs = (train_X, train_y)
    theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_thetas, args=myargs)

    print "Final theta: "
    print theta
    
    check_test_data(test_X, test_y, theta)