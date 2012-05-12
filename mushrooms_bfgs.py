'''
Logistic regression classification on the "Mushrooms" data set (http://archive.ics.uci.edu/ml/datasets/Mushroom)

Author: AC Grama http://acgrama.blogspot.com
Date: 24.04.2012
'''
import csv, datareader, math, mlpy, matplotlib.pyplot as plt, numpy, random, scipy.optimize

def sigmoid(val):
    ''' This method computes the sigmoid function value for the given parameter
    
    Args:
        val: input parameter
        
    Returns:
        The value of the sigmoid function for the given parameter.
    '''
    return 1.0 / (1.0 + numpy.e ** (-1.0 * val))
    
def compute_hypothesis(X_row, theta):
    ''' This method computes the hypothesis for the sample X_row, with the current theta values.
    
    Args:
        X_row: a sample from the data set
        theta: vector containing the theta values
        
    Returns:
        The value of the hypothesis for the given data sample and theta values.
    '''
    theta = numpy.array(theta)
    X_row = numpy.array(X_row)
    theta.shape = (1, X_row.size)
    h_theta = sigmoid(X_row.dot(theta.T))
    return h_theta[0]

def computeCost(theta, X, y):
    ''' This method computes the cost for the data set X and theta values w.r.t. the target values y
    
    Args:
        X: the data set
        theta: vector containing the theta values
        y: vector containing the true cost for the samples in the data set

    Returns:
        A vector containing cost values for each sample in X, for the given theta values and true costs in y.
    '''
    new_theta = numpy.array(theta)
    new_X = numpy.array(X)
    new_y = numpy.array(y)
    m = new_y.size
    h = sigmoid(new_X.dot(new_theta.T))
    new_h = numpy.array(h)
    J = new_y.T.dot(numpy.log(new_h)) + (1.0 - new_y.T).dot(numpy.log(1.0 - new_h))
    cost = (-1.0 / m) * J.sum()
#    print "Cost: ", cost
    return cost
      
def predict(X_row, theta):
    ''' This method applies the optimized model to the sample X_row, with the theta values found after optimizing the cost function, to predict the result.
    
    Args:
        X_row: a sample from the data set
        theta: vector containing the theta values
        
    Returns:
        The class predicted for the sample in X_row, using the given theta values.
    '''
    predicted_y = compute_hypothesis(X_row, theta)
    if predicted_y >= 0.5:
        return 1
    else:
        return 0
    
def check_test_data(test_X, test_y, theta):
    ''' This method applies the optimized model to the test data set, with the theta values found after optimizing the cost function. 
    
    Args:
        test_X: the test data set
        test_y: the test set's true results
        theta: vector containing the theta values
    '''        
    correct = 0
    for i in range(len(test_X)):
        prediction = predict(test_X[i], theta)
        #print "Predicted ", prediction, ", actual ", test_y[i]
        if prediction == test_y[i]:
            correct += 1
    print "Correct predictions: ", correct, "/", len(test_X)

def plot_data(X, y):
    pca = mlpy.PCA()
    pca.learn(X)
    z = pca.transform(X, k=2)

    plt.set_cmap(plt.cm.Paired)
    fig1 = plt.figure(1)
    title = plt.title("PCA on mushroom dataset")
    plot = plt.scatter(z[:, 0], z[:, 1], c=y)
    labx = plt.xlabel("First component")
    laby = plt.ylabel("Second component")
    plt.show()

if __name__ == "__main__":
    print "Parsing input data..."
    input_file = 'agaricus-lepiota.data' 
    input_test_file = ''
    custom_delimiter = ',' 
    proportion_factor = float(1)/3
    split = True 
    input_columns = range(1, 23) 
    output_column = 0
    input_literal_columns = [1] * 23
    input_label_mapping = {1:{'b':0, 'c':1, 'x':2, 'f':3, 'k':4, 's':5}, 2:{'f':0, 'g':1, 'y':2, 's':3}, 3:{'n':0, 'b':1, 'c':2, 'g':3, 'r':4, 'p':5, 'u':6, 'e':7, 'w':8, 'y':9}, 4:{'t':0, 'f':1}, 
        5:{'a':0, 'l':1, 'c':2, 'y':3, 'f':4, 'm':5, 'n':6, 'p':7, 's':8}, 6:{'a':0, 'd':1, 'f':2, 'n':3}, 7:{'c':0, 'w':1, 'd':2}, 8:{'b':0, 'n':1}, 9:{'k':0, 'n':1, 'b':2, 'h':3, 'g':4, 'r':5, 'o':6, 'p':7, 'u':8, 'e':9, 'w':10, 'y':11},
        10:{'e':0, 't':1}, 11:{'b':0, 'c':1, 'u':2, 'e':3, 'z':4, 'r':5, '?':6}, 12:{'f':0, 'y':1, 'k':2, 's':3}, 13:{'f':0, 'y':1, 'k':2, 's':3}, 14:{'n':0, 'b':1, 'c':2, 'g':3, 'o':4, 'p':5, 'e':6, 'w':7, 'y':8}, 
        15:{'n':0, 'b':1, 'c':2, 'g':3, 'o':4, 'p':5, 'e':6, 'w':7, 'y':8}, 16:{'p':0, 'u':1}, 17:{'n':0, 'o':1, 'w':2, 'y':3}, 18:{'n':0, 'o':1, 't':2}, 19:{'c':0, 'e':1, 'f':2, 'l':3, 'n':4, 'p':5, 's':6, 'z':7},
        20:{'k':0, 'n':1, 'b':2, 'h':3, 'r':4, 'o':5, 'u':6, 'w':7, 'y':8}, 21:{'a':0, 'c':1, 'n':2, 's':3, 'v':4, 'y':5}, 22:{'g':0, 'l':1, 'm':2, 'p':3, 'u':4, 'w':5, 'd':6}}
    output_literal = True
    output_label_mapping = {'p':1, 'e':0}
   
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, True, custom_delimiter, proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    print "Parsing complete!\n"
   
    # Uncomment the following line to use PCA and to plot the training data set
    #plot_data(train_X, train_y)
   
    print "Optimizing...\n"
    initial_values = numpy.zeros((len(train_X[0]), 1))
    myargs = (train_X, train_y)
    theta = scipy.optimize.fmin_bfgs(computeCost, x0=initial_values, args=myargs)

    print "Final theta: "
    print theta
    
    check_test_data(test_X, test_y, theta)