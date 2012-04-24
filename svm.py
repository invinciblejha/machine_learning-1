import csv, datareader_nonvector, math, numpy, random, scipy.optimize


if __name__ == "__main__":
    (train_X, train_y, test_X, test_y) = datareader_nonvector.readInputData('iris.data', '', ',', float(1)/5, True, [0, 1, 2, 3], 4, True, {'Iris-versicolor': 0, 'Iris-setosa': 0, 'Iris-virginica': 1})
   
