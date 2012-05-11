'''
k-Means clustering on the "Plants" data set (http://archive.ics.uci.edu/ml/datasets/Plants)

Author: AC Grama http://acgrama.blogspot.com
Date: 05.05.2012
'''

import csv, datareader, math, mlpy, matplotlib.pyplot as plt, numpy, random, scipy.optimize

if __name__ == "__main__":
    print "Parsing input data..."
    
    input_file = 'plants.data' 
    input_test_file = ''
    custom_delimiter = ',' 
    proportion_factor = float(1)/3
    split = True 
    input_columns = []
    output_column = 0
    input_literal_columns = [1] * 8
    input_label_mapping = {}
    output_literal = True
    output_label_mapping = {}
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, custom_delimiter, 
        proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    print "Parsing complete!\n"

