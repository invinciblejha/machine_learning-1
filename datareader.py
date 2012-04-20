'''
Helper class for parsing input data and generating train/test sets.
Author: AC Grama http://acgrama.blogspot.com
Date: 20.04.2012
'''
import csv, random

def  rename_y(initial_train_y, target_class):
    ''' Renames the labels to binary.
    '''
    train_y = []
    for i in range(len(initial_train_y)):
        if initial_train_y[i] == target_class:
            train_y.append(1)
        else:
            train_y.append(0)
    return train_y

def randomize_inputs(X, y):
    ''' Randomizes the input samples, just in case they are neatly ordered in the raw form.
    '''
    sequence = range(len(y))
    random.shuffle(sequence)

    new_X = []
    new_y = []
    for i in sequence:
        new_X.append(X[i])
        new_y.append(y[i])
    
    return (new_X, new_y)
 
def readInputData(input_file, custom_delimiter, proportion_factor):
    ''' Main method for parsing the input data. The input data is expected in CSV format, with a delimiter that can be specified as parameter.
    The method generates a random permutation of the read data to be safe in case the original raw data is nicely ordered.
    It uses the proportion_factor to determine how much data should be for training and how much for testing.
    '''
    data_reader = csv.reader(open(input_file, 'rb'), delimiter=custom_delimiter)
    X = []
    y = []
    for row in data_reader:
        line_x = []
        line_x.append(float(row[0]))
        line_x.append(float(row[1]))
        line_x.append(float(row[2]))
        line_x.append(float(row[3]))
        X.append(line_x)
        y.append(row[4])
    
    (X, y) = randomize_inputs(X, y)
    
    splice_index = int(len(y) * proportion_factor)
    train_X = X[splice_index:]
    train_y = y[splice_index:]
    test_X = X[:splice_index]
    test_y = y[:splice_index]

    return (train_X, train_y, test_X, test_y)