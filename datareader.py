'''
Helper class for parsing input data and generating train/test sets.
Author: AC Grama http://acgrama.blogspot.com
Date: 20.04.2012
'''
import csv, numpy, random

def  rename_y(initial_train_y, target_class):
    ''' In case we are doing one-vs-all classification, this function will set the category for samples from the target class to "1" and for samples from any other classes to "0"
    
    Args:
        initial_train_y: the full set of classes for the training set
        target_class: the class which is the target in a particular iteration of one-vs-all classification.
        
    Returns:
        A vector containing the categories for all the samples, in which the category for samples from the target class is "1" and for samples from any other classes is "0".
    '''
    train_y = []
    for i in range(len(initial_train_y)):
        if initial_train_y[i] == target_class:
            train_y.append(1)
        else:
            train_y.append(0)
    return numpy.array(train_y)

def randomize_inputs(X, y):
    ''' Randomizes the input samples, just in case they are neatly ordered in the raw form.
    
    Args:
        X: data samples
        y: outputs for the data samples in X
        
    Returns:
        The shuffled input data samples.
    '''
    sequence = range(len(y))
    random.shuffle(sequence)

    new_X = []
    new_y = []
    for i in sequence:
        new_X.append(X[i])
        new_y.append(y[i])
    
    return (new_X, new_y)
 
def parse_input(input_file, custom_delimiter, input_columns, output_column, is_test, input_literal_columns, input_label_mapping, output_literal, output_label_mapping):
    ''' This function parses the input data file.
    
    Args:
        input_file: The file containing the input data.
        custom_delimiter: The delimiter used in the input files.
        input_columns: Which columns in the input data are inputs (X). If input_columns is empty, the data samples have variable length.
        output_column: Which column in the input data is output value (y).
        is_test: Set to True if we are parsing input from a test set.
        input_literal_columns: Which columns in the input data have a literal description and need to be mapped to custom numeric values.
        input_label_mapping: Mapping for input literal columns.
        output_literal: Boolean, shows whether output is literal or numeric.
        output_label_mapping: Mapping for output literal column. If this is empty and output_literal=True, it means that we have some clustering task where every sample has a specific output -- the outputs will be marked 0, 1, etc.
    
    Returns:
        A set (X, y) containing the input data.    
    '''
    data_reader = csv.reader(open(input_file, 'rb'), delimiter=custom_delimiter)
    
    if not is_test:
        X = []
        y = []
        index = 0
        for row in data_reader:
            line_x = [1] # Add the X0=1
            while '' in row:
                row.remove("")
            if input_columns != []:
                for i in input_columns:
                    if input_literal_columns[i] == 1:
                        line_x.append(float(input_label_mapping[i][row[i]]))
                    else:
                        line_x.append(float(row[i]))
                X.append(line_x)
                if output_literal:
                    if output_label_mapping != {}:
                        y.append(float(output_label_mapping[row[output_column]]))
                    else:
                        y.append(index)
                        index += 1
                else:
                    y.append(float(row[output_column]))
            else:
                for i in range(len(row)):
                    line_x.append(float(row[i]))
                X.append(line_x)
        
        (X, y) = randomize_inputs(X, y)
    else:
        X = []
        for row in data_reader:
            line_x = [1]
            for i in range(len(row)):
                line_x.append(float(row[i]))
            X.append(line_x)
        
        y = [0.0] * len(X) # Dummy y
        (X, y) = randomize_inputs(X, y)
    
    return (X, y)

def readInputData(input_file, input_test_file, custom_delimiter, proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping):
    ''' Main method for parsing the input data. The input data is expected in CSV format, with a delimiter that can be specified as parameter.
    The method generates a random permutation of the read data to be safe in case the original raw data is nicely ordered.
    It uses the proportion_factor to determine how much data should be for training and how much for testing.
    
    Args:
        input_file: The file containing the input data.
        input_test_file: The file containing the test data (if applicable).
        custom_delimiter: The delimiter used in the input files.
        proportion_factor: If there is no special input_test_file, a percentage of proportion_factor% from the input_file will be used as test data. The samples are randomly selected.
        split: If true, the test data will be taken from input_file. Otherwise, from input_test_file.
        input_columns: Which columns in the input data are inputs (X).
        output_column: Which column in the input data is output value (y).
        input_literal_columns: Which columns in the input data have a literal description and need to be mapped to custom numeric values.
        input_label_mapping: Mapping for input literal columns.
        output_literal: Boolean, shows whether output is literal or numeric.
        output_label_mapping: Mapping for output literal column.
    
    Returns:
        A set (train_X, train_y, test_X, test_y) containing training data and test data. The test_y array contains dummy values.
    '''
    (X, y) = parse_input(input_file, custom_delimiter, input_columns, output_column, False, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    
    if split:
        splice_index = int(len(y) * proportion_factor)
        train_X = X[splice_index:]
        train_y = y[splice_index:]
        test_X = X[:splice_index]
        test_y = y[:splice_index]
        return (numpy.array(train_X), numpy.array(train_y), numpy.array(test_X), numpy.array(test_y))
    else: # Take test values from input_test_file -- we assume same format as input_file!
        (test_X, test_y) = parse_input(input_test_file, custom_delimiter, input_columns, output_column, True, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
        return (numpy.array(X), numpy.array(y), numpy.array(test_X), numpy.array(test_y))