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
 
def parse_input(input_file, custom_delimiter, input_columns, output_column, is_test, input_literal_columns, input_label_mapping, output_literal, output_label_mapping):
    data_reader = csv.reader(open(input_file, 'rb'), delimiter=custom_delimiter)
    
    if not is_test:
        X = []
        y = []
        for row in data_reader:
            line_x = []
            for i in input_columns:
                if input_literal_columns[i] == 1:
                    line_x.append(float(input_label_mapping[i][row[i]]))
                else:
                    line_x.append(float(row[i]))
            X.append(line_x)
            if output_literal:
                y.append(float(output_label_mapping[row[output_column]]))
            else:
                y.append(float(row[output_column]))
        
        (X, y) = randomize_inputs(X, y)
    else:
        X = []
        for row in data_reader:
            line_x = []
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
    '''
    (X, y) = parse_input(input_file, custom_delimiter, input_columns, output_column, False, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    
    if split:
        splice_index = int(len(y) * proportion_factor)
        train_X = X[splice_index:]
        train_y = y[splice_index:]
        test_X = X[:splice_index]
        test_y = y[:splice_index]
        return (train_X, train_y, test_X, test_y)
    else: # Take test values from input_test_file -- we assume same format as input_file!
        (test_X, test_y) = parse_input(input_test_file, custom_delimiter, input_columns, output_column, True, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
        return (X, y, test_X, test_y)