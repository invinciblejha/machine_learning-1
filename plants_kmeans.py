'''
k-Means clustering on the "Plants" data set (http://archive.ics.uci.edu/ml/datasets/Plants)

Author: AC Grama http://acgrama.blogspot.com
Date: 05.05.2012
'''

import csv, datareader

if __name__ == "__main__":
    print "Parsing input data..."
    
    input_file = 'plants.data' 
    input_test_file = ''
    custom_delimiter = ',' 
    proportion_factor = float(1)/3
    split = True 
    input_columns = []
    output_column = 0
    input_literal_columns = []
    input_label_mapping = {}
    output_literal = True
    output_label_mapping = {}
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, False, custom_delimiter, 
        proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    print "Parsing complete!\n"
    
    print train_X
    
    state_abbrev = {'ab':0, 'ak':1, 'ar':2, 'az':3, 'ca':4, 'co':5, 'ct':6, 'de':7, 'dc':8, 'fl ':9, 'ga':10, 'hi':11, 'id':12, 'il':13, 'in':14, 'ia':15, 'ks':16, 'ky':17, 'la':18, 'me':19, 'md':20, 
    'ma':21, 'mi':22, 'mn':23, 'ms':24, 'mo':25, 'mt':26, 'ne':27, 'nv':28, 'nh':29, 'nj':30, 'nm':31, 'ny':32, 'nc':33, 'nd':34, 'oh':35, 'ok':36, 'or':37, 'pa':38, 'pr':39, 'ri':40, 
    'sc':41, 'sd':42, 'tn':43, 'tx':44, 'ut':45, 'vt':46, 'va':47, 'vi':48, 'wa':49, 'wv':50, 'wi':51, 'wy':52, 'al':53, 'bc':54, 'mb':55, 'nb':56, 'lb':57, 'nf':58, 'nt':59, 'ns':60, 
    'nu':61, 'on':62, 'qc':63, 'sk':64, 'yt':65, 'dengl':66, 'fraspm':67}

