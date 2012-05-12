'''
k-Means clustering on the "Plants" data set (http://archive.ics.uci.edu/ml/datasets/Plants)

Author: AC Grama http://acgrama.blogspot.com
Date: 05.05.2012
'''

import csv, datareader, matplotlib.pyplot as plt, mlpy, numpy

def map_X(X, state_abbrev, is_test):
    new_X = []
    for row in train_X:
        if not is_test:
            new_X_line = [0]*71
            row.remove(1)
            new_X_line[0] = 1
        else:
            new_X_line = [0]*70
        for element in row:
            new_X_line[state_abbrev[element]] = 1
        new_X.append(new_X_line)
    return new_X
            
def plot_data(X, y):
    pca = mlpy.PCA()
    pca.learn(X)
    z = pca.transform(X, k=2)

    plt.set_cmap(plt.cm.Paired)
    fig1 = plt.figure(1)
    title = plt.title("PCA on iris dataset")
    plot = plt.scatter(z[:, 1], z[:, 0], c=y)
    labx = plt.xlabel("First component")
    laby = plt.ylabel("Second component")
    plt.show()
    
if __name__ == "__main__":
    print "Parsing input data..."
    
    input_file = 'plants.data' 
    input_test_file = ''
    custom_delimiter = ',' 
    proportion_factor = float(1)/3
    split = True 
    input_columns = range(1, 68)
    output_column = 0
    input_literal_columns = []
    input_label_mapping = {}
    output_literal = True
    output_label_mapping = {}
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, False, custom_delimiter, 
        proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)
    print "Parsing complete!\n"
    
    state_abbrev = {'ab':0, 'ak':1, 'ar':2, 'az':3, 'ca':4, 'co':5, 'ct':6, 'de':7, 'dc':8, 'fl':9, 'ga':10, 'hi':11, 'id':12, 'il':13, 'in':14, 'ia':15, 'ks':16, 'ky':17, 'la':18, 'me':19, 'md':20, 
    'ma':21, 'mi':22, 'mn':23, 'ms':24, 'mo':25, 'mt':26, 'ne':27, 'nv':28, 'nh':29, 'nj':30, 'nm':31, 'ny':32, 'nc':33, 'nd':34, 'oh':35, 'ok':36, 'or':37, 'pa':38, 'pr':39, 'ri':40, 
    'sc':41, 'sd':42, 'tn':43, 'tx':44, 'ut':45, 'vt':46, 'va':47, 'vi':48, 'wa':49, 'wv':50, 'wi':51, 'wy':52, 'al':53, 'bc':54, 'mb':55, 'nb':56, 'lb':57, 'nf':58, 'nt':59, 'ns':60, 
    'nu':61, 'on':62, 'qc':63, 'sk':64, 'yt':65, 'dengl':66, 'fraspm':67, 'pe':68, 'gl':69}

    new_train_X = numpy.array(map_X(train_X, state_abbrev, False))
    new_test_X = numpy.array(map_X(test_X, state_abbrev, True))
    new_train_y = numpy.array(range(0, len(train_y)))

 