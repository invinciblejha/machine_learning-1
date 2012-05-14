'''
k-Means clustering  on the "Aggregation" data set (http://cs.joensuu.fi/sipu/datasets/)

Author: AC Grama http://acgrama.blogspot.com
Date: 05.05.2012
'''
import datareader, math, matplotlib.pyplot as plt, mlpy, numpy, random, scipy, scipy.optimize, time

ITERATIONS = 500
EPSILON = 0.001 # When the difference between centroid position in step k and in step k+1 falls below this, we have converged

def initialize_centroids(X, number_of_centroids):
    ''' Randomly initializes a number of centroids equal to the value given by the parameter. The centroids will be initialized to random samples from X.
        
        Args:
            X: the samples.
            number_of_centroids: The number of centroids to be initialized.
            
        Returns:
            The list of initialized centroids. Each element in the list has an x and y position.
    '''
    centroids = []
    random.seed()
    for i in range(number_of_centroids):
        x = X[random.uniform(0, len(X))][0]
        y = X[random.uniform(0, len(X))][1]
        centroids.append([x, y])
    return centroids
    
def cluster_assignment_step(X, centroids):
    ''' Assigns each sample to the cluster with the centroid closest to the sample.
        
        Args:
            X: samples.
            centroids: List of centroids.
            
        Returns:
            The list of centroids. Element k in the list is the cluster to which sample k is currently assigned.
    '''
    y = [-1] * len(X)
    
    for i in range(len(X)):
        xpos = X[i][0]
        ypos = X[i][1]
        min = 1000000
        min_cluster = -1
        
        for j in range(len(centroids)):
            centroidx = centroids[j][0]
            centroidy = centroids[j][1]
            dist = math.sqrt((xpos-centroidx)**2 + (ypos-centroidy)**2)
            if dist < min:
                min = dist
                min_cluster = j
        
        y[i] = min_cluster
    
    return y
    
def centroid_movement_step(X, y, centroids):
    ''' Moves each centroid to the average location of points in that cluster.
        
        Args:
            X: samples.
            y: current cluster assignments.
            centroids: List of centroids.
            
        Returns:
            The updated list of centroids.
    '''
    for i in range(len(centroids)):
        # Compute average location of points in cluster i
        sumx = 0
        sumy = 0
        nr_vals  = 0
        for k in range(len(X)):
            if y[k]==i:
                sumx += X[k][0]
                sumy += X[k][1]
                nr_vals +=1 
        if nr_vals != 0:
            centroids[i][0] = sumx/nr_vals
            centroids[i][1] = sumy/nr_vals
    return centroids
    
def converged(old_centroids, new_centroids):
    ''' Verifies whether the algorithm has converged: true, if none of the new centroids have a distance to their old counterparts that is bigger than EPSILON.
        
        Args:
            old_centroids: List of old centroids.
            new_centroids: List of new centroids.
            
        Returns:
            True, if the algorithm has converged.
    '''
    flag = False
    for i in range(len(old_centroids)):
        dist = math.sqrt((old_centroids[i][0] - new_centroids[i][0])**2 + (old_centroids[i][1] - new_centroids[i][1])**2)
        if dist < EPSILON:
            flag = True
    return flag

def compute_distortion(X, y, centroids):
    ''' Computes the distortion for a particular clustering.
        
        Args:
            X: samples.
            y: current cluster assignments.
            centroids: List of centroids.
            
        Returns:
            The distortion of this particular clustering.
    '''
    distortion = 0
    for i in range(len(X)):
        distortion += math.sqrt((X[i][0] - centroids[y[i]][0])**2 + (X[i][1] - centroids[y[i]][1])**2)
    
    distortion /= len(X)
    return distortion
    
def plot_data(X, y):
    ''' Plots the given clustering, colouring the points in a cluster with the same colour.
        
        Args:
            X: samples.
            y: current cluster assignments.
    '''
    plt.set_cmap(plt.cm.Paired)
    fig1 = plt.figure(1)
    title = plt.title("Plot of the Aggregation dataset")
    plot = plt.scatter(X[:, 0], X[:, 1], c=y)
    labx = plt.xlabel("x")
    laby = plt.ylabel("y")
    plt.show()
    
if __name__ == "__main__":
    print "k-Means clustering  on the 'Aggregation' data set (http://cs.joensuu.fi/sipu/datasets/)\n\n"
    
    print "Parsing input data..."
    input_file = 'aggregation.data' 
    input_test_file = ''
    custom_delimiter = ' ' 
    proportion_factor = float(1)/5
    split = True 
    input_columns = range(2) 
    output_column = 2
    input_literal_columns = [0] * 2
    input_label_mapping = {}
    output_literal = False
    output_label_mapping = {}
    (train_X, train_y, test_X, test_y) = datareader.readInputData(input_file, input_test_file, True, custom_delimiter, proportion_factor, split, input_columns, output_column, input_literal_columns, input_label_mapping, output_literal, output_label_mapping)   
    # Eliminate the intercept in X[0] => we don't need it for clustering!
    train_X = train_X[:, 1:]
    print "Parsing complete!\n"
  
    # Randomly assign the initial 7 cluster centroids
    number_of_centroids = 7
    centroids = initialize_centroids(train_X, number_of_centroids)
    min_centroids = []
    min_y = []
    min_distortion = 1000000
    
    print "Performing ", ITERATIONS, " iterations to determine the clustering with least distortion!"
    raw_input("Press Enter to continue...")

    for i in range(ITERATIONS):
        while True:
            # Perform the cluster assignment step
            y = cluster_assignment_step(train_X, centroids)
            
            # Perform the centroid movement step
            new_centroids = centroid_movement_step(train_X, y, centroids)
            
            if converged(centroids, new_centroids):
                break
        distortion = compute_distortion(train_X, y, new_centroids)
        if distortion < min_distortion:
            min_distortion = distortion
            min_y = y
            min_centroids = new_centroids
        print "Iteration ", i#, " -- distortion=", min_distortion
    
    #plot_data(train_X, train_y) # Uncomment this line to see what the actual clusters are
    plot_data(train_X, min_y)