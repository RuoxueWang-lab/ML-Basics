import numpy as np
from collections import Counter


def KNN(train_X, train_y, test_X, neighbours=1):
    """
    Predict labels for test data based on neighbourhood in
    training set.

    # Arguments:
        train_X: an array of sample data for training, where rows
            are samples and columns are features.
        train_y: vector of class labels corresponding to the training
            samples, must be same length as number of rows in X
        test_X: an array of sample data to generate predictions for,
            in same layout as train_X.
        neighbours: how many neighbours to canvass at each test point

    # Returns
        test_y: predicted labels for the samples in test_X
    """
    assert(train_X.shape[0] == train_y.shape[0])
    assert(train_X.shape[1] == test_X.shape[1])

    test_y = np.zeros(test_X.shape[0])

    # each test sample must be compared against every training sample,
    # making k-NN computationally expensive for large training sets
    for ii in range(test_X.shape[0]):

        # need to associate distances with labels, so generate a list of tuples
        dists = [ (np.linalg.norm(test_X[ii] - train_X[jj]), train_y[jj]) for jj in range(train_X.shape[0]) ]

        # sort on distance
        dists.sort(key=lambda z: z[0])

        # take the k nearest and extract their labels
        candidates = [ z[1] for z in dists[:neighbours] ]

        # Count the frequency of each label in the candidates list
        label_counts = Counter(candidates)
        
        # Find the label with the maximum count
        most_common_label = label_counts.most_common(1)[0][0]
        test_y[ii] = most_common_label

    return test_y

