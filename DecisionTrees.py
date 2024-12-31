import numpy as np
from collections import Counter

def vote(y):
    """
    Find the most common label in a set of labels.

    # Arguments:
        y: a set of labels
        
    # Returns:
        label: the most common label
    """
    label_counts = Counter(y)
    return label_counts.most_common(1)[0][0]
    

def misclassification ( y, cls, weights=None ):
    """
    Calculate (optionally-weighted) misclassification error for
    a given set of labels if assigned the given class.

    # Arguments
        y: a set of class labels
        cls: a candidate classification for the set
        weights: optional weights vector specifying relative
            importance of the samples labelled by y

    # Returns
        err: the misclassification error of the candidate labels
    """
    if weights is None: 
        weights = 1/len(y)
    
    return np.sum(weights * (y != cls))


def gini_impurity(y):
    """
    Evaluate the gini impurity for the labels

    # Arguments:
        y: a set of class labels
    
    # Returns:
        err: the gini_impurity
    """
    classes = np.unique(y)
    pk_list_sqr = [(np.sum(y==k) / len(y))**2 for k in classes]
    return 1 - np.sum(pk_list_sqr)


def decision_node_split ( X, y, cls=None, weights=None, min_size=3 ):
    """
    Find (by brute force) a split point that best improves the weighted
    misclassification error rate compared to the original one (or not, if
    there is no improvement possible).

    Features are assumed to be numeric and the test condition is
    greater-or-equal.

    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of class labels corresponding to the samples,
            must be same length as number of rows in X
        cls: class label currently assigned to the whole set
            (if not specified we use the most common class in y, or
            the lowest such if 2 or more classes occur equally)
        weights: optional weights vector specifying relevant importance
            of the samples
        min_size: don't create child nodes smaller than this

    # Returns:
        feature: index of the feature to test (or None, if no split)
        thresh: value of the feature to test (or None, if no split)
        c0: class assigned to the set with feature < thresh (or None, if no split)
        c1: class assigned to the set with feature >= thresh (or None, if no split)
    """
    assert(X.shape[0] == len(y))

    # If there aren't enough samples no split will be legal
    if len(y) < min_size * 2:
        return None, None, None, None

    # Default to the most common class
    if cls is None: 
        cls = vote(y)

    # Default to equal weighting
    if weights is None:
        weights = np.ones(len(y))/len(y)

    # Define the parent node loss
    best_loss = misclassification(y, cls=cls, weights=weights)

    # When the best_loss is zero, we've found the perfect class label for the node
    if best_loss == 0: 
        return None, None, None, None

    # keep track of our best candidate to date
    best_feat, best_thresh = None, None
    best_c0, best_c1 = None, None

    # Loop through all features and threshold values
    for feat in range(X.shape[-1]):
        for thresh in np.unique(X[:,feat]):
            # Use boolean index sets
            set1 = X[:,feat] >= thresh
            set0 = ~set1

            # Skip splits producing too small children
            if (np.sum(set0) < min_size) or (np.sum(set1) < min_size):
                continue

            # Select the labels and matching weights
            y0 = y[set0]
            y1 = y[set1]

            w0 = weights[set0]
            w1 = weights[set1]

            # Any label that occurs in the y0, y1 is a candidate to assign
            cc0 = np.unique(y0)
            cc1 = np.unique(y1)

            # Calculate the loss associated with each candidate label in each child
            # Note that because we pass the weights, these losses will scale for node size
            losses0 = [misclassification(y0, cls=cc, weights=w0) for cc in cc0]
            losses1 = [misclassification(y1, cls=cc, weights=w1) for cc in cc1]

            # determine best labels to assign to each node
            c0 = cc0[np.argmin(losses0)]
            c1 = cc1[np.argmin(losses1)]
                
            # and the corresponding weighted losses
            loss = np.min(losses0) + np.min(losses1)
                
            # special case: if split is perfect we can stop searching right now
            if loss == 0:
                return feat, thresh, c0, c1

            # if loss is better than we've seen so far, update
            # note that this (along with the special case above) implicitly defines
            # a tie-break policy: first come, first served
            if loss < best_loss:
                best_feat = feat
                best_thresh = thresh
                best_c0 = c0
                best_c1 = c1
                best_loss = loss

    # note that if we didn't find a useful split these will all still be None
    return best_feat, best_thresh, best_c0, best_c1


def decision_tree_split_with_gini( X, y, min_size=3 ):
    """
    Find (by brute force) a split point that best improves the weighted
    misclassification error rate compared to the original one (or not, if
    there is no improvement possible).

    Features are assumed to be numeric and the test condition is
    greater-or-equal.

    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of class labels corresponding to the samples,
            must be same length as number of rows in X
        cls: class label currently assigned to the whole set
            (if not specified we use the most common class in y, or
            the lowest such if 2 or more classes occur equally)
        weights: optional weights vector specifying relevant importance
            of the samples
        min_size: don't create child nodes smaller than this

    # Returns:
        feature: index of the feature to test (or None, if no split)
        thresh: value of the feature to test (or None, if no split)
        c0: class assigned to the set with feature < thresh (or None, if no split)
        c1: class assigned to the set with feature >= thresh (or None, if no split)
    """
    assert(X.shape[0] == len(y))

    # If there aren't enough samples no split will be legal
    if len(y) < min_size * 2:
        return None, None
    
    # Define the parent node loss
    best_loss = gini_impurity(y)

    # When the best_loss is zero, we've found the perfect class label for the node
    if best_loss == 0: 
        return None, None

    # keep track of our best candidate to date
    best_feat, best_thresh = None, None
    
    # Loop through all features and threshold values
    for feat in range(X.shape[-1]):
        for thresh in np.unique(X[:,feat]):
            # Use boolean index sets
            set1 = X[:,feat] >= thresh
            set0 = ~set1

            # Skip splits producing too small children
            if (np.sum(set0) < min_size) or (np.sum(set1) < min_size):
                continue

            # Select the labels
            y0 = y[set0]
            y1 = y[set1]

            # Calculate the gini impurity for y0, y1 and add them together
            loss = (len(y0) / len(y)) * gini_impurity(y0) + (len(y1) / len(y)) * gini_impurity(y1)
                
            # special case: if split is perfect we can stop searching right now
            if loss == 0:
                return feat, thresh

            # if loss is better than we've seen so far, update
            # note that this (along with the special case above) implicitly defines
            # a tie-break policy: first come, first served
            if loss < best_loss:
                best_feat = feat
                best_thresh = thresh
                best_loss = loss

    # note that if we didn't find a useful split these will all still be None
    return best_feat, best_thresh
    


def decision_tree_train ( X, y, loss_func='misclassification', cls=None, weights=None, min_size=3, depth=0, max_depth=10 ):
    """
    Recursively choose split points for a training dataset
    until no further improvement occurs.

    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of class labels corresponding to the samples,
            must be same length as number of rows in X
        loss_func: a string that represents the loss function we use for the decision tree split
                   it's 'misclassfication' by default
        cls: class label currently assigned to the whole set
            (if not specified we use the most common class in y, or
            the lowest such if 2 or more classes occur equally)
        weights: optional weights vector specifying relevant importance
            of the samples
        min_size: don't create child nodes smaller than this
        depth: current recursion depth
        max_depth: maximum allowed recursion depth

    # Returns:
        tree: a dict containing (some of) the following keys:
            'kind' : either 'leaf' or 'decision'
            'class' : the class assigned to this node (for a leaf)
            'feature' : index of feature on which to split (for a decision)
            'thresh' : threshold at which to split the feature (for a decision)
            'below' : a nested tree applicable when feature < thresh
            'above' : a nested tree applicable when feature >= thresh
    """
    # Default to the most common class
    if cls is None: 
        label_counts = Counter(y)
        cls = label_counts.most_common(1)[0][0]

    # If we've gone as deep as we can, stop
    if depth == max_depth:
        return { 'kind' : 'leaf', 'class' : cls }

    # Default to equal weighting
    if weights is None:
        weights = np.ones(len(y))/len(y)

    # Search for a split with misclassification error
    if loss_func =='misclassification':
        feat, thresh, cls0, cls1 = decision_node_split ( X, y, cls=cls, weights=weights, min_size=min_size )
         
        # If there isn't one split with less loss(best_loss=0), stop, we're at the leaf
        if feat is None:
            return { 'kind' : 'leaf', 'class' : cls }
    
    # Search for a split with gini impurity
    if loss_func == 'gini impurity':
        feat, thresh = decision_tree_split_with_gini(X, y, min_size=min_size )
        cls1, cls0= None, None
        # If there isn't one split with less loss(best_loss=0), stop, we're at the leaf
        if feat is None:
            return { 'kind' : 'leaf', 'class' : vote(y) }

    # Index the child data
    set1 = X[:,feat] >= thresh
    set0 = ~set1

    return { 'kind' : 'decision',
             'feature' : feat,
             'thresh' : thresh,

             # Recurse to obtain child trees
             'above' : decision_tree_train(X[set1,:], y[set1], loss_func, cls1, weights[set1], min_size, depth+1, max_depth),
             'below' : decision_tree_train(X[set0,:], y[set0], loss_func, cls0, weights[set0], min_size, depth+1, max_depth) }
    
    
def decision_tree_predict ( tree, X ):
    """
    Predict labels for test data using a fitted decision tree.

    # Arguments
        tree: a decision tree dictionary returned by decision_tree_train
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """
    # Auxiliary function to predict the classification of a single sample
    def decision_tree_predict_single ( tree, x ):
        # Descend the tree until we reach the leaf
        while True:
            if tree['kind'] == 'leaf':
                return tree['class']
            
            # If x[tree['feature']] >= tree['thresh'], then the single sample x belongs to tree['above'] smaples
            # The class that the set of tree['above'] samples belong to is the class of the single sample x
            tree = tree['above'] if x[tree['feature']] >= tree['thresh'] else tree['below']
    
    # Apply the decision_tree_prediction_single function to each sample row, 
    # and store the class, that each sample classified to, into an array
    return np.array([decision_tree_predict_single(tree, X[ii,:]) for ii in range(X.shape[0])])