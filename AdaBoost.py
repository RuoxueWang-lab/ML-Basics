import numpy as np
import DecisionTrees

def adaboost_train ( X, y, k, min_size=1, max_depth=1, epsilon=1e-8 ):
    """
    Iteratively train a set of decision tree classifiers
    using AdaBoost.

    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        k: the maximum number of weak classifiers to train
        min_size: don't create child nodes smaller than this
        max_depth: maximum tree depth -- by default we just
            use decision stumps
        epsilon: threshold below which the error is considered 0

    # Returns:
        trees: a list of tree dicts as returned by decision_tree_train
        alphas: a vector of weights indicating how much credence to
            given each of the decision tree predictions
    """
    weights = np.ones(X.shape[0])/X.shape[0]
    alphas = []
    trees = []

    for ii in range(k):
        trees.append(DecisionTrees.decision_tree_train(X, y, weights=weights, min_size=min_size, max_depth=max_depth))
        pred_y = DecisionTrees.decision_tree_predict(trees[-1], X)
        err = np.dot(weights, pred_y != y)

        # Break the loop if the classification is already perfect, no more weak classifiers need to be trained
        # as no mistake needs to be corrected by further weak classifiers
        if err < epsilon:
            alphas.append(1)
            break

        alphas.append(np.log((1 - err)/err))

        weights = weights * np.exp(alphas[-1] * (pred_y != y))
        weights = weights / np.sum(weights)

    return trees, np.array(alphas)


def adaboost_predict ( trees, alphas, X ):
    """
    Predict labels for test data using a fitted AdaBoost
    ensemble of decision trees.

    # Arguments
        trees: a list of decision tree dicts
        alphas: a vector of weights for the trees
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """
    # Due to new convention of AdaBoost: classes are denoted by {-1, 1},
    # we need to time the result of the decision_tree_predict function by 2 and minus 1
    # (convention for decision tree is {0,1})
    DT_pred = np.array([DecisionTrees.decision_tree_predict(tree, X) for tree in trees]).T * 2 -1
    sum = DT_pred @ alphas
    y = (sum >= 0).astype(int)
    return y