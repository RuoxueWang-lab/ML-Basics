import numpy as np
import DecisionTrees
from collections import Counter


def random_forest_train ( X, y, k, min_size=3, max_depth=10 ):
    """
    Train a (simplified) random forest of decision trees.

    # Arguments:
        X: an array of sample data
        y: vector of binary class labels
        k: the number of trees in the forest
        min_size: don't create child nodes smaller than this
        max_depth: maximum tree depth

    # Returns:
        forest: a list of tree dicts as returned by decision_tree_train
    """
    forest = []
    for i in range(k):
        # Genetate X.shape[0] number of indices from the number of rows of X 
        boost_idx = np.random.choice(X.shape[0], X.shape[0])
        
        # Then our boostrap samples are:
        X_boost = X[boost_idx, :]
        y_boost = y[boost_idx]
        
        # Append each tree into the forest
        forest.append(DecisionTrees.decision_tree_train(X_boost, y_boost, min_size=min_size, max_depth=max_depth))
        
    return forest


def random_forest_predict ( forest, X ):
    """
    Predict labels for test data using a fitted randomforest of decision trees.

    # Arguments
        forest: a list of decision tree dicts
        X: an array of sample data

    # Returns
        y: the predicted labels
    """
    # Get the predictions predicted by different trees
    preds = np.array([DecisionTrees.decision_tree_predict(tree, X) for tree in forest])
    
    # Predict each sample by votes from all trees
    final_predict = []
    for col in range(preds.shape[1]):
        sample_predict = preds[:, col] # Different predictions for each sample by different trees in the forest
        label_counts = Counter(sample_predict)
        final_predict.append(label_counts.most_common(1)[0][0])
        
    return np.array(final_predict)
        