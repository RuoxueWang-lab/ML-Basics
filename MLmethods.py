import numpy as np
import GredienrDecent

def LinearRegression(X, y):
    """
    Implement Linear Regression Method

    Args:
        X: an array of sample data, where rows are samples
                   and columns are features (assume there are at least
                   as many samples as features). caller is responsible
                   for prepending x0=1 terms if required.
        y: vector of measured lables of samples, must be the same length as numer of rows in X
        
    Returns:
        w (array): a vector of optimized weights.
    """
    w = np.linalg.solve(X.T @ X, X.T @ y)
    return w


def RidgeRegression(X, y, l2, exclude_x0=False):
    """
    Implement L2-penalised least-squares (ridge) regression
    using its closed form expression.

    # Arguments
        X: an array of sample data, where rows are samples
           and columns are features (assume there are at least
           as many samples as features). caller is responsible
           for prepending x0=1 terms if required.
        y: vector of measured (or simulated) labels for the samples,
           must be same length as number of rows in X
        l2: optional L2 regularisation weight. if zero (the default)
           then this reduces to unregularised least squares
        exclude_x0 (bool): whether to treat the first data feature as representing
           an intercept term and exclude it from the regularisation.
           note that this defaults to False, meaning regularise everything

    # Returns
        w (array): the fitted vector of weights
    """
    reg = np.identity(X.shape[1])

    if exclude_x0:
        # Since if we don't want to regulize the intercept, we set i_{1,1} of the indentity matrix to be 0
        reg[0,0] = 0

    return np.linalg.solve(X.T @ X + l2 * reg, X.T @ y)


def logistic_regression_with_batch_GD (X, y, w0=None, lr=0.05,
                                       loss_stop=1e-4, weight_stop=1e-4, max_iter=100):
    """
    Fit a logistic regression classifier to data.

    # Arguments
        X: an array of sample data, where rows are samples
           and columns are features. caller is responsible
           for prepending x0=1 terms if required.
        y: vector of binary class labels for the samples,
           must be same length as number of rows in X
        w0: starting value of the weights, if omitted
           then all zeros are used
        lr: learning rate, ie fraction of gradients by
           which to update weights at each iteration
        loss_stop: stop iterating if the loss changes
            by less than this (absolute)
        weight_stop: stop iterating if weights change by less
            than this (L2 norm)
        max_iter: stop iterating after iterating this
            many times

    # Returns
        ws: a list of fitted weights at each iteration
        losses: a list of the loss values at each iteration
    """
    def sigmoid ( z ):
        return 1/(1 + np.exp(-z))

    def logistic_forward ( X, w ):
        return sigmoid(X @ w)

    def logistic_loss ( X, y, w, eps=1e-10 ):
        g = logistic_forward(X, w)
        return (np.dot(-y, np.log(g + eps)) - np.dot((1 - y), np.log(1 - g + eps)))/len(y)

    def logistic_grad ( X, y, w ):
        g = logistic_forward(X, w)
        return X.T @ (g - y)

    if w0 is None: 
        w0 = np.zeros(X.shape[-1])

    return GredienrDecent.batch_gradient_descent( w0,
                                                  loss_func = lambda z: logistic_loss(X, y, z),
                                                  grad_loss_func = lambda z: logistic_grad(X, y, z),
                                                  lr = lr,
                                                  loss_stop=loss_stop, w_stop=weight_stop, max_iter=max_iter)
    
    
def logistic_regression_with_stochastic_GD ( X, y, batch_size=1, w0=None, lr=0.05,
                                             loss_stop=1e-4, weight_stop=1e-4, max_iter=100 ):
    """
    Fit a logistic regression classifier to data.

    # Arguments
        X: an array of sample data, where rows are samples
           and columns are features. caller is responsible
           for prepending x0=1 terms if required.
        y: vector of binary class labels for the samples,
           must be same length as number of rows in X
        batch_size: the size of our batch, it's set to 1 by default.
        w0: starting value of the weights, if omitted
           then all zeros are used
        lr: learning rate, ie fraction of gradients by
           which to update weights at each iteration
        loss_stop: stop iterating if the loss changes
            by less than this (absolute)
        weight_stop: stop iterating if weights change by less
            than this (L2 norm)
        max_iter: stop iterating after iterating this
            many times

    # Returns
        ws: a list of fitted weights at each iteration
        losses: a list of the loss values at each iteration
    """
    data = list(zip(X,y))
    
    def sigmoid ( z ):
        return 1/(1 + np.exp(-z))

    def logistic_forward (X, w):
        return sigmoid(X @ w)

    def logistic_loss ( X, y, w, eps=1e-10 ):
        g = logistic_forward(X, w)
        return (np.dot(-y, np.log(g + eps)) - np.dot((1 - y), np.log(1 - g + eps)))/len(y)
    
    def logistic_grad ( data, w, batch_indices):
        XBatch = np.array([data[i][0] for i in batch_indices])
        yBatch = np.array([data[i][1] for i in batch_indices])
        g = logistic_forward(XBatch, w)
        return XBatch.T @ (g - yBatch)
    
    if w0 is None:
        w0 = np.zeros((X.shape[-1]))   
    
    return GredienrDecent.stochastic_gradient_descent(w0,
                                                      loss_func = lambda w: logistic_loss(X, y, w),
                                                      grad_loss_func = logistic_grad,
                                                      data = data,
                                                      batch_size = batch_size,
                                                      lr = lr,
                                                      loss_stop=loss_stop, w_stop=weight_stop, max_iter=max_iter)
    

    
    
    




    





