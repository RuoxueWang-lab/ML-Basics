import numpy as np

def generate_noisy_linear(num_samples, weights, sigma, limits):
    """
    Draw samples from a linear model with additive Gaussian noise.

    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the model
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i

    # Returns
        X: a matrix of sample inputs, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x (len(weights) - 1)
        y: a vector of num_samples output values
    """
    # generate random matrix of the input features
    X = np.random.uniform(low=limits[0], high=limits[1], size=(num_samples, len(weights) - 1))

    # calculate the corresponding model output values
    y = X @ weights[1:] + weights[0]

    # add some Gaussian noise
    y = y + np.random.normal(scale=sigma, size=y.shape)

    return X, y


def generate_linearly_separable(num_samples, weights, limits):
    """
    Draw samples from a binary model with a given linear
    decision boundary.

    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector defining the decision boundary
            (including a bias term at index 0)
        limits: a tuple (low, high) specifying the value
            range of all the input features x_i

    # Returns
        X: a matrix of sample vectors, where
            the samples are the rows and the
            features are the columns
            ie, its size should be:
              num_samples x (len(weights) - 1)
        y: a vector of num_samples binary labels
    """
    # generate random matrix of the input features
    X = np.random.uniform(low=limits[0], high=limits[1], size=(num_samples, len(weights) - 1))

    # calculate the corresponding score values
    xw = X @ weights[1:] + weights[0]

    # threshold to get the labels
    y = (xw >= 0).astype(np.float64)

    return X, y


def monomial_projection_1d ( X, degree ):
    """
    Map 1d data to an expanded basis of monomials
    up to the given degree. Note that we start
    the terms from degree 0, which is equivalent
    to adding a dummy feature x0 = 1.

    # Arguments
        X: an array of sample data, where rows are samples
            and the single column is the input feature.
        degree: maximum degree of the monomial terms

    # Returns
        Xm: an array of the transformed data, with the
            same number of rows (samples) as X, and
            with degree+1 columns (features):
            1, x, x**2, x**3, ..., x**degree
    """
    Xm = np.zeros((X.shape[0], degree+1))
    Xm[:,0] = 1
    Xm[:,1] = X[:,0]

    for ii in range(2, degree+1):
        Xm[:,ii] = X[:,0] ** ii

    return Xm


def generate_noisy_poly_1d ( num_samples, weights, sigma, limits ):
    """
    Draw samples from a 1D polynomial model with additive
    Gaussian noise.

    # Arguments
        num_samples: number of samples to generate
            (ie, the number of rows in the returned X
            and the length of the returned y)
        weights: vector of the polynomial coefficients
            (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value
            range for the single input dimension x1
        rng: an instance of numpy.random.Generator
            from which to draw random numbers

    # Returns
        X: a matrix of sample inputs, where
            the samples are the rows and the
            single column is the 1D feature x1
            ie, its size should be:
              num_samples x 1
        y: a vector of num_samples output values
    """
    # generate the sample data (1d, but as a matrix)
    X = np.random.uniform(low=limits[0], high=limits[1], size=(num_samples,1))

    # map it to monomials
    Xm = monomial_projection_1d(X, len(weights)-1)

    # compute the clean polynomial values
    y = Xm @ weights

    # add some Gaussian noise
    y = y + np.random.normal(scale=sigma, size=y.shape)

    return X, y