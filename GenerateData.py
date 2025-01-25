import numpy as np

def generate_noisy_linear(num_samples, weights, sigma, limits):
    """
    Draw samples from a linear model with additive Gaussian noise.

    # Arguments
        num_samples: number of samples to generate
        weights: vector of weights
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value range of all the input features x_i

    # Returns
        X: a matrix of sample inputs
        y: a vector output values
    """
    X = np.random.uniform(low=limits[0], high=limits[1], size=(num_samples, len(weights) - 1))

    y = X @ weights[1:] + weights[0]

    y = y + np.random.normal(scale=sigma, size=y.shape)

    return X, y


def generate_linearly_separable(num_samples, weights, limits):
    """
    Draw samples from a binary model with a given linear
    decision boundary.

    # Arguments
        num_samples: number of samples to generate
        weights: vector of weights
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value range of all the input features x_i

    # Returns
        X: a matrix of sample inputs
        y: a vector output values
    """
    X = np.random.uniform(low=limits[0], high=limits[1], size=(num_samples, len(weights) - 1))

    xw = X @ weights[1:] + weights[0]

    y = (xw >= 0).astype(np.float64)

    return X, y


def monomial_projection_1d ( X, degree ):
    """
    Map 1d data to an expanded basis of monomials up to the given degree.

    # Arguments
        X: an array of sample data
        degree: maximum degree of the monomial terms

    # Returns
        Xm: an array of the transformed data
    """
    Xm = np.zeros((X.shape[0], degree+1))
    Xm[:,0] = 1
    Xm[:,1] = X[:,0]

    for ii in range(2, degree+1):
        Xm[:,ii] = X[:,0] ** ii

    return Xm


def generate_noisy_poly_1d ( num_samples, weights, sigma, limits ):
    """
    Draw samples from a 1D polynomial model with additive Gaussian noise.

    # Arguments
        num_samples: number of samples to generate
        weights: vector of the polynomial coefficients (including a bias term at index 0)
        sigma: standard deviation of the additive noise
        limits: a tuple (low, high) specifying the value range for the single input dimension x1

    # Returns
        X: a matrix of sample input
        y: a vector of output values
    """
    X = np.random.uniform(low=limits[0], high=limits[1], size=(num_samples,1))

    # Map X to monomials
    Xm = monomial_projection_1d(X, len(weights)-1)

    # Compute the clean output values
    y = Xm @ weights

    # Add some Gaussian noise
    y = y + np.random.normal(scale=sigma, size=y.shape)

    return X, y


def generate_margined_binary_data ( num_samples, count, limits):
    """
    Draw random samples from a linearly-separable binary model with some non-negligible margin between classes. 

    # Arguments
        num_samples: number of samples to generate
        count: the number of feature dimensions
        limits: a tuple (low, high) specifying the value range of all the features x_i
    # Returns
        X: a matrix of sample vector
        y: a vector of binary labels
    """
    # Generate a lot of points that we need
    X = np.random.uniform(low=limits[0], high=limits[1], size=(num_samples * 2, count))

    # Specify the margin size
    margin = (limits[1] - limits[0])/10

    # Set the boundary
    mid = (limits[0] + limits[1])/2

    # keep points further than margin from this boundary
    X = X[(X[:,0] < (mid-margin)) | (X[:,0] > (mid+margin)),:]
    y = X[:,0] > mid

    return X[:num_samples,:], y[:num_samples]


def generate_binary_nonlinear_2d(num_samples, limits):
    """
    Draw random samples from a binary model that is *not* linearly separable in its 2D feature space.

    # Arguments
    num_samples: number of samples to generate
    limits: a tuple (low, high) specifying the value range of all the features x_i

    # Returns
    X: a matrix of sample vectors
    y: a vector of binary labels
    """
    # Generate random 2D points in the specified range
    X = np.random.uniform(limits[0], limits[1], size=(num_samples, 2))

    # Internal function to generate binary labels
    def binary_decision_stripes(X):
        tmp = X[:, 0] + X[:, 1]
        return (np.floor(tmp / 4) == np.floor((tmp + 2) / 4)).astype(int)

    # Generate labels based on the binary decision function
    y = binary_decision_stripes(X)

    return X, y
