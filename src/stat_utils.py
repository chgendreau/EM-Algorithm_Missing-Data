import numpy as np

def nan_covariance(data): #not sure how to initialise without a function like this, np.cov() does not work well as there is alot of NaN in the matrix

    """
    replaces Nans by column means and calculates empirical covariance
    :param data: data in (n,p) numpy array form, with Nans
    :return: empirical covariance
    """

    n, m = data.shape # this function basicaly computes covariance ignoring the mssing data
    means = np.nanmean(data, axis=0)
    data_filled = np.where(np.isnan(data), means, data)

    return np.cov(data_filled, rowvar=False)

def log_likelihood1(X, mu, Sigma, distribution='gaussian'): 
    """
    Calculates complete (Gaussian by default) loglikelihood
    :param X: (n,p) numpy
    :param mu: (p,) numpy
    :param Sigma: (p,p) numpy
    :return: float loglikelihood
    """
    if distribution == 'gaussian':
        N, p = X.shape
        mu = mu.reshape(1, -1)  # Reshape mu to ensure correct matrix operations
        X_centered = X - mu
        Sigma_inv = np.linalg.inv(Sigma)
        term = np.einsum('ij,ji->i', X_centered, np.dot(Sigma_inv, X_centered.T))
        result = -N/2 * np.log(np.linalg.det(Sigma)) - 0.5 * np.sum(term)
    else:
        raise NotImplementedError
    return result

def regularize_covariance(Sigma, eigenvalue_tol=1e-2, ridge_coefficient=5e-1, verbose=False, version2=False):
    """
    regularizes covariance matrices with very small eigenvalues by adding lambda*In

    :param Sigma: (p,p) numpy array (positive definite covariance matrix, must be possible to do eigenvalue decomposition)

    :param eigenvalue_tol : threshold of the smallest eigenvalue that leads to regularization

    :param ridge_coefficient : the lambda such that lambda*In gets added if the threshold is reached

    :param version2: If "True", this executes a completely different version of the function, and the role of the parameters changes slightly. If the ration of smallest to biggest eigenvalue is smaller than eigenvalue_tol, we add ridge_coefficient times the biggest eigenvalue to the diagonal.
    """

    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    # eigenvectors are in the columns
    p = Sigma.shape[0]

    # I added this block which adds ridge*biggest_eigenvalue*Identity if the smallest eigenvalue is less than tol*biggest eigenvalue
    if version2:
        lambda1 = max(eigenvalues)
        lambdan = min(eigenvalues)
        if verbose:
            print("Smallest | largest  eigenvalue : ", lambdan, lambda1)

        if lambdan / lambda1 < eigenvalue_tol:
            if verbose:
                print("We added :", ridge_coefficient * lambda1, " to the diagonal.")
            Sigma = Sigma + ridge_coefficient * lambda1 * np.eye(p)
        return Sigma

    small_indices = np.where(eigenvalues < eigenvalue_tol)[0]
    if verbose:
        # print("The eigenvectors are the columns of :\n",eigenvectors)
        print("The smallest eigenvalue is :", min(eigenvalues))
        print("The eigenvalues below the tolerance of ", eigenvalue_tol, " are :\n", eigenvalues[small_indices])

    if len(small_indices) > 0:
        Sigma = Sigma + ridge_coefficient * np.eye(p)
        if verbose:
            print("We added ", ridge_coefficient, " to the diagonal.")
    return Sigma

#Perhaps we need to change the name of this function to something more appropriate
def gaussian_MLE(data):
    """
    Returns empirical mean and covariance.
    """
    n = data.shape[0]
    p = data.shape[1]

    means = np.mean(data, axis=0)

    Sigma=np.cov(data,rowvar=False)

    return means, Sigma

