import updated_impyute
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
import torch
from produce_NA import *
import importlib
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Function to generate synthetic data
def generate_synthetic_data(n_samples=100, n_features=5):
    """
    TO DO: add other ways to generate data (e.g. sparse covariance matrix, non-gaussian data, etc.)
    Generates data for a given number of samples and features.
    :param n_samples: number of samples
    :param n_features: number of features
    :return: (n_samples, n_features) numpy array
    """
    mean = np.random.rand(n_features) * 10
    cov = np.random.rand(n_features, n_features)
    cov = np.dot(cov, cov.transpose())  # Ensure the covariance matrix is positive semi-definite
    data = multivariate_normal.rvs(mean=mean, cov=cov, size=n_samples)
    return data, mean, cov



# Updated generate_synthetic_data to take optional mean/covariance arguments instead of simulating randomly
def generate_synthetic_data(n_samples=100, n_features=5,mean=None,cov=None):
    """
    TO DO: add other ways to generate data (e.g. sparse covariance matrix, non-gaussian data, etc.)
    Generates data for a given number of samples and features.
    :param n_samples: number of samples
    :param n_features: number of features
    :param cov: optionally supllied n_features^2 covariance matrix
    :param mean: optionally supply mean; (n_features,) vector
    :return: (n_samples, n_features) numpy array
    """
    if mean is None:
        mean = np.random.rand(n_features) * 10
    if cov is None:
        cov = np.random.rand(n_features, n_features)
        cov = np.dot(cov, cov.transpose())  # Ensure the covariance matrix is positive semi-definite
    data = multivariate_normal.rvs(mean=mean, cov=cov, size=n_samples)
    return data, mean, cov


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


#Old log_likelihood1 function, which is slow and bad
"""
def log_likelihood1(X, mu, Sigma):
    
    Calculates complete Gaussian loglikelihood
    :param X: (n,p) numpy
    :param mu: (p,) numpy
    :param Sigma: (p,p) numpy
    :return: float loglikelihood (or actually (1,1) array technically I think)
    
    N=X.shape[0]
    p=X.shape[1]
    result=-N/2*np.log(np.linalg.det(Sigma))
    X_muT=X-mu.T
    for n in range(N):
        result=result-0.5*(X[n,:]-mu.T).dot(np.linalg.solve(Sigma, X[n,:].T-mu)) #is repetitively summing here bad?
    return result
"""

#New log_likelihood1 function that has x20 better performance
def log_likelihood1(X, mu, Sigma): 
    """
    Calculates complete Gaussian loglikelihood
    :param X: (n,p) numpy
    :param mu: (p,) numpy
    :param Sigma: (p,p) numpy
    :return: float loglikelihood
    """
    N, p = X.shape
    mu = mu.reshape(1, -1)  # Reshape mu to ensure correct matrix operations
    X_centered = X - mu
    Sigma_inv = np.linalg.inv(Sigma)
    term = np.einsum('ij,ji->i', X_centered, np.dot(Sigma_inv, X_centered.T))
    result = -N/2 * np.log(np.linalg.det(Sigma)) - 0.5 * np.sum(term)
    return result


def calculate_xhat(xn, mu, Sigma, missing_indices):

    """
    imputes the missing data in an observation (say, the n-th row of X) by the conditional mean given the observed data, based off of current estimates for mu and Sigma.

    :param xn: (p,) numpy array; observation n, ie n-th row of data matrix X
    :param mu: (p,) numpy array; current estimate
    :param Sigma: (p,p) numpy array; current estimate
    :param missing_indices: The indices where xn has Nans, 1D numpy array (or ORIGINALLY we had Nans)
    :return: xhatn, the numpy vector where nans have been replaced by the conditional mean given the observed data in xn

    """
    p = Sigma.shape[0]

    xhatn = xn.copy()
    observed_indices = np.arange(p)[~np.isin(np.arange(p), missing_indices)]

    Sigma22 = Sigma[missing_indices, :][:, missing_indices]
    Sigma11 = Sigma[observed_indices, :][:, observed_indices]
    Sigma21 = Sigma[missing_indices, :][:, observed_indices]

    mu1 = mu[observed_indices]
    mu2 = mu[missing_indices]
    xhatn_1 = xhatn[observed_indices]
    mu2_conditional = mu2 + Sigma21.dot(np.linalg.solve(Sigma11, xhatn_1 - mu1))
    xhatn[missing_indices,] = mu2_conditional
    # this doesnt seem to work
    # reason: Array is passed as a view, potential danger!
    # hmm still doesnt work wtf
    # Its because a numpy array of ints rounds floats that are added to it wtfffff, fixed now

    return xhatn

"""
def regularize_covariance(Sigma, eigenvalue_tol=1e-3, ridge_coefficient=1e-1, verbose=False):
    
    regularizes covariance matrices with very small eigenvalues by adding lambda*In

    Sigma: (p,p) numpy array (positive definite covariance matrix, must be possible to do eigenvalue decomposition)

    eigenvalue_tol : threshold of the smallest eigenvalue that leads to regularization

    ridge_coefficient : the lambda such that lambda*In gets added if the threshold is reached
    

    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    # eigenvectors are in the columns
    p = Sigma.shape[0]

    small_indices = np.where(eigenvalues < eigenvalue_tol)
    if verbose:
        print("The eigenvectors are the columns of :\n", eigenvectors)
        print("The smallest eigenvalue is :", min(eigenvalues))
        print("The eigenvalues below the tolerance of ", eigenvalue_tol, " are :\n", eigenvalues[small_indices])
        print("We added ", ridge_coefficient, " to the diagonal.")
    if len(small_indices) > 0:
        Sigma = Sigma + ridge_coefficient * np.eye(p)
    return Sigma

"""

#Emil updated the function, adding an (optional) second version
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

def calculate_C(Sigma, missing_indices, verbose=False):
    """
    Say we have X=(X1,X2) a random vector.
    :param Sigma: (p,p) numpy, current estimate of Sigma
    :param missing_indices: numpy 1D array; indices of the unobserved coordinates of the p-dimensional gaussian random vector ORIGINALLY
    :param verbose: Whether to print fancy stuff
    :return: Conditional covariance of the random vector X=(X1,X2) given that X1=x1 is observed. This means that some pairwise covariances are fixed at 0.
    """
    p = Sigma.shape[0]
    m = missing_indices.shape[0]
    if verbose:
        print("Original unconditional covariance : \n", Sigma)
    observed_indices = np.arange(p)[~np.isin(np.arange(p), missing_indices)]
    appended_indices = np.append(observed_indices, missing_indices)
    Sigma = (Sigma[appended_indices, :])[:, appended_indices]

    if verbose:
        print("Unconditional covariance rearranged so that missing indices are at the bottom : \n", Sigma)
    Sigma11 = Sigma[0:(p - m), 0:(p - m)]
    Sigma21 = Sigma[(p - m):p, 0:(p - m)]
    Sigma22 = Sigma[(p - m):p, (p - m):p]
    Sigma22_conditional = Sigma22 - Sigma21.dot(np.linalg.solve(Sigma11, Sigma21.T))

    reverse_permutation = np.argsort(appended_indices)
    result = np.full((p, p), 0)
    result[(p - m):p, (p - m):p] = Sigma22_conditional
    if verbose:
        print("The conditional covariance when ordered is : \n", result)
    result = (result[reverse_permutation, :])[:, reverse_permutation]
    if verbose:
        print("After permuting back we get : \n", result)
    return (result)



#modif: Emil added a tiny verbose option
def em_algorithm(data, max_iter=100, tol=1e-6, eigenvalue_tol=1e-3, ridge_coefficient=1e-1,regularize_version2=False,verbose=False):
    """
    Our EM Algorithm implementation for p-variate Gaussian data with missing entries. Performs imputation and estimates parameters.
    :param data: (n,p) numpy array. Each row represents realization of p-variate Gaussian, and may contain NAs
    :param max_iter: max iterations to run
    :param tol: convergence criterion for the improvement in likelihood (technically, this should be used with observed likelihood, I think. But we used it with complete likelihood for the time being. It still works well.
    :param eigenvalue_tol: For regularize_covariance()
    :param ridge_coefficient: For regularize_covariance()
    return: The final imputed data (n,p) numpy array; the estimated mean mu; the estimated covariance Sigma
    """
    n, p = data.shape

    # Initialize mean and covariance estimates
    means = np.nanmean(data, axis=0)
    covariance = nan_covariance(data)
    covariance = regularize_covariance(covariance, eigenvalue_tol, ridge_coefficient,version2=regularize_version2)

    # Create an array to hold imputed data
    imputed_data = np.where(np.isnan(data), np.nanmean(data, axis=0), data)
    # imputed_data = np.where(np.isnan(data), None, data)

    old_log_likelihood = log_likelihood1(imputed_data, means, covariance)
    # print(old_log_likelihood)

    for iteration in range(max_iter):
        # E-step: Estimate missing values
        for i in range(n):
            missing = np.where(np.isnan(data[i]))[0]
            if np.isnan(data[i]).any():
                imputed_data[i,] = calculate_xhat(imputed_data[i,], means, covariance,
                                                  missing)  # this function works with Nans in the missing entries too, why do we impute?

        # M-step: Update mean and covariance estimates
        means = np.mean(imputed_data, axis=0)
        new_covariance = np.full((p, p), 0)

        # Add the conditional covariance for missing data
        for i in range(n):
            missing = np.where(np.isnan(data[i]))[0]
            # if missing.any(): #with this condition, are you not skipping xhat-mu *(xhat-mu).T when there is no missing data? I think so...
            new_covariance = new_covariance + calculate_C(covariance, missing, verbose=False) + (
                        imputed_data[i,] - means).reshape(-1, 1) @ np.transpose(
                (imputed_data[i,] - means).reshape(-1, 1))

        covariance = new_covariance / n
        covariance = regularize_covariance(covariance)
        # print(covariance)
        # Convergence test based on log likelihood
        new_log_likelihood = log_likelihood1(imputed_data, means, covariance)
        # print(new_log_likelihood)
        difference = new_log_likelihood - old_log_likelihood
        if verbose:
            print("The new log likelihood is :", new_log_likelihood, "  Difference of : ",difference)

        if np.abs(
                difference) < tol:  # absolute value not necessary and potentially undesirable, as in theory should always be positive
            if verbose:
                print("Convergence achieved! \n")
            break
        old_log_likelihood = new_log_likelihood

    return imputed_data, means, covariance






import random


def generate_sparse_covariance(dim, sparsity_probability=0.15):
    """
    generates a covariance matrix with a lot of zeros, by generating a Cholesky factor with zeros and doing L*Lt

    :param dim: dimension p of covariance matrix desired

    :param sparsity_probability: between 0 and 1, 0 is diagonal (very sparse), 1 has no zeros. Probability of a lower-triangular element of Cholesky factor L of being nonzero

    """

    mat = np.full((dim, dim), 0).astype(float)
    a = 3  # sample between 0 and 3
    diagonal = a * np.random.random_sample(size=dim) + 0.1
    np.fill_diagonal(mat, diagonal)
    for i in range(dim):
        for j in range(i):
            p = random.uniform(0, 1)
            # print(p)
            if p < sparsity_probability:
                mat[i, j] = (random.uniform(-4, 4))
    covariance = mat.dot(mat.T)
    # print(covariance)
    return covariance


# Imagine you have a dim*dim pixels in your square picture. Say dim=16. Each pixel has a random value, that represents say, color, which is a Gaussian.
# The covariance between two pixels is proportional to their euclidean distance. Make a 16^2*16^2 matrix D where dij encodes the distance between pixels i and j
# This means we have a d-dimensional Gaussian with d=16^2 (quite big) whose covariance is D (or a function thereof)

def generate_2Dgrid_correlation(dim, h=1.5):
    """
    Imagine you have a dim*dim pixels in your square picture. Say dim=16. Each pixel has a random value, that represents say, color, which is a Gaussian.
    The covariance between two pixels is proportional to their euclidean distance. Make a 16^2*16^2 matrix D where dij encodes the distance between pixels i and j
    This means we have a d-dimensional Gaussian with d=16^2 (quite big) whose covariance is D (or a function thereof)
    This function returns a correlation matrix of such a grid.


    :param dim: dimension of the GRID (the correlation matrix will have dim^2*dim^2 dimensions)

    :param h: bandwidth parameter. The correlation between pixel z1 and pixel z3 is exp(-||z1-z3||/h) (no square)

    return: correlation matrix (dim^2*dim^2 numpy array)
    """
    seq = np.arange(dim)
    # print(seq)
    pairs = np.full((dim, dim, 2), 0)

    for i in seq:
        for j in seq:
            pairs[i, j, :] = np.array([seq[i], seq[j]])
    pairs = pairs.reshape(dim ** 2, 2)
    # print("pairs:",pairs)
    distances = np.full((dim ** 2, dim ** 2), 0).astype(float)  # very important
    for i in range(distances.shape[0]):
        for j in range(distances.shape[0]):
            # print("The two vectors:",pairs[i],pairs[j],"the norm of their difference: ",np.linalg.norm(pairs[i]-pairs[j]))
            distances[i, j] = np.linalg.norm(pairs[i] - pairs[j])
    result = np.exp(-distances / h)  # this function is a bit arbitrary, we could choose the squared norm or whatever
    # print("determinant:",np.linalg.det(result))
    return result


# we need a function which, from imputed data, makes parameter estimation.
def gaussian_MLE(data):
    """
    Returns empirical mean and covariance.
    """
    n = data.shape[0]
    p = data.shape[1]

    means = np.mean(data, axis=0)

    Sigma=np.cov(data,rowvar=False)

    return means, Sigma







def plot_MCAR(data, missing_data_percentages=np.arange(5, 65, 5)):
    # Range of missing data percentages
    #missing_data_percentages = np.arange(5, 65, 5)
    mse_values = []

    # Loop over different percentages of missing data
    for p_miss in missing_data_percentages:
        # Simulate missing data
        missing_data_info = produce_NA(data, p_miss / 100.0)
        X_incomp = missing_data_info['X_incomp'].numpy()

        # Apply EM algorithm
        imputed_data, _, _ = em_algorithm(X_incomp)
        #imputed_data = updated_impyute.em(X_incomp, eps=0.1)

        # Calculate MSE
        mse = mean_squared_error(data, imputed_data)
        mse_values.append(mse)

    # Plotting the results
    plt.plot(missing_data_percentages, mse_values, marker='o')
    plt.xlabel('Percentage of Missing Data (%)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE of Imputed Data vs. Percentage of Missing Data')
    plt.grid(True)
    plt.show()

def plot_MAR(data, missing_data_percentages = np.arange(5, 65, 5)):
    # Range of missing data percentages
    mse_values = []

    # Loop over different percentages of missing data
    for p_miss in missing_data_percentages:
        # Simulate missing data
        p_obs = 0.5
        missing_data_info = produce_NA(data, p_miss / 100.0, mecha='MAR', p_obs=p_obs, opt="quantile")
        X_incomp = missing_data_info['X_incomp'].numpy()

        # Apply EM algorithm
        imputed_data, _, _ = em_algorithm(X_incomp)
        #imputed_data = updated_impyute.em(X_incomp, eps=0.1)

        # Calculate MSE
        mse = mean_squared_error(data, imputed_data)
        mse_values.append(mse)

    # Plotting the results
    plt.plot(missing_data_percentages, mse_values, marker='o')
    plt.xlabel('Percentage of Missing Data (%)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE of Imputed Data vs. Percentage of Missing Data')
    plt.grid(True)
    plt.show()

def plot_MNAR(data, missing_data_percentages = np.arange(5, 65, 5)):
    mse_values = []

    # Loop over different percentages of missing data
    for p_miss in missing_data_percentages:
        # Simulate missing data
        p_obs = 0.2
        q = 0.7
        missing_data_info = produce_NA(data, p_miss / 100.0, mecha='MNAR', p_obs=p_obs, q=q, opt="logistic")
        X_incomp = missing_data_info['X_incomp'].numpy()

        # Apply EM algorithm
        imputed_data, _, _ = em_algorithm(X_incomp)
        #imputed_data = updated_impyute.em(X_incomp, eps=0.1)

        # Calculate MSE
        mse = mean_squared_error(data, imputed_data)
        mse_values.append(mse)

    # Plotting the results
    plt.plot(missing_data_percentages, mse_values, marker='o')
    plt.xlabel('Percentage of Missing Data (%)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE of Imputed Data vs. Percentage of Missing Data')
    plt.grid(True)
    plt.show()


def plot_combined(data, missing_data_percentages=np.arange(5, 65, 5), em_iterations=5, opt="logistic"):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    methods = ['EM', 'Median', 'KNN', 'Iterative']
    p_obs = 0.2

    for i, mechanism in enumerate(['MCAR', 'MAR', 'MNAR']):
        mse_values = {method: [] for method in methods}

        for p_miss in missing_data_percentages:
            mse_agg = {method: [] for method in methods}

            for _ in range(em_iterations):
                # Generate missing data based on the mechanism
                if mechanism == 'MCAR':
                    missing_data_info = produce_NA(data, p_miss / 100.0, mecha=mechanism)
                elif mechanism == 'MAR':
                    missing_data_info = produce_NA(data, (p_miss / 100.0)/(1-p_obs), mecha=mechanism, p_obs=p_obs)
                elif mechanism == 'MNAR':
                    missing_data_info = produce_NA(data, (p_miss / 100.0)/(1-p_obs), mecha=mechanism, p_obs=p_obs, q=0.7, opt=opt)

                X_incomp = missing_data_info['X_incomp'].numpy()

                # EM Algorithm
                imputed_data_em, _, _ = em_algorithm(X_incomp)
                mse_agg['EM'].append(mean_squared_error(data, imputed_data_em))

                # Median Imputation
                imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
                imputed_data_median = imp_median.fit_transform(X_incomp)
                mse_agg['Median'].append(mean_squared_error(data, imputed_data_median))

                # KNN Imputation
                imp_knn = KNNImputer(n_neighbors=5)
                imputed_data_knn = imp_knn.fit_transform(X_incomp)
                mse_agg['KNN'].append(mean_squared_error(data, imputed_data_knn))

                # Iterative Imputer
                imp_iterative = IterativeImputer(max_iter=100)
                imputed_data_iterative = imp_iterative.fit_transform(X_incomp)
                mse_agg['Iterative'].append(mean_squared_error(data, imputed_data_iterative))

            # Averaging over iterations
            for method in methods:
                mse_values[method].append(np.mean(mse_agg[method]))

        # Plot results
        for method in methods:
            axs[i].plot(missing_data_percentages, mse_values[method], marker='o', label=method)
        axs[i].set_title(mechanism)
        axs[i].set_xlabel('Percentage of Missing Data (%)')
        axs[i].grid(True)

    axs[0].set_ylabel('Average Mean Squared Error (MSE)')
    axs[0].legend()
    fig.suptitle('Comparison of MSE for Different Missing Data Mechanisms')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_all_differences_combined(data, true_mean, true_cov, missing_data_percentages=np.arange(5, 65, 5), em_iterations=5, opt="logistic",verbose=False):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharey='row')
    
    methods = ['EM', 'Median', 'KNN', 'Iterative']  # Define methods list
    p_obs = 0.2
    for i, mechanism in enumerate(['MCAR', 'MAR', 'MNAR']):
        if verbose:
            print("Calculating for mechanism : ",mechanism)
        mean_diffs = {method: [] for method in methods}
        cov_diffs = {method: [] for method in methods}

        for p_miss in missing_data_percentages:
            if verbose:
                print("Calculating for",p_miss,"% of missing data.")
            for method in methods:
                mean_diff_agg = []
                cov_diff_agg = []

                for _ in range(em_iterations):
                    if mechanism == 'MCAR':
                        missing_data_info = produce_NA(data, p_miss / 100.0, mecha=mechanism)
                    elif mechanism == 'MAR':
                        missing_data_info = produce_NA(data, (p_miss / 100.0)/(1-p_obs), mecha=mechanism, p_obs=p_obs)
                    elif mechanism == 'MNAR':
                        missing_data_info = produce_NA(data, (p_miss / 100.0)/(1-p_obs), mecha=mechanism, p_obs=p_obs, opt = opt)

                    X_incomp = missing_data_info['X_incomp'].numpy()

                    # Choose imputation method
                    if method == 'EM':
                        imputed_data, estimated_mean, estimated_cov = em_algorithm(X_incomp)
                    else:
                        if method == 'Median':
                            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
                        elif method == 'KNN':
                            imputer = KNNImputer(n_neighbors=5)
                        elif method == 'Iterative':
                            imputer = IterativeImputer(max_iter=100)
                        
                        imputed_data = imputer.fit_transform(X_incomp)
                        estimated_mean, estimated_cov = gaussian_MLE(imputed_data)

                    # Calculate differences
                    mean_diff = np.linalg.norm(true_mean - estimated_mean) / (np.linalg.norm(true_mean))
                    cov_diff = np.linalg.norm(true_cov - estimated_cov, ord='fro') / (np.linalg.norm(true_cov, ord='fro'))
                    mean_diff_agg.append(mean_diff)
                    cov_diff_agg.append(cov_diff)

                # Averaging over iterations
                mean_diffs[method].append(np.mean(mean_diff_agg))
                cov_diffs[method].append(np.mean(cov_diff_agg))

        # Plot mean differences
        for method in methods:
            axs[0, i].plot(missing_data_percentages, mean_diffs[method], marker='o', label=method)
            axs[0, i].set_title(f'{mechanism} - Mean Difference')
            axs[0, i].set_xlabel('Percentage of Missing Data (%)')
            axs[0, i].set_ylabel('Average RMSE in estimation of mean')
            axs[0, i].grid(True)
            axs[0, i].legend()

        # Plot covariance differences
        for method in methods:
            axs[1, i].plot(missing_data_percentages, cov_diffs[method], marker='o', label=method, linestyle='--')
            axs[1, i].set_title(f'{mechanism} - Covariance Difference')
            axs[1, i].set_xlabel('Percentage of Missing Data (%)')
            axs[1, i].set_ylabel('Average Frobenius Norm of Covariance Difference')
            axs[1, i].grid(True)
            axs[1, i].legend()

    plt.suptitle('Comparison of True and Estimated Parameters Across Different Missing Data Mechanisms')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


#-------------------------------------------
#Emil did some of George's work twice and made his own functions to plot, here they are, so they don't get lost:
#-------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def plot_imputation_methods(data,mecha="MCAR",filename=None):
    """
    Function which generates the three graphs in test new impyute for different missingness patterns.

    :param data: (n,p) matrix with p-dimensional gaussians in rows (no NAs). Values will be removed by produce_NA function, according to specified pattern of missingness.
    """


    # Range of missing data percentages
    missing_data_percentages = np.arange(5, 75, 5)
    methods = ['EM', 'Median', 'KNN', 'Iterative']
    mse_values = {method: [] for method in methods}

    # Loop over different percentages of missing data
    for p_miss in missing_data_percentages:
        print("Calculating the values for ",p_miss,"% of missing data.")
        p_obs = 0.2
        if mecha=="MNAR":
            missing_data_info = produce_NA(data, (p_miss / 100.0)/(1-p_obs), mecha='MNAR', p_obs=p_obs, opt="logistic")
        elif mecha=="MAR":
            missing_data_info = produce_NA(data, (p_miss / 100.0)/(1-p_obs), mecha='MAR', p_obs=p_obs)
        elif mecha=="MCAR":
            missing_data_info = produce_NA(data, p_miss / 100.0)
        else:
            print("Invalid mecha entered.")
        X_incomp = missing_data_info['X_incomp'].numpy()

        # EM Algorithm
        imputed_data_em, _, _ = em_algorithm(X_incomp)
        mse_values['EM'].append(mean_squared_error(data, imputed_data_em))

        # Median Imputation
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        imputed_data_median = imp_median.fit_transform(X_incomp)
        mse_values['Median'].append(mean_squared_error(data, imputed_data_median))

        # KNN Imputation
        imp_knn = KNNImputer(n_neighbors=5)
        imputed_data_knn = imp_knn.fit_transform(X_incomp)
        mse_values['KNN'].append(mean_squared_error(data, imputed_data_knn))

        # Iterative Imputer (Multivariate feature imputation)
        imp_iterative = IterativeImputer(max_iter=100)
        imputed_data_iterative = imp_iterative.fit_transform(X_incomp)
        mse_values['Iterative'].append(mean_squared_error(data, imputed_data_iterative))

    # Plotting the results
    plt.figure(figsize=(12, 8))
    for method in methods:
        plt.plot(missing_data_percentages, mse_values[method], marker='o', label=method)

    plt.xlabel('Percentage of Missing Data (%)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of MSE for Different Imputation Methods')
    plt.legend()
    plt.grid(True)
    if not filename is None: #save the figure
        plt.savefig(filename)
    plt.show()


def compare_imputation_methods(data, mecha="MCAR"):
    """
    Evolution of plot_imputation_methods: doesnt directly do the plot, returns the data vectors instead. Includes the MSE for the imputed data, as well as the mean and covariance calculated at every point.
    returns: missing_data_percentages, mse_values, mean_values, covariance_values
    missing_data_percentages: list 5,10,15...
    mse_values: dict with keys "MAR" and so on. mse_values["MAR"] is a list which contains the MSE of the imputed data for each missing data percentage
    mean_values,covariance values: Same as mse_values, except containing mean and covariance for each percentage point
    """
    # Range of missing data percentages
    missing_data_percentages = np.arange(5, 75, 5)
    methods = ['EM', 'Median', 'KNN', 'Iterative']

    mse_values = {method: [] for method in methods}
    mean_values = {method: [] for method in methods}
    covariance_values = {method: [] for method in methods}

    # Loop over different percentages of missing data
    for p_miss in missing_data_percentages:
        print("Calculating the values for ", p_miss, "% of missing data.")
        p_obs = 0.2
        if mecha == "MNAR":
            missing_data_info = produce_NA(data, (p_miss / 100.0)/(1-p_obs), mecha='MNAR', p_obs=p_obs, opt="logistic")
        elif mecha == "MAR":
            p_obs = 0.5
            missing_data_info = produce_NA(data, (p_miss / 100.0)/(1-p_obs), mecha='MAR', p_obs=p_obs)
        elif mecha == "MCAR":
            missing_data_info = produce_NA(data, p_miss / 100.0)
        else:
            print("Invalid mecha entered.")
        X_incomp = missing_data_info['X_incomp'].numpy()

        # EM Algorithm
        imputed_data_em, means, covariance = em_algorithm(X_incomp)
        mse_values['EM'].append(mean_squared_error(data, imputed_data_em))
        mean_values['EM'].append(means)
        covariance_values['EM'].append(covariance)

        # Median Imputation
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        imputed_data_median = imp_median.fit_transform(X_incomp)
        mse_values['Median'].append(mean_squared_error(data, imputed_data_median))

        # get the parameter estimates from the imputed data
        means, covariance = gaussian_MLE(imputed_data_median)
        mean_values['Median'].append(means)
        covariance_values['Median'].append(covariance)

        # KNN Imputation
        imp_knn = KNNImputer(n_neighbors=5)
        imputed_data_knn = imp_knn.fit_transform(X_incomp)
        mse_values['KNN'].append(mean_squared_error(data, imputed_data_knn))

        # get the parameter estimates from the imputed data
        means, covariance = gaussian_MLE(imputed_data_knn)
        mean_values['KNN'].append(means)
        covariance_values['KNN'].append(covariance)

        # Iterative Imputer (Multivariate feature imputation)
        imp_iterative = IterativeImputer(max_iter=100)
        imputed_data_iterative = imp_iterative.fit_transform(X_incomp)
        mse_values['Iterative'].append(mean_squared_error(data, imputed_data_iterative))

        # get the parameter estimates from the imputed data
        means, covariance = gaussian_MLE(imputed_data_iterative)
        mean_values['Iterative'].append(means)
        covariance_values['Iterative'].append(covariance)

    return missing_data_percentages, mse_values, mean_values, covariance_values

