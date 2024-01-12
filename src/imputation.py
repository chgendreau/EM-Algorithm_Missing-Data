import numpy as np
from stat_utils import *
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from data_generation import *

########## Functions for em algoritm ##########
def calculate_xhat(xn, mu, Sigma, missing_indices):
    """
    Imputes the missing data in an observation (say, the n-th row of X) by the conditional mean given the observed data, based off of current estimates for mu and Sigma.

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

    return xhatn

def calculate_C(Sigma, missing_indices, verbose=False):
    """
    Computes conditional covariance of the random vector X=(X1,X2) given that X1=x1 is observed. This means that some pairwise covariances are fixed at 0.Say we have X=(X1,X2) a random vector.
    :param Sigma: (p,p) numpy, current estimate of Sigma
    :param missing_indices: numpy 1D array; indices of the unobserved coordinates of the p-dimensional gaussian random vector ORIGINALLY
    :param verbose: Whether to print fancy stuff
    :return: Conditional covariance
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

def em_algorithm(data, max_iter=100, tol=1e-6, eigenvalue_tol=1e-3, ridge_coefficient=1e-1,regularize_version2=False,verbose=False):
    """
    EM Algorithm implementation for p-variate Gaussian data with missing entries. Performs imputation and estimates parameters.
    
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
    iteration = 0
    difference = 10 #arbitrary value > tol
    while iteration < max_iter and np.abs(difference) > tol:
        iteration += 1
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
                        imputed_data[i,] - means).reshape(-1, 1) @ np.transpose((imputed_data[i,] - means).reshape(-1, 1))

        covariance = new_covariance / n
        covariance = regularize_covariance(covariance)

        # Convergence test based on log likelihood
        new_log_likelihood = log_likelihood1(imputed_data, means, covariance)
        difference = new_log_likelihood - old_log_likelihood
        if verbose:
            #print("The new log likelihood is :", new_log_likelihood, "  Difference of : ",difference)
            if np.abs(difference) < tol:  # absolute value not necessary and potentially undesirable, as in theory should always be positive
                print("EM algorithm: Convergence achieved! \n")

        old_log_likelihood = new_log_likelihood
    #print('EM iterations completed: ', iteration)

    return imputed_data, means, covariance




########## Functions for imputation ##########
def impute_data(X_incomp, method = 'EM', max_iter =100, em_tol = 1e-6, em_eigenvalue_tol = 1e-3, em_ridge_coefficient = 1e-1, em_regularize_version2 = False, verbose = False, KNN_neighbours = 5):
    """
    Imputes missing values in a dataset using the EM algorithm.
    
    :param X_incomp: (n,p) numpy array. Each colum represents a variable and each row an observation. May contain NAs.
    :param method: 'EM', 'Median', 'KNN', 'Iterative' (i.e. MICE)
    :param max_iter: Option for EM and KNN: max iterations to run
    :param em_tol: Option for EM: convergence criterion for the improvement in likelihood (technically, this should be used with observed likelihood, I think. But we used it with complete likelihood for the time being. It still works well.
    :param em_eigenvalue_tol: Option for EM, for regularize_covariance() (see stat_utils.py)
    :param em_ridge_coefficient: Option for EM for regularize_covariance() (see stat_utils.py)
    :param em_regularize_version2: Option for EM for regularize_covariance() (see stat_utils.py)
    :param verbose: Option for EM: Whether to print fancy stuff
    :param KNN_neighbours: Option for KNN: Number of neighbours to use
    :return: The final imputed data (n,p) numpy array; the estimated mean mu; the estimated covariance Sigma
    """
    if method == 'EM':
        imp_data, est_mean, est_cov = em_algorithm(X_incomp, max_iter=max_iter, tol=em_tol, eigenvalue_tol=em_eigenvalue_tol, ridge_coefficient=em_ridge_coefficient,regularize_version2=em_regularize_version2,verbose=verbose)
    else:
        if method == 'Median':
            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        elif method == 'KNN':
            imputer = KNNImputer(n_neighbors=KNN_neighbours)
        elif method == 'Iterative' or method == 'MICE':
            imputer = IterativeImputer(max_iter=max_iter, initial_strategy='median')
        else:
            raise ValueError('Invalid imputation method: ' + method)
        
        imp_data = imputer.fit_transform(X_incomp)
        est_mean, est_cov = gaussian_MLE(imp_data)   


    return (imp_data, est_mean, est_cov)



def compute_mses(complete_data, mechanism, methods: list, missing_data_percentages, real_mean = None, real_cov = None, p_obs = 0.2, opt = 'logistic', q = 0.5, verbose =False):
    """
    Compute rescaled MSEs (rescaled by the square of the real 2-norm) for different imputation methods and a given missing data mechanism for a range of missing data percentages.
    The function first amputes the data according to the specified mechanism and percentage, then imputes it using each method, and finally computes the MSE between the original complete data and the imputed data.
    Parameters:
    - complete_data: A np.ndarray representing the complete dataset.
    - mechanism: A string representing the missing data mechanism to be used ('MCAR', 'MAR' or 'MNAR').
    - methods: A list of strings representing the imputation methods to be used (e.g., ['EM', 'Median', 'KNN', 'Iterative']).
    - missing_data_percentages: A numpy array or list of percentages (values between 0 and 100) indicating the proportion of data to be amputed.
    - real_mean: A np.ndarray representing the real mean of the dataset.
    - real_cov: A np.ndarray representing the real covariance matrix of the dataset.
    Return:
    - Dictionary with rescaled MSE values for each missing data percentage for each method.
    """
   
    mse_values_data = {method: [] for method in methods}
    error_means = {method: [] for method in methods}
    error_covs = {method: [] for method in methods}

    #print warning if the missing percentages are not in the right range
    if np.any(missing_data_percentages < 1) :
        print("Warning: missing data percentages should be between 0 and 100 and not between 0 and 1")
    missing_data_proba = missing_data_percentages/100.0
    for p_miss in missing_data_proba:
        if verbose:
            print(mechanism + " - Missing data percentage: ", p_miss)
        X_incomp = generate_incomplete_data(complete_data, miss_proba=p_miss, mecha=mechanism, p_obs = 0.2, opt = 'logistic', q = 0.5)
        for method in methods:
            imputed_data, est_mean, est_cov = impute_data(X_incomp, method)
            #MSE of data
            mse_data = mean_squared_error(complete_data, imputed_data)/mean_squared_error(complete_data, np.zeros(complete_data.shape))
            mse_values_data[method].append(mse_data)
            #MSE of means
            if real_mean is not None:
                mse_mean = mean_squared_error(est_mean, real_mean)/mean_squared_error(real_mean, np.zeros(real_mean.shape))
                error_means[method].append(mse_mean)
            else:
                error_means[method].append(None)
            #MSE of covs
            if real_cov is not None:
                mse_cov = mean_squared_error(est_cov, real_cov)/mean_squared_error(real_cov, np.zeros(real_cov.shape))
                error_covs[method].append(mse_cov)
            else:
                error_covs[method].append(None)

    return mse_values_data, error_means, error_covs

def compute_all_mses(complete_data, mechanisms = ['MCAR', 'MAR', 'MNAR'], methods =  ['EM', 'Median', 'KNN', 'Iterative'], 
                     missing_data_percentages = np.arange(5, 65, 5), real_mean = None, real_cov = None, p_obs = 0.2, 
                     opt = 'logistic', q = 0.5, em_iterations = 5, verbose = False):
    
    mse_values_agg = {mechanism: {method: [] for method in methods} for mechanism in mechanisms}
    mse_means_agg = {mechanism: {method: [] for method in methods} for mechanism in mechanisms}
    mse_covs_agg = {mechanism: {method: [] for method in methods} for mechanism in mechanisms}

    mse_values_avg = {mechanism: [] for mechanism in mechanisms}
    mse_means_avg = {mechanism: [] for mechanism in mechanisms}
    mse_covs_avg = {mechanism: [] for mechanism in mechanisms}

    for i, mechanism in enumerate(mechanisms):
        if verbose:
            print("Mechanism: ", mechanism)
        for k in range(em_iterations):
            mse_data, mse_mean, mse_cov = compute_mses(complete_data, mechanism, methods, real_mean=real_mean, real_cov=real_cov, missing_data_percentages=missing_data_percentages, p_obs=p_obs, opt=opt, q=q)
            for method in methods:
                mse_values_agg[mechanism][method].append(mse_data[method])
                mse_means_agg[mechanism][method].append(mse_mean[method])
                mse_covs_agg[mechanism][method].append(mse_cov[method])

        mse_values_avg[mechanism] = {method: np.mean(mse_values_agg[mechanism][method], axis=0) for method in methods}
        mse_means_avg[mechanism] = {method: np.mean(mse_means_agg[mechanism][method], axis=0) for method in methods}
        mse_covs_avg[mechanism] = {method: np.mean(mse_covs_agg[mechanism][method], axis=0) for method in methods}
    
    return mse_values_avg, mse_means_avg, mse_covs_avg


