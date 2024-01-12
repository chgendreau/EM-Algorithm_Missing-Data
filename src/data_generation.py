import numpy as np
import scipy.stats as stats
import random
from scipy.stats import shapiro, normaltest
from produce_NA import *

#might need to change the name to clarify the normal distribution
#might be good to change the covariance matrix which has only positive entries and close to zero entries
def generate_synthetic_data(n_samples=100, n_features=5,mean=None,cov=None):
    """
    Generate synthetic data from a multivariate normal distribution.
    If no mean and covariance matrix are supplied, they are randomly generated.

    :param n_samples: number of samples
    :param n_features: number of features
    :param cov: optionally supllied n_features^2 covariance matrix
    :param mean: optionally supply mean; (n_features,) vector
    :return: (n_samples, n_features) numpy array
    """
    if mean is None:
        mean = np.random.randn(n_features) * 10
    if cov is None:
        cov = np.random.randn(n_features, n_features)
        cov = np.dot(cov, cov.transpose())  # Ensure the covariance matrix is positive semi-definite
    data = stats.multivariate_normal.rvs(mean=mean, cov=cov, size=n_samples)
    return data, mean, cov

def generate_sparse_covariance(dim, sparsity_probability=0.15):
    """
    Generates a covariance matrix with a lot of zeros, by generating a Cholesky factor with zeros and doing L*Lt
    ***
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
            if p < sparsity_probability:
                mat[i, j] = (random.uniform(-4, 4))
    covariance = mat.dot(mat.T)
    return covariance

def generate_2Dgrid_correlation(dim, h=1.5):
    """
    Generates a correlation matrix for a 2D grid of pixels, where the correlation between two pixels is a function of their euclidean distance

    :param dim: dimension of the GRID (the correlation matrix will have dim^2*dim^2 dimensions)
    :param h: bandwidth parameter. The correlation between pixel z1 and pixel z3 is exp(-||z1-z3||/h) (no square)
    :return: correlation matrix (dim^2*dim^2 numpy array)

    Details:
    Imagine you have a dim*dim pixels in your square picture. Say dim=16. Each pixel has a random value, that represents say, color, which is a Gaussian.
    The covariance between two pixels is proportional to their euclidean distance. Make a 16^2*16^2 matrix D where dij encodes the distance between pixels i and j
    This means we have a d-dimensional Gaussian with d=16^2 (quite big) whose covariance is D (or a function thereof)
    This function returns a correlation matrix of such a grid.
    """
    seq = np.arange(dim)
    pairs = np.full((dim, dim, 2), 0)

    for i in seq:
        for j in seq:
            pairs[i, j, :] = np.array([seq[i], seq[j]])
    pairs = pairs.reshape(dim ** 2, 2)
    # print("pairs:",pairs)
    distances = np.full((dim ** 2, dim ** 2), 0).astype(float)  # very important
    for i in range(distances.shape[0]):
        for j in range(distances.shape[0]):
            distances[i, j] = np.linalg.norm(pairs[i] - pairs[j])
    result = np.exp(-distances / h)  # this function is a bit arbitrary, we could choose the squared norm or whatever
    return result

def generate_student_t(df, mean, scale_matrix, n_samples=100, quasi_normal = False):
    """
    Generate data from a multivariate Student-t distribution.
    If quasi_normal is true, it tries to generate data that is quasi-normal (i.e. passes normality tests)

    df: Degrees of freedom for the Student-t distribution.
    mean: Mean vector for the Student-t distribution.
    scale_matrix: scale matrix for the Student-t distribution (not equal to covariance).
    n: Number of samples to generate.
    quasi_normal: If True, tries to generate quasi-normal data.
    return: A numpy ndarray of generated data
    """
    if quasi_normal:   
        iter = 0
        seed = np.random.get_state()[1][0]
        while iter < 100:
            iter += 1
            np.random.seed(seed+iter) #changing seed until we get a quasi_normal sample
            X = stats.multivariate_t.rvs(df=df, loc=mean, shape=scale_matrix, size=n_samples)
            _, p_value_shapiro = shapiro(X)
            _, p_value_normaltest = normaltest(X)

            if p_value_shapiro > 0.05 and (p_value_normaltest > 0.05).all():
                print('Normality not rejected')
                break
        print("Normality tests for X: Shapiro p-value = {}, Normaltest p-value = {}".format(p_value_shapiro, p_value_normaltest))
        np.random.seed(seed) #setting old seed back
        if iter == 100:
            print("Warning: could not generate quasi-normal data")
    else:
        X = stats.multivariate_t.rvs(df=df, loc=mean, shape=scale_matrix, size=n_samples)
    return X

#generation of incomplete data
def generate_incomplete_data(data, miss_proba = 0.3, mecha = 'MCAR', p_obs = 0.2, opt = 'logistic', q = 0.5):
    """
    Generate incomplete data from a complete dataset.
    :param data: (n_samples, n_features) numpy array of complete data
    :param miss_proba: proba of missing data (in total, not per variable)
    :param mecha: missing data mechanism, either 'MCAR', 'MAR' or 'MNAR'
    :param p_obs: parameter for 'MAR' and 'MNAR' mechanisms, probability of fully observed variables
    :param opt: option for 'MNAR' mechanism, either 'logistic', 'quantile' or 'selfmasked'
    :param q: parameter for 'MNAR' mechanism with opt = 'quantile', threshold at which quantile cut should occur    
    :return: incomplete data (n_samples, n_features) numpy array
    """
    if isinstance(data, tuple):
        data = np.array(data)
       
    n_samples, n_features = data.shape
    p_miss = miss_proba/(1-p_obs) # percentage of missing data for variables with missing data, useless for MCAR
    if mecha == 'MCAR':
        missing_data_info = produce_NA(data, miss_proba, mecha=mecha)
    elif mecha == 'MAR':
        missing_data_info = produce_NA(data, p_miss, mecha=mecha, p_obs=p_obs)
    elif mecha == 'MNAR' and opt == 'logistic':
        missing_data_info = produce_NA(data, p_miss, mecha=mecha, p_obs=p_obs, opt = opt)
    elif mecha == 'MNAR' and opt == 'quantile':
        missing_data_info = produce_NA(data, p_miss, mecha=mecha, p_obs=p_obs, opt = opt, q = q)
    elif mecha == 'MNAR' and opt == 'selfmasked':
        missing_data_info = produce_NA(data, p_miss, mecha=mecha, p_obs=p_obs, opt = opt)
    else:
        raise NotImplementedError
    
    data_incomp = missing_data_info['X_incomp'].numpy()
    return data_incomp
