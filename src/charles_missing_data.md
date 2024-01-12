# Introduction (attention, le début du paragraphe a été repris du dernier quarto)
Missing data are common in applications and real life datasets especially in Economics, Political and Social sciences either because some entities do not want to share some statistics or because they are not collected. Technical reasons could also cause missing data such as machine failures or non-responses. Missing data can significantly affect the conclusions drown from a dataset and uncertainty about the conclusions increases as the missing data increase.




# Non-Gaussian Distributions

## Real Data with no distribution assumption

We now try to perform the same analysis starting from a complete dataset of unknown distribution. In fact, the EM algorithm relies on the assumption that the underlying distribution is known which is not the case for the other imputation (or inference relying on incomplete data) methods.

The dataset considered is not the source of interest and will not be described, the goal is simply to test the methods on data on which we cannot make any a priori assumption on the distribution. We extracted a subset of some dataset about wine quality, considering 300 observations of some variables (fixed acidity, volitile acidity, density and pH). The dataset is available [here](data/winequality-white.csv).

We apply the Shapiro-Wilk test for normality which rejected the assumption of normality with a high certainty (the p-value of order $10^{-35}$), as expected. As in the previous scenarios, we imputed the data and made inference on the mean and the covariance matrix using various methods. The results are shown in the following figure.

```{python, echo = False}
#import data
df = pd.read_csv('data/winequality-white.csv', sep=';')
#extracting a subset of the dataset
df = df[['fixed acidity', 'volatile acidity', 'citric acid', 'pH']]
df = df.iloc[1000:1300]

from scipy.stats import shapiro, normaltest
#test for normality

# Assuming your data is stored in the variable 'data'
stat, p_value = shapiro(df)

# Print the test statistic and p-value
print("Shairo p-value:", p_value)
print("Normality rejected (at 5%)" if p_value < 0.05 else "Normality not rejected (at 5%)")

#the real mean and covariance are seen here as the empirical mean and covariance of the complete data
X = df.values
real_mean = np.mean(X, axis=0)
real_cov = np.cov(X, rowvar=False)
#plotting results with different percentages of missing data and missing data mechanisms
plot_all_differences_combined(X, real_mean, real_cov, verbose =True, plot_title = 'Comparison of imputation methods for different missing data mechanisms on the Wine dataset')

```

Quite surprisingly, we see that the imputation coming from the EM algorithm yields a satisfying result of the same quality as the other methods that do not rely on the assumption of Gaussianity (which is not satisfied here). For the mean estimation, we see a significance difference between MCAR for which EM is stable for every methods and for the percentage of missing data and MNAR and MAR for which we have instability given the different methods.

For the Covariance, we observe that the Iterative method outperforms the others for all missing data mechanisms and if the Median, KNN and Iterative methods seem to yield to a convergence, the EM algorithm does not at all (the normalized MSE being of order 1, testifying of a wrong convergence).

## Student's-t Data

The multivariate Student-t distribution is useful for modeling datasets with heavy tails and is often used in finance. Quite often, especially in finance, the assumption of gaussianity on Student-t data can lead to bad estimations, we will therefore analyse this.

The multivariate Student-t has the following parameters

1.  **Mean vector** $\mu$: A $d$-dimensional vector representing the mean of the distribution.
2.  **Scale matrix** $\Sigma$: A positive definite $d \times d$ matrix.
3.  **Degrees of freedom** $\nu > 2$: A scalar value that determines the shape of the distribution's tails. As $\nu$ increases, the Student-t distribution approaches the normal distribution.

Its density function is given by:

$$
f(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}, \nu) = \frac{\Gamma\left(\frac{\nu + d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) \nu^{\frac{d}{2}} \pi^{\frac{d}{2}} |\boldsymbol{\Sigma}|^{\frac{1}{2}}} \left(1 + \frac{1}{\nu} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)^{-\frac{\nu + d}{2}}
$$

where $\mathbf{x}$ is a $d$-dimensional data vector, $\Gamma$ is the gamma function, and $|\Sigma|$ is the determinant of $\Sigma$. The variance is given by $\nu \Sigma / (\nu -2)$.

```{python, echo = False}
# Generation of the multivariate t-distributed data
n = 200  # number of samples
d = 10  # number of features
df = 4  # degrees of freedom
mean = np.random.randn(d)  # mean vector
scale_matrix = 2*np.random.rand(d,d) - 2*np.random.rand(d,d) 
scale_matrix = scale_matrix.T.dot(scale_matrix) # make Scale matrix symmetric positive definite
real_cov = scale_matrix * df / (df - 2)  # real covariance matrix
print(np.linalg.eigvals(scale_matrix))  # check eigenvalues

X = generate_student_t(df, mean = mean, scale_matrix = scale_matrix, n_samples = n)
plot_all_differences_combined(X, real_mean = mean, real_cov = real_cov, plot_title = "Comparison of the imputation of Student's-t data and estimation of parameters across different missingness levels and mechanisms", verbose =False)
```

For Student's-t data, EM converges as the other methods but but with no clear advantage of applying it knowing its computation cost compared to the other methods, which performed as well.

We now generate Student's-t data that passes normality tests. Indeed, as $\nu \to \infty$, the Student's-t distribution converges to the normal distribution. If we simulate data with $\nu$ large enough ($\nu =10$) and we simulate the data until normality tests are passed, we observe that as for normal data, the EM clearly outperforms the other methods for the estimation of the covariance matrix as it was the case for the purely Gaussian data.

```{python, echo = False}
# Generation of Student's-t distributed data that looks like normal data
n = 300  # number of samples
d = 10  # number of features
df = 12  # degrees of freedom. As df -> inf, Student's-t -> Normal
mean = np.random.randn(d)  # mean vector
scale_matrix = 2*np.random.rand(d,d) - 2*np.random.rand(d,d)  # scale matrix
scale_matrix = scale_matrix.T.dot(scale_matrix) # make scale matrix symmetric positive definite
real_cov_quasi_normal = scale_matrix * df / (df - 2)  # real covariance matrix

X_quasi_normal = generate_student_t(df, mean = mean, scale_matrix = scale_matrix, n_samples = n, quasi_normal=True)

plot_all_differences_combined(X_quasi_normal, mean, real_cov_quasi_normal, plot_title = "Comparison of the imputation of quasi normal data and estimation of parameters across different missingness levels and mechanisms", verbose = False)
```

# Missing Data (PEUT ETRE EN 1ERE PARTIE)

In this question, we will discuss about the different types of missingness and how we can generate them given a complete dataset.

## Overview and notations

### This paragraph might should go in the introduction

Missing data are common in applications and real life datasets especially in Economics, Political and Social sciences either because some entities do not want to share some statistics or because they are not collected. Technical reasons could also cause missing data such as machine failures or non-responses. Missing data can significantly affect the conclusions drown from a dataset and uncertainty about the conclusions increases as the missing data increase.

Consider a data set $X \in \Omega_1 \times \cdots \times \Omega_p$ which is a concatenation of $d$ columns $X_j \in \Omega_j$, where $\Omega_j$ is the support of the variable $X_j$ which is of dimension $dim(\Omega_j) = n$, representing the number of observations. This gives us a dataset of dimension $n \times d$. For example, one could have $\Omega_j = \mathbb{R}^n, \mathbb{Z}^n, \mathcal{S}^n$, where $\mathcal{S} = \{s_1, ...., s_k\}$, for some quantitative or qualitative values $s_i$, $i=1..., k; ~ k \in \mathbb{N}$

Consider the response matrix $R\in \{0,1\}^{n \times d}$ defined by $R_{ij} = 1$ if $X_{ij}$ is observed and $0$ otherwise. We now partition the data matrix $X = \{X^{obs}, X^{miss}\}$, where $X^{obs}$ and $X^{miss}$ are the matrices containing the observed and missing values: $X^{obs}_{ij} = X_{ij} I_{\{R_{ij}=1\}}$, $X^{miss}_{ij} = X_{ij} I_{\{R_{ij}=0\}}$.

In the code, the matrix $R$ is seen as a boolean tensor called a *mask* and has value *True* at position $(i,j)$, whenever $X_{ij}$, the i-th observation of the j-th variable is observed. In order to generate missing data, one has to generate a mask $R$ and then apply it to the complete data $X$. If one could easily generate a mask from $n \times d$ independent Bernouilli samples. To generate a mask $R$ in a non-independent way, one can use a logistic model with a sigmoid function. This consists in using the sigmoid function $$\sigma(z) = \frac{1}{1 + e^{-z}}.$$ One will generate from a d-dimensional standard normal distribution weights $W$ and find a vector of intercepts $b \in \mathbb{R}^n$ such that $\sigma(WX + b) \in (0,1)^n$. We can then define the probability $$\mathbb{P}(R_{ij} = 1 \mid X) = \sigma((WX)_i + b_i)$$ and generate a mask $R$ respecting these probabilities. Observe that the missingness $R\_{ij}$ of the i-th observation of the variable $j$ depends on the other variables $X_{ij}$, $j=1,...,d$.

## Missing data mechanisms

There exists different types of missing data mechanisms which are grouped in the following categories.

### Missing completely at random (MCAR)

Observations are said to be be missing completely at random (MCAR) if $R \perp X$ that is if $\mathbb{P}(R \in A \mid X^{obs}, X^{miss}) = \mathbb{P}(R \in A)$ for any $A \in \sigma\{0,1\}^{n \times d}$. To generate such missingness, one would start form a complete data matrix $X$ and generate an independent matrix $R$.

In the code, the mask corresponding to $R$ is generated independently from a $n \times d$ uniform distribution and a given probabibility of observed values $p_{obs}$. With such mask, the data will have in expectation a proportion of missing data of $p_{miss} = 1 - p_{obs}$. Hence, here the missingness does not depend on the variables.

### Missing at random (MAR)

Observations are said to be be missing at random (MAR) if $R \perp X^{miss}$ that is if $\mathbb{P}(R \in A \mid X^{obs}, X^{miss}) = \mathbb{P}(R \in A \mid X^{obs})$ for any $A \in \sigma\{0,1\}^{n \times p}$. To generate such missingness, one would start form a complete data matrix $X$ and generate a matrix $R$ independent from $X^{miss}$ but not from the observed data $X^{obs}$.

To generate such missingness, one needs again to generate a mask $R$ that depends on the observed values but not on the missing ones. To do so, we used a mask generated selecting at random (uniformly) $p_{obs} \cdot d$ variables (or columns) which will have no missing values. For the other variables, we use a logistic model to generate missingness with a fixed missing probability common for each missing variables. This then gives us a MAR response matrix.

### Missing not at random (MNAR)

If missingness is not MCAR or MAR, it is said to be missing not at random (MNAR). To generate such data, one has various options. The first one, is to consider a self-masked model which will apply the logistic model to all variables, meaning that every variable will potentially have missing values.

The second one is to select a certain proportion of variables which will be used as inputs for the logistic model and the remaining variables will have missing probabilities according to the logistic model. Then a MCAR mechanism is applied to the input variables. After this transformation one indeed has a dependence of the missingness of the two groups of variables and hence the mask $R$ depends on missing observations ($X^{miss}$) and on the observed ones ($X^{obs}$).

The code used to generate missing data and its description was taken from <https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values> ADD REFERENCE
