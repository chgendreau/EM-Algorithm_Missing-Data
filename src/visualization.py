from imputation import *
from data_generation import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def plot_combined(data, missing_data_percentages=np.arange(5, 65, 5), em_iterations=5, opt="logistic", p_obs=0.2, q=0.5, methods = ['EM', 'Median', 'KNN', 'Iterative'], plot_title = 'Comparison of normalized MSE for Different Missing Data Mechanisms'):
    """
    Given a dataset, plot the MSE for the different missing data mechanisms and the different imputation methods.
    """
    mechanisms = ['MCAR', 'MAR', 'MNAR']
    fig, axs = plt.subplots(1, 3, figsize=(9, 20), sharex=True, sharey='row')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    #mse_values = {method: [] for method in methods}
    mse_values_agg = {method: [] for method in methods}

    for i, mechanism in enumerate(mechanisms):
        for k in range(em_iterations):
            mse_data, _, _ = compute_mses(data, mechanism, methods, missing_data_percentages=missing_data_percentages, p_obs=p_obs, opt=opt, q=q)
            for method in methods:
                mse_values_agg[method].append(mse_data[method])
        
        mse_values_avg = {method: np.mean(mse_values_agg[method], axis=0) for method in methods}
        

        # Plot results
        for method in methods:
            axs[i].plot(missing_data_percentages, mse_values_avg[method], marker='o', label=method)
        axs[i].set_title(mechanism)
        axs[i].set_xlabel('Percentage of Missing Data (%)')
        axs[i].grid(True)

    axs[0].set_ylabel('Average normalized MSE')
    axs[0].legend()
    fig.suptitle(plot_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_all_differences_combined(data, real_mean, real_cov, missing_data_percentages=np.arange(5, 65, 5), em_iterations=5, opt="logistic", p_obs=0.2, q=0.5, verbose=False, plot_MSE_data=True, filename=None, methods = ['EM', 'Median', 'KNN', 'Iterative'], plot_title = 'Comparison of True and Estimated Data and Parameters Across Different Missing Data Mechanisms'):
    """
    Given a dataset, plot the normalized MSEs of the imputed data, estimated mean and estimated covariance for the different missing data mechanisms and the different imputation methods.
    """
    plt.rcParams.update({'font.size': 20})  # Set global font size
    mechanisms = ['MCAR', 'MAR', 'MNAR']

    mse_values_agg = {method: [] for method in methods}
    mse_means_agg = {method: [] for method in methods}
    mse_covs_agg = {method: [] for method in methods}

    if plot_MSE_data:
        fig, axs = plt.subplots(3, 3, figsize=(18, 30), sharex=True, sharey='row')
    else:
        fig, axs = plt.subplots(2, 3, figsize=(18, 20), sharex=True, sharey='row')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i, mechanism in enumerate(mechanisms):
        if False:
            print("Mechanism: ", mechanism)
        for k in range(em_iterations):
            mse_data, mse_mean, mse_cov = compute_mses(data, mechanism, methods, real_mean=real_mean, real_cov=real_cov, missing_data_percentages=missing_data_percentages, p_obs=p_obs, opt=opt, q=q, verbose = verbose)
            for method in methods:
                mse_values_agg[method].append(mse_data[method])
                mse_means_agg[method].append(mse_mean[method])
                mse_covs_agg[method].append(mse_cov[method])

        mse_values_avg = {method: np.mean(mse_values_agg[method], axis=0) for method in methods}
        mse_means_avg = {method: np.mean(mse_means_agg[method], axis=0) for method in methods}
        mse_covs_avg = {method: np.mean(mse_covs_agg[method], axis=0) for method in methods}

        if plot_MSE_data:
            # Plot data MSE
            for method in methods:
                axs[0, i].plot(missing_data_percentages, mse_values_avg[method], marker='o', label=method)
                axs[0, i].set_title(f'{mechanism} - Imputed data MSE')
                axs[0, i].set_xlabel('Percentage of Missing Data (%)')
                axs[0, i].set_ylabel('Average Relative MSE of the imputed data')
                axs[0, i].grid(True)
                axs[0, i].legend()

        # Plot mean MSE
        for method in methods:
            axs[1, i].plot(missing_data_percentages, mse_means_avg[method], marker='o', label=method)
            axs[1, i].set_title(f'{mechanism} - Mean MSE')
            axs[1, i].set_xlabel('Percentage of Missing Data (%)')
            axs[1, i].set_ylabel('Average Relative MSE in estimation of mean')
            axs[1, i].grid(True)
            axs[1, i].legend()

        # Plot covariance MSE
        for method in methods:
            axs[2, i].plot(missing_data_percentages, mse_covs_avg[method], marker='o', label=method, linestyle='--')
            axs[2, i].set_title(f'{mechanism} - Covariance MSE')
            axs[2, i].set_xlabel('Percentage of Missing Data (%)')
            axs[2, i].set_ylabel('Average Relative MSE in estimation of Covariance')
            axs[2, i].grid(True)
            axs[2, i].legend()

    plt.suptitle(plot_title, fontsize=25, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    if filename is not None:
        plt.savefig(filename)