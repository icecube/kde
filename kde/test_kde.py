#coding: utf-8

import numpy as np


def test_kde(version, sampling_method, bw_method, n_samples, adaptive,
             alpha=0.3, weight_adaptive_bw=False):
    """Test the KDE routines of the kde package.


    Parameters
    ----------
    version : string
        One of "pykde" or "cudakde"
    sampling_method : string
        One of "uniform" or "exponential"
    bw_method : string
        One of "silverman" or "scott"
    n_samples : int > 0
        Number of random samples to use
    adaptive : bool
        Whether to use adaptive-bandwidth KDE
    alpha : float
        Alpha parameter (used onl for adaptive-BW KDE)
    weight_adaptive_bw : bool
        Whether to apply weights to samples


    Raises
    ------
    Exception if test fails

    """
    # Translate inputs
    version = version.strip().lower()
    sampling_method = sampling_method.strip().lower()
    bw_method = bw_method.strip().lower()

    if version == "pykde":
        from kde.pykde import bootstrap_kde, gaussian_kde
    elif version == "cudakde":
        from kde.cudakde import bootstrap_kde, gaussian_kde
    else:
        raise ValueError('`version` must be one of "pykde" or "cudakde".')

    # Define a data model and generate some random data between 0 and 10
    # Number of trials
    n_samples = int(n_samples)

    # Exponential Model
    expec = lambda x: 1./(np.exp(-10)-1.)**2 * np.exp(-x)

    # Generated data and reweighted to the exponential model
    np.random.seed(0)
    if sampling_method == "uniform":
        # Uniformly-generated data and weights
        x1 = np.random.uniform(0, 10, n_samples)
        x1_weights = np.exp(-x1)
    elif sampling_method == "exponential":
        # Exponentially-generated data and weights
        x1 = np.random.exponential(2, n_samples)
        x1_weights = np.exp(-0.5*x1)
    else:
        raise ValueError('`sampling_method` must be one of "uniform" or'
                         ' "exponential".')

    # Exponentially generated data (w/o weights)
    x2 = np.random.exponential(1, n_samples)

    #
    # Get histograms
    #

    # Define bins
    bins = np.linspace(0, 10, 31)

    # Weighted data
    hist_weights = np.histogram(x1, bins=bins, weights=x1_weights,
                                density=True)

    # Exponential data
    hist_expo = np.histogram(x2, bins=bins, density=True)

    #
    # Get KDE kernels
    #

    # Kernels for weighted data (w/o adaptive kernels)
    kernel_weights = gaussian_kde(x1, weights=x1_weights, bw_method=bw_method)

    # Kernels for weighted data (with adaptive kernels)
    kernel_weights_adaptive = gaussian_kde(
        x1, weights=x1_weights, bw_method=bw_method, adaptive=adaptive,
        weight_adaptive_bw=weight_adaptive_bw, alpha=alpha
    )

    # Kernels for exponential data (w/o adaptive kernels)
    kernel_expo = gaussian_kde(x2, bw_method=bw_method)

    # Kernels for exponential data (with adaptive kernels)
    kernel_expo_adaptive = gaussian_kde(x2, bw_method=bw_method,
                                        adaptive=adaptive, alpha=alpha)

    #
    # Plot histograms and KDEs
    #

    # Define evaluation points
    X = np.linspace(0, 10, 1001)

    # In presence of boundaries reflect the KDEs at the boundary

    # Define reflection range
    x_below = (-2., 0.)
    # Refelection only necessary if data is uniformly generated between [0,10]
    x_above = (10., 12.)

    # Define evaluation points beyond the boundaries (below 0 and above 10)
    mask_below = (X <= (x_below[1]-(x_below[0]-x_below[1])))
    X_below = x_below[1] - (X[mask_below] - x_below[1])

    mask_above = (X >= (x_above[0]-(x_above[1]-x_above[0])))
    X_above = x_above[0] + (x_above[0] - X[mask_above])

    Y_weights = kernel_weights(X)
    Y_weights[mask_below] += kernel_weights(X_below)
    if sampling_method == "uniform":
        Y_weights[mask_above] += kernel_weights(X_above)

    Y_weights_adaptive = kernel_weights_adaptive(X)
    Y_weights_adaptive[mask_below] += kernel_weights_adaptive(X_below)
    if sampling_method == "uniform":
        Y_weights_adaptive[mask_above] += kernel_weights_adaptive(X_above)

    #
    # Plots for exponential data
    #

    Y_expo = kernel_expo(X)
    Y_expo[mask_below] += kernel_expo(X_below)

    Y_expo_adaptive = kernel_expo_adaptive(X)
    Y_expo_adaptive[mask_below] += kernel_expo_adaptive(X_below)

    #
    # For an error estimate on an evaluation point use bootstrapping
    #

    # Define the number of bootstrap iterations
    nbootstraps = 1000

    #
    # Get bootstrapped KDE kernels (settings as set above)
    #

    # Kernels for weighted data (w/o adaptive kernels)
    bootstrap_kernel_weights = bootstrap_kde(x1, weights=x1_weights,
                                             bw_method=bw_method,
                                             niter=nbootstraps)

    # Kernels for weighted data (with adaptive kernels)
    bootstrap_kernel_weights_adaptive = bootstrap_kde(
        x1, weights=x1_weights, bw_method=bw_method, adaptive=adaptive,
        weight_adaptive_bw=weight_adaptive_bw, alpha=alpha, niter=nbootstraps
    )

    # Kernels for exponential data (w/o adaptive kernels)
    bootstrap_kernel_expo = bootstrap_kde(x2, bw_method=bw_method,
                                          niter=nbootstraps)

    # Kernels for exponential data (with adaptive kernels)
    bootstrap_kernel_expo_adaptive = bootstrap_kde(x2, bw_method=bw_method,
                                                   adaptive=adaptive,
                                                   alpha=alpha,
                                                   niter=nbootstraps)

    # Plots using reflection and bootstrapping

    # Plots for weighted data

    Y_weights = bootstrap_kernel_weights(X)
    Y_weights_below = bootstrap_kernel_weights(X_below)
    Y_weights_above = bootstrap_kernel_weights(X_above)

    Y_weights[0][mask_below] += Y_weights_below[0]
    Y_weights[1][mask_below] = np.sqrt(
        Y_weights[1][mask_below]**2 + Y_weights_below[1]**2
    )
    if sampling_method == "uniform":
        Y_weights[0][mask_above] += Y_weights_above[0]
        Y_weights[1][mask_above] = np.sqrt(
            Y_weights[1][mask_above]**2 + Y_weights_above[1]**2
        )

    Y_weights_adaptive = bootstrap_kernel_weights_adaptive(X)
    Y_weights_adaptive_below = bootstrap_kernel_weights_adaptive(X_below)
    Y_weights_adaptive_above = bootstrap_kernel_weights_adaptive(X_above)

    Y_weights_adaptive[0][mask_below] += Y_weights_adaptive_below[0]
    Y_weights_adaptive[1][mask_below] = np.sqrt(
        Y_weights_adaptive[1][mask_below]**2 + Y_weights_adaptive_below[1]**2
    )
    if sampling_method == "uniform":
        Y_weights_adaptive[0][mask_above] += Y_weights_adaptive_above[0]
        Y_weights_adaptive[1][mask_above] = np.sqrt(
            Y_weights_adaptive[1][mask_above]**2
            + Y_weights_adaptive_above[1]**2
        )

    Y_expo = bootstrap_kernel_expo(X)
    Y_expo_below = bootstrap_kernel_expo(X_below)
    Y_expo_above = bootstrap_kernel_expo(X_above)

    Y_expo[0][mask_below] += Y_expo_below[0]
    Y_expo[1][mask_below] = np.sqrt(
        Y_expo[1][mask_below]**2 + Y_expo_below[1]**2
    )

    Y_expo_adaptive = bootstrap_kernel_expo_adaptive(X)
    Y_expo_adaptive_below = bootstrap_kernel_expo_adaptive(X_below)
    Y_expo_adaptive_above = bootstrap_kernel_expo_adaptive(X_above)

    Y_expo_adaptive[0][mask_below] += Y_expo_adaptive_below[0]
    Y_expo_adaptive[1][mask_below] = np.sqrt(
        Y_expo_adaptive[1][mask_below]**2 + Y_expo_adaptive_below[1]**2
    )


if __name__ == "__main__":
    test_kde(version='cudakde',
             sampling_method='exponential',
             bw_method='silverman',
             n_samples=100,
             adaptive=True,
             alpha=0.3,
             weight_adaptive_bw=True)
    print "<< test_kde.py / cudakde : PASSED >>"
    test_kde(version='pykde',
             sampling_method='exponential',
             bw_method='silverman',
             n_samples=100,
             adaptive=True,
             alpha=0.3,
             weight_adaptive_bw=False)
    print "<< test_kde.py / pykde : PASSED >>"

