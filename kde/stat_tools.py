import numpy as n

def rebin(a, *args, **kwargs):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    method = kwargs.get("method","sum")
    verbose = kwargs.get("verbose",False)

    shape = a.shape
    lenShape = len(shape)
    factor = n.asarray(shape)/n.asarray(args)
    evList = ['a.reshape('] + \
    ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
    [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)]
    
    if method == "sum":
        pass
    elif method == "average":
        evList += ['/factor[%d]'%i for i in range(lenShape)]
    else:
        raise AttributeError("method: %s not defined" %method)

    if verbose:
        print ''.join(evList)

    return eval(''.join(evList))

def covariance_form(point, mean, cov):
    """
    Calculates 2D map of covariance form (2D quadratic approximation to -2lnL)
    """
    cov_inv = n.linalg.inv(cov)
    diff = point - mean

    stats = [] 
    for y_i in range(len(diff)):
        current_y = []
        for x_i in range(len(diff[y_i])):
            a = n.matrix(diff[y_i][x_i]) 
            current_y.append((a * cov_inv * a.transpose()).item(0))
        stats.append(current_y)
    return n.array(stats)

def estimate_cov_from_contour(xaxis, yaxis, zmesh, point):
    """
    Calculates estimate of covariance matrix from 2D Hessian of -2lnL
    Note:
    RectBivariateSpline expects zmesh to have shape (len(xaxis), len(yaxis))
    but my mesh has shape (len(yaxis), len(xaxis))
    thus everything is mirrored
    """
    from scipy.interpolate import RectBivariateSpline
    x,y = point
    spline = RectBivariateSpline(yaxis, xaxis, n.asarray(zmesh))
    dx2 = 0.5 * spline(y, x, mth=None, dx=0, dy=2, grid=False)
    dy2 = 0.5 * spline(y, x, mth=None, dx=2, dy=0, grid=False)
    dxdy = 0.5 * spline(y, x, mth=None, dx=1, dy=1, grid=False)

    hessian = n.matrix([[dx2, dxdy],[dxdy, dy2]])
    cov = n.linalg.inv(hessian)
    return cov

def interpolate_statistic(xaxis, yaxis, zmesh, xaxis_new, yaxis_new):
    """
    Calculates 2D spline surface of -2lnL test-statistic. 
    The same spline is used to calculate derivatives in "estimate_cov_from_contour(xaxis, yaxis, zmesh, point)"
    Note:
    RectBivariateSpline expects zmesh to have shape (len(xaxis), len(yaxis))
    but my mesh has shape (len(yaxis), len(xaxis))
    thus everything is mirrored
    """    
    from scipy.interpolate import RectBivariateSpline
    spline = RectBivariateSpline(yaxis, xaxis, n.asarray(zmesh))
    stats = [[spline(yaxis_new[yi], xaxis_new[xi], mth=None, dx=0, dy=0, grid=False) for xi in range(len(xaxis_new))] for yi in range(len(yaxis_new))]
    return n.array(stats)

def wilks_test(profiles):
    """
    Calculate the compatibility of statistically independent measurements.
    Here, we assume that Wilks' theorem holds.
        profiles : list of (x,y,llh) for different measurements
    """
    sum_llhs = 0.
    for xpar,ypar,llhs in profiles:
        xmin = n.min([n.min(x) for x,y,z in profiles])
        xmax = n.max([n.max(x) for x,y,z in profiles])
        ymin = n.min([n.min(y) for x,y,z in profiles])
        ymax = n.max([n.max(y) for x,y,z in profiles])
        x = n.linspace(xmin, xmax, 1000)
        y = n.linspace(ymin, ymax, 1000)
        sum_llhs += interpolate_statistic(xpar, ypar, llhs, x, y)

    chi2 = n.min(sum_llhs)
    ndof = 2*(len(profiles)-1)
    from scipy.stats import chisqprob
    pvalue = chisqprob(chi2,ndof)
    from scipy.special import erfinv
    nsigma = erfinv(1-pvalue) * n.sqrt(2) # 2-sided significance

    return (chi2,ndof,pvalue,nsigma)

def walds_test(profile1,profile2):
    """
    Calculate the compatibility of two statistically independent measurements
    using normal approximation (Wald's method).
    This assumes that the log-likelihood space is approximately elliptically.
        profile1 : (x,y,llh) for measurement 1
        profile2 : (x,y,llh) for measurement 2
    """
    bestfits,covariances = [],[]
    for x,y,llhs in [profile1,profile2]:
        idx_min = n.unravel_index(llhs.argmin(), llhs.shape)
        bestfit = x[idx_min[1]],y[idx_min[0]]
        bestfits.append(bestfit)
        covariance = estimate_cov_from_contour(x,y,llhs,bestfit)
        covariances.append(covariance)

    diff = n.matrix(bestfits[0]) - n.matrix(bestfits[1])
    cov_inv = n.linalg.inv(covariances[0] + covariances[1])

    chi2 = diff*cov_inv*diff.transpose()
    ndof = 2
    from scipy.stats import chisqprob
    pvalue = chisqprob(chi2,ndof)
    from scipy.special import erfinv
    nsigma = erfinv(1-pvalue) * n.sqrt(2) # 2-sided significance

    return (chi2,ndof,pvalue,nsigma)

def _weighted_quantile_arg(values, weights, q=0.5):
    indices = n.argsort(values)
    sorted_indices = n.arange(len(values))[indices]
    medianidx = (weights[indices].cumsum()/weights[indices].sum()).searchsorted(q)
    if (0 <= medianidx) and (medianidx < len(values)):
        return sorted_indices[medianidx]
    else:
        return n.nan

def weighted_quantile(values, weights, q=0.5):
    if len(values) != len(weights):
        raise ValueError("shape of `values` and `weights` doesn't match!")
    index = _weighted_quantile_arg(values, weights, q=q)
    if index != n.nan:
        return values[index]
    else:
        return n.nan

def weighted_median(values, weights):
    return weighted_quantile(values, weights, q=0.5)

def weighted_cov(m, y=None, weights=None, bias=0):

    """
    Estimate a (weighted) covariance matrix, given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. 
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        form as that of `m`.
    weights : array_like, optional
              A 1-D array containing the weights of the data points.
              This option should be used if data points have different weights
              in order to calculate the weighted covariance.
    bias : int, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``.

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    Examples
    --------
    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> weighted_cov(x)
    array([[ 1., -1.],
           [-1.,  1.]])

    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how `x` and `y` are combined:

    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> X = np.vstack((x,y))
    >>> print weighted_cov(X)
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print weighted_cov(x, y)
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print weighted_cov(x)
    11.71

    """
    X = n.array(m, ndmin=2, dtype=float)
    if X.size == 0:
        # handle empty arrays
        return np.array(m)
    axis = 0
    tup = (slice(None), n.newaxis)

    N = X.shape[1]

    if weights is not None:
        weights = n.asarray(weights)/n.sum(weights)
        if len(weights)!=N:
            raise ValueError("unequal dimension of `data` and `weights`.")

    if y is not None:
        y = n.array(y, copy=False, ndmin=2, dtype=float)
        X = n.concatenate((X, y), axis)

    X -= n.average(X,axis=1-axis,weights=weights)[tup]

    if bias == 0:
        if weights is not None:
            fact = n.sum(weights)/(n.sum(weights)**2 - n.sum(weights**2))
        else:
            fact = 1./float(N - 1)
    else:
        if weights is not None:
            fact = 1./n.sum(weights)
        else:
            fact = 1./float(N)

    if weights is not None:
        return (n.dot(weights * X, X.T.conj()) * fact).squeeze()
    else:
        return (n.dot(X, X.T.conj()) * fact).squeeze()

