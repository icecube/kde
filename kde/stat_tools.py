# pylint: disable=line-too-long, invalid-name


from __future__ import absolute_import, division, print_function

__license__ = """MIT License

Copyright (c) 2014-2019 Sebastian Schoenen and Martin Leuermann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


def rebin(a, *args, **kwargs):
    """Rebin ndarray data into a smaller ndarray of the same rank whose
    dimensions are factors of the original dimensions. eg. An array with 6
    columns and 4 rows can be reduced to have 6,3,2 or 1 columns and 4,2 or 1
    rows.

    Examples
    --------
    >>> a = np.rand(6, 4)
    >>> b = rebin(a, 3, 2)
    >>> print(b.shape)
    (2, 2)

    >>> a = np.rand(6)
    >>> b = rebin(a, 2)
    >>> print b.shape
    (3,)

    """
    method = kwargs.get("method", "sum")
    verbose = kwargs.get("verbose", False)

    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) / np.asarray(args) # pylint: disable=unused-variable
    evList = (
        ['a.reshape('] +
        ['args[%d],factor[%d],'%(i, i) for i in range(lenShape)] +
        [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)]
    )

    if method == "sum":
        pass
    elif method == "average":
        evList += ['/factor[%d]'%i for i in range(lenShape)]
    else:
        raise AttributeError("method: %s not defined" % method)

    evStr = ''.join(evList)

    if verbose:
        print(evStr)

    return eval(evStr) # pylint: disable=eval-used


def covariance_form(point, mean, cov):
    """Calculate 2D map of covariance form (2D quadratic approximation to
    -2lnL)

    """
    cov_inv = np.linalg.inv(cov)
    diff = point - mean

    stats = []
    for y_i in range(len(diff)):
        current_y = []
        for x_i in range(len(diff[y_i])):
            a = np.matrix(diff[y_i][x_i])
            current_y.append((a * cov_inv * a.transpose()).item(0))
        stats.append(current_y)
    return np.array(stats)


def estimate_cov_from_contour(xaxis, yaxis, zmesh, point):
    """Calculate estimate of covariance matrix from 2D Hessian of -2lnL

    Note:
    RectBivariateSpline expects zmesh to have shape (len(xaxis), len(yaxis))
    but my mesh has shape (len(yaxis), len(xaxis)) thus everything is mirrored

    """
    from scipy.interpolate import RectBivariateSpline
    x, y = point
    spline = RectBivariateSpline(yaxis, xaxis, np.asarray(zmesh))
    dx2 = 0.5 * spline(y, x, mth=None, dx=0, dy=2, grid=False)
    dy2 = 0.5 * spline(y, x, mth=None, dx=2, dy=0, grid=False)
    dxdy = 0.5 * spline(y, x, mth=None, dx=1, dy=1, grid=False)

    hessian = np.matrix([[dx2, dxdy], [dxdy, dy2]])
    cov = np.linalg.inv(hessian)
    return cov


def interpolate_statistic(xaxis, yaxis, zmesh, xaxis_new, yaxis_new):
    """Calculate 2D spline surface of -2lnL test-statistic.

    The same spline is used to calculate derivatives in
    "estimate_cov_from_contour(xaxis, yaxis, zmesh, point)"

    Note:
    RectBivariateSpline expects zmesh to have shape (len(xaxis), len(yaxis))
    but my mesh has shape (len(yaxis), len(xaxis))
    thus everything is mirrored

    """
    from scipy.interpolate import RectBivariateSpline
    spline = RectBivariateSpline(yaxis, xaxis, np.asarray(zmesh))
    stats = [[spline(yaxis_new[yi], xaxis_new[xi], mth=None, dx=0, dy=0, grid=False)
              for xi in range(len(xaxis_new))]
             for yi in range(len(yaxis_new))]
    return np.array(stats)


def wilks_test(profiles):
    """Calculate the compatibility of statistically independent measurements.

    Here, we assume that Wilks' theorem holds.

    Parameters
    ----------
    profiles : list of (x, y, llh) for different measurements

    """
    from scipy.stats import chisqprob
    from scipy.special import erfinv

    xmin, xmax = +np.inf, -np.inf
    ymin, ymax = +np.inf, -np.inf
    for x, y, _ in profiles:
        xmin_, xmax_ = np.min(x), np.max(x)
        if xmin_ < xmin:
            xmin = xmin_
        if xmax_ > xmax:
            xmax = xmax_

        ymin_, ymax_ = np.min(y), np.max(y)
        if ymin_ < ymin:
            ymin = ymin_
        if ymax_ > ymax:
            ymax = ymax_

    x = np.linspace(xmin, xmax, 1000)
    y = np.linspace(ymin, ymax, 1000)

    sum_llhs = 0
    for xpar, ypar, llhs in profiles:
        sum_llhs += interpolate_statistic(xpar, ypar, llhs, x, y)

    chi2 = np.min(sum_llhs)
    ndof = 2 * (len(profiles) - 1)
    pvalue = chisqprob(chi2, ndof)
    nsigma = erfinv(1 - pvalue) * np.sqrt(2) # 2-sided significance

    return (chi2, ndof, pvalue, nsigma)


def walds_test(profile1, profile2):
    """Calculate the compatibility of two statistically independent
    measurements using normal approximation (Wald's method).

    This assumes that the log-likelihood space is approximately elliptically.

    Parameters
    ----------
    profile1 : (x,y,llh) for measurement 1
    profile2 : (x,y,llh) for measurement 2

    """
    from scipy.stats import chisqprob
    from scipy.special import erfinv
    bestfits, covariances = [], []
    for x, y, llhs in [profile1, profile2]:
        idx_min = np.unravel_index(llhs.argmin(), llhs.shape)
        bestfit = x[idx_min[1]], y[idx_min[0]]
        bestfits.append(bestfit)
        covariance = estimate_cov_from_contour(x, y, llhs, bestfit)
        covariances.append(covariance)

    diff = np.matrix(bestfits[0]) - np.matrix(bestfits[1])
    cov_inv = np.linalg.inv(covariances[0] + covariances[1])

    chi2 = diff*cov_inv*diff.transpose()
    ndof = 2
    pvalue = chisqprob(chi2, ndof)
    nsigma = erfinv(1-pvalue) * np.sqrt(2) # 2-sided significance

    return (chi2, ndof, pvalue, nsigma)


def _weighted_quantile_arg(values, weights, q=0.5):
    indices = np.argsort(values)
    sorted_indices = np.arange(len(values))[indices]
    medianidx = (weights[indices].cumsum()/weights[indices].sum()).searchsorted(q)
    if (medianidx >= 0) and (medianidx < len(values)):
        return sorted_indices[medianidx]
    return np.nan


def weighted_quantile(values, weights, q=0.5):
    if len(values) != len(weights):
        raise ValueError("shape of `values` and `weights` doesn't match!")
    index = _weighted_quantile_arg(values, weights, q=q)
    if index != np.nan:
        return values[index]
    return np.nan


def weighted_median(values, weights):
    return weighted_quantile(values, weights, q=0.5)


def weighted_cov(m, y=None, weights=None, bias=0):
    """Estimate a (weighted) covariance matrix, given data.

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
        A 1-D array containing the weights of the data points. This option
        should be used if data points have different weights in order to
        calculate the weighted covariance.
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
    >>> print(weighted_cov(X))
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print(weighted_cov(x, y))
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print(weighted_cov(x))
    11.71

    """
    X = np.array(m, ndmin=2, dtype=float)
    if X.size == 0:
        # handle empty arrays
        return np.array(m)

    axis = 0
    tup = (slice(None), np.newaxis)

    N = X.shape[1]

    if weights is not None:
        weights = np.asarray(weights)/np.sum(weights)
        if len(weights) != N:
            raise ValueError("unequal dimension of `data` and `weights`.")

    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=float)
        X = np.concatenate((X, y), axis)

    X -= np.average(X, axis=1-axis, weights=weights)[tup]

    if bias == 0:
        if weights is not None:
            fact = np.sum(weights) / (np.sum(weights)**2 - np.sum(weights**2))
        else:
            fact = 1 / (N - 1)
    else:
        if weights is not None:
            fact = 1 / np.sum(weights)
        else:
            fact = 1 / N

    if weights is not None:
        return (np.dot(weights * X, X.T.conj()) * fact).squeeze()

    return (np.dot(X, X.T.conj()) * fact).squeeze()
