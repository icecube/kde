# Author: Sebastian Schoenen
# Date: 2014-01-17
"""
Class for kernel density estimation.
Currently, only Gaussian kernels are implemented.
"""

from __future__ import division, print_function
from copy import copy

import numexpr
from numpy import *
from scipy import linalg

from stat_tools import weighted_cov

__all__ = ['gaussian_kde', 'bootstrap_kde']


class gaussian_kde(object):
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    It includes automatic bandwidth determination.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    weights : array_like
        A 1-D array containing the weights of the data points.
        This option should be used if data points are weighted
        in order to calculate the a weighted KDE.
    adaptive : boolean
        Should adaptive kernel density estimation be applied?
        For the implementation see:
        Algorithm 3.1 in DOI: 10.1214/154957804100000000 
    weight_adaptive_bw : boolean
        If using the adaptive kernel density estimation it can be chosen
        if the adaptive bandwidth should be calculated by
        the weighted (True) or unweighted (False) dataset.
    alpha : float
        The sensitivity parameter alpha in the range [0,1] is needed for the
        adaptive KDE. Also for this see:
        Algorithm 3.1 in DOI: 10.1214/154957804100000000 
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.
    local_covariance : ndarray
        An array of covariance matrices of `dataset`
        scaled by the calculated adaptive bandwidth.
    local_inv_cov : ndarray
        The inverse of `local_covariance`.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    kde.set_bandwidth(bw_method='scott') : None
        Computes the bandwidth, i.e. the coefficient that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        .. versionadded:: 0.11.0
    kde.covariance_factor : float
        Computes the coefficient (`kde.factor`) that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        The default is `scotts_factor`.  A subclass can overwrite this method
        to provide a different method, or set it through a call to
        `kde.set_bandwidth`.
    kde._compute_adaptive_covariance : None
        Computes the adaptive bandwidth for `dataset`.


    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        n * (d + 2) / 4.)**(-1. / (d + 4)).

    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.

    Examples
    --------
    Generate some random two-dimensional data:

    >>> from scipy import stats
    >>> def measure(n):
    >>>     "Measurement model, return two coupled measurements."
    >>>     m1 = n.random.normal(size=n)
    >>>     m2 = n.random.normal(scale=0.5, size=n)
    >>>     return m1+m2, m1-m2

    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()

    Perform a kernel density estimate on the data:

    >>> X, Y = n.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = n.vstack([X.ravel(), Y.ravel()])
    >>> values = n.vstack([m1, m2])
    >>> kernel = gaussian_kde(values)
    >>> Z = n.reshape(kernel(positions).T, X.shape)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.imshow(n.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()

    """
    def __init__(self, dataset, weights=None, kde_values=None,
                 adaptive=False, weight_adaptive_bw=False, alpha=0.3, bw_method='silverman'):
        self.dataset = atleast_2d(dataset)
        self.d, self.n = self.dataset.shape
        max_array_length = 1e8
        """Maximum amount of data in memory (~2GB, scales linearly)"""
        self.m_max = int(floor(max_array_length/self.n))
        if self.n > max_array_length:
            raise ValueError("`dataset` is too large (too many array entries)!")
        
        if weights is not None and len(weights)==self.n:
            self.weights = weights
        elif weights is None:
            self.weights = ones(self.n)
        else:
            raise ValueError("unequal dimension of `dataset` and `weights`.")

        self.kde_values = kde_values
        if self.kde_values is not None:
            print("Warning: By giving `kde_values`, `weight_adaptive_bw` is"
                  " useless. You have to be sure what was used to calculate"
                  " those values!")
            if len(self.kde_values)!=self.n:
                raise ValueError("unequal dimension of `dataset` and `kde_values`.")
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        
        # compute covariance matrix 
        self.set_bandwidth(bw_method=bw_method)
        
        self.adaptive = adaptive
        if self.adaptive:
            self.weight_adaptive_bw = weight_adaptive_bw
            try:
                self.alpha = float(alpha)
            except:
                raise ValueError("`alpha` has to be a number.")
            if self.alpha < 0. or self.alpha > 1.:
                raise ValueError("`alpha` has to be in the range [0,1].")
            self._compute_adaptive_covariance()
        elif not self.adaptive and self.kde_values is not None:
            raise ValueError("Giving `kde_values`, `adaptive` cannot be False!")

    def evaluate(self, points, adaptive=False):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        points = dot(self.inv_cov12, atleast_2d(points))
        ds = self.ds
        normalized_weights = self._normalized_weights
        
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (m,d))
                d, m = points.shape
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d, self.d)
                raise ValueError(msg)
        
        nloops = int(ceil(m/self.m_max))
        dm = self.m_max
        modulo_dm = m%dm
        results = empty((m,), dtype=float)
        if adaptive:
            inv_loc_bw = self.inv_loc_bw

            for i in xrange(nloops):
                index = i*dm
                if modulo_dm and i==(nloops-1):
                    dm = modulo_dm
                pt = points[:,index:index+dm].T.reshape(dm,self.d,1)

                # has to be done due to BUG in `numexpr` (`sum` in `numexpr` != `numpy.sum`)
                if self.d == 1:
                    energy = numexpr.evaluate("(ds - pt)**2", optimization='aggressive').reshape(dm, self.n)
                else:
                    energy = numexpr.evaluate("sum((ds - pt)**2, axis=1)", optimization='aggressive')

                results[index:index+dm] = numexpr.evaluate("sum(normalized_weights * exp(-0.5 * energy * inv_loc_bw), axis=1)", optimization='aggressive')
                del pt

        else:
            for i in xrange(nloops):
                index = i*dm
                if modulo_dm and i==(nloops-1):
                    dm = modulo_dm
                pt = points[:,index:index+dm].T.reshape(dm,self.d,1)

                # has to be done due to BUG in `numexpr` (`sum` in `numexpr` != `numpy.sum`)
                if self.d == 1:
                    energy = numexpr.evaluate("(ds - pt)**2", optimization='aggressive').reshape(dm, self.n)
                else:
                    energy = numexpr.evaluate("sum((ds - pt)**2, axis=1)", optimization='aggressive')

                results[index:index+dm] = numexpr.evaluate("sum(normalized_weights * exp(-0.5 * energy), axis=1)", optimization='aggressive')
                del pt

        return results

    def __call__(self, points):
        return self.evaluate(points, adaptive=self.adaptive)

    def scotts_factor(self):
        return power(self.n, -1./(self.d+4))

    def silverman_factor(self):
        return power(self.n*(self.d+2.0)/4.0, -1./(self.d+4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.

        Examples
        --------
        >>> x1 = n.array([-7, -5, 1, 4, 5.])
        >>> kde = stats.gaussian_kde(x1)
        >>> xs = n.linspace(-10, 10, num=50)
        >>> y1 = kde(xs)
        >>> kde.set_bandwidth(bw_method='silverman')
        >>> y2 = kde(xs)
        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
        >>> y3 = kde(xs)

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.plot(x1, n.ones(x1.shape) / (4. * x1.size), 'bo',
        ...         label='Data points (rescaled)')
        >>> ax.plot(xs, y1, label='Scott (default)')
        >>> ax.plot(xs, y2, label='Silverman')
        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
        >>> ax.legend()
        >>> plt.show()

        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif isscalar(bw_method) and not isinstance(bw_method, basestring):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        data_covariance = atleast_2d(weighted_cov(self.dataset, weights=self.weights, bias=False))
        data_inv_cov = linalg.inv(data_covariance)

        covariance = data_covariance * factor**2
        inv_cov = data_inv_cov / factor**2
        self.inv_cov12 = linalg.cholesky(inv_cov).T

        self.ds = dot(self.inv_cov12, self.dataset)

        norm_factor = sqrt(linalg.det(2*pi*covariance))
        #inv_norm_factor = 1. / (norm_factor * sum(self.weights))
        self._normalized_weights = self.weights / (norm_factor * sum(self.weights))

    def _compute_adaptive_covariance(self):
        """Computes an adaptive covariance matrix for each Gaussian kernel using
        _compute_covariance().
        """
        # evaluate dataset for kde without adaptive kernel:
        if self.kde_values == None:
            if self.weight_adaptive_bw:
                self.kde_values = self.evaluate(self.dataset, adaptive=False)
            else:
                weights_temp = copy(self.weights)
                self.weights = ones(self.n)
                self._compute_covariance()
                self.kde_values = self.evaluate(self.dataset, adaptive=False)
                self.weights = weights_temp
                self._compute_covariance()

        # Define global bandwidth `glob_bw` by using the kde without adaptive kernel:
        # NOTE: is this really self.n or should it be sum(weights)?
        glob_bw = exp(1./self.n * sum(log(self.kde_values)))
        # Define local bandwidth `loc_bw`:
        self.inv_loc_bw = power(self.kde_values/glob_bw, 2.*self.alpha)

        #inv_local_norm_factors = self._inv_norm_factor * power(self.inv_loc_bw, 0.5*self.d)
        self._normalized_weights = self._normalized_weights * power(self.inv_loc_bw, 0.5*self.d)

class bootstrap_kde(object):
    """Bootstrapping to estimate uncertainty in KDE.

    Parameters
    ----------
    dataset
    niter : int > 0
    **kwargs
        Passed on to `gaussian_kde`, except 'weights' which,if present, is
        extracted and re-sampled in the same manner as `dataset`.

    """
    def __init__(self, dataset, niter=10, **kwargs):
        self.kernels = []
        self.bootstrap_indices = []

        self.dataset = atleast_2d(dataset)
        self.d, self.n = self.dataset.shape
        if kwargs.has_key("weights"):
            weights = kwargs.pop("weights")
        else:
            weights = None

        for i in xrange(niter):
            indices = self.get_bootstrap_indices()
            self.bootstrap_indices.append(indices)
            if weights is not None:
                kernel = gaussian_kde(self.dataset[:,indices], weights=weights[indices], **kwargs)
                self.kernels.append(kernel)
            else:
                kernel = gaussian_kde(self.dataset[:,indices], **kwargs)
                self.kernels.append(kernel)

    def __call__(self, points):
        return self.evaluate(points)

    def evaluate(self, points):
        points = atleast_2d(points)
        d, m = points.shape
        means, sqmeans = zeros(m), zeros(m)
        for kernel in self.kernels:
            values = kernel(points)
            means += values
            sqmeans += values**2
        means /= len(self.kernels)
        sqmeans /= len(self.kernels)
        errors = sqrt(sqmeans - means**2)
        return means, errors

    def get_bootstrap_indices(self):
        """Get random indices used to resample (with replacement) `dataset`.

        Returns
        -------
        bootstrap_indices : array

        """
        indices = arange(self.n)
        bootstrap_indices = random.choice(self.n, size=self.n, replace=True)
        return bootstrap_indices

