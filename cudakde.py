import numpy as n
import warnings

from classes import KDE

class gaussian_kde(KDE):
    def __init__(self, data, weights=[], kde_values=None, use_cuda=True, adaptive=False, weight_adaptive_bw=False, alpha=0.3, bw_method='silverman'):
        if kde_values != None:
            raise NotImplementedError("`kde_values` is not supported for cudakde.")
        KDE.__init__(self, data, use_cuda, weights=weights, alpha=alpha, method=bw_method)

        self.weighted = len(weights) > 0
        
        if adaptive:
            if not self.weighted and weight_adaptive_bw:
                warnings.warn("Since `weights` aren't given `weight_adaptive_bw` will have no effect!")
            self.calcLambdas(weights=weight_adaptive_bw, weightedCov=weight_adaptive_bw)
        else:
            self.lambdas = n.ones(self.n)

    def __call__(self,points):
        points = n.atleast_2d(points)
        self.kde(points, weights=self.weighted, weightedCov=self.weighted)
        return n.array(self.values)


class bootstrap_kde(object):
    def __init__(self, data, niter=10, weights=[], **kwargs):
        self.kernels = []
        self.bootstrap_indices = []

        self.data = n.atleast_2d(data)
        self.d, self.n = self.data.shape
        self.weighted = len(weights) > 0

        for i in xrange(int(niter)):
            indices = self.get_bootstrap_indices()
            self.bootstrap_indices.append(indices)
            if self.weighted:
                kernel = gaussian_kde(data[:,indices], weights=weights[indices], **kwargs)
            else:
                kernel = gaussian_kde(data[:,indices], **kwargs)
            self.kernels.append(kernel)

    def __call__(self,points):
        return self.evaluate(points)

    def evaluate(self,points):
        points = n.atleast_2d(points)
        d, m = points.shape
        means,sqmeans = n.zeros(m),n.zeros(m)
        for kernel in self.kernels:
            values = kernel(points)
            means += values
            sqmeans += values**2
        means /= len(self.kernels)
        sqmeans /= len(self.kernels)
        errors = n.sqrt(sqmeans - means**2)
        return means, errors

    def get_bootstrap_indices(self):
        indices = n.arange(self.n)
        bootstrap_indices = n.random.choice(self.n, size=self.n, replace=True)
        return bootstrap_indices
