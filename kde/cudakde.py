# pylint: disable=invalid-name


from __future__ import absolute_import, division

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

import warnings
import numpy as n
from .classes import KDE


class gaussian_kde(KDE):
    def __init__(self, data, weights=None, kde_values=None, use_cuda=True,
                 adaptive=False, weight_adaptive_bw=False, alpha=0.3,
                 bw_method='silverman'):
        if kde_values != None:
            raise NotImplementedError("`kde_values` is not supported for"
                                      " cudakde.")
        KDE.__init__(self, data, use_cuda, weights=weights, alpha=alpha,
                     method=bw_method)

        self.weighted = False if weights is None or len(weights) == 0 else True

        if adaptive:
            if not self.weighted and weight_adaptive_bw:
                warnings.warn("Since `weights` aren't given"
                              " `weight_adaptive_bw` will have no effect!")
            self.calcLambdas(weights=weight_adaptive_bw,
                             weightedCov=weight_adaptive_bw)
        else:
            self.lambdas = n.ones(self.n)

    def __call__(self, points):
        points = n.atleast_2d(points)
        self.kde(points, weights=self.weighted, weightedCov=self.weighted)
        return n.array(self.values)


class bootstrap_kde(object):
    def __init__(self, data, niter=10, weights=None, **kwargs):
        assert int(niter) == float(niter)
        niter = int(niter)

        self.kernels = []
        self.bootstrap_indices = []

        self.data = n.atleast_2d(data)
        self.d, self.n = self.data.shape
        self.weighted = False if weights is None or len(weights) == 0 else True

        for _ in range(niter):
            indices = n.array(self.get_bootstrap_indices())
            self.bootstrap_indices.append(indices)
            if self.weighted:
                kernel = gaussian_kde(data[..., indices],
                                      weights=weights[indices],
                                      **kwargs)
            else:
                kernel = gaussian_kde(data[..., indices], **kwargs)
            self.kernels.append(kernel)

    def __call__(self, points):
        return self.evaluate(points)

    def evaluate(self, points):
        points = n.atleast_2d(points)
        _, m = points.shape
        means, sqmeans = n.zeros(m), n.zeros(m)
        for kernel in self.kernels:
            values = kernel(points)
            means += values
            sqmeans += values**2
        means /= len(self.kernels)
        sqmeans /= len(self.kernels)
        errors = n.sqrt(sqmeans - means**2)
        return means, errors

    def get_bootstrap_indices(self):
        bootstrap_indices = n.random.choice(self.n, size=self.n, replace=True)
        return bootstrap_indices
