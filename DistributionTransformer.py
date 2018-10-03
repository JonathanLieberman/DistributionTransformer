"""
DistributionTransformer intends to be an sklearn compatible transformer which allows the user to fit and transform
distributions from a list of common distributions for use in various models

Currently the list of supported distributions is:
    Uniform
    Normal
    Exponential

Future development may include partial fitting/data streaming for "Big Data" applications and appropriate distributions
"""

from scipy.stats import expon
from math import log, exp
from scipy.special import erfinv, erf
from numpy import mean, std

# testing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
test = pd.DataFrame(np.arange(.001,1,.001))


class DistributionTransformer:
    def __init__(self, dist_in, dist_out='uniform', epsilon=1e-6 ):
        self.distribution_fit_ = None
        self.dist_in_ = dist_in
        self.dist_out_ = dist_out
        self.epsilon_ = epsilon
        self.supported_dists_ = ['exponential',
                                 'normal',
                                 'uniform']

    def fit(self, X):
        # Check for appropriate input
        self._check_input()

        # fit models
        if self.dist_in_ == 'uniform':
            self.param_in_1_ = X.min()
            self.param_in_2_ = X.max()
        if self.dist_in_ == 'exponential':
            loc, scale = expon.fit(X, floc = 0)
            self.param_in_1_ = scale # make parameter system
            self.param_in_2_ = loc
        if self.dist_in_ == 'normal':
            mu = mean(X)
            sigma = std(X)
            self.param_in_1_ = mu
            self.param_in_2_ = sigma
        # no need to fit a uniform distribution
        self.distribution_fit_ = self.dist_in_
        return self

    def transform(self, X):
        # Check to see if self has attribute lambda, if not, raise error
        # Use self.lambda to transform exponential distribution to normal distribution
        if not (self.distribution_fit_ == self.dist_in_) and not(self.dist_in_ == 'uniform'):
            raise ValueError('object has not been fit to appropriate distribution') # throw in actual names
        # scale = self.scale_
        # X = X.apply(lambda x: (2**.5)*erfinv(1-2*exp(-scale*x)), axis=1)
        cdf = self.to_cdf(X)
        Xnew = self.from_cdf(cdf)
        return Xnew

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_cdf(self, X):
        # X -> cdf
        # Check self.dist_in_ is defined and
        self._check_input()
        #if ~hasattr(self, 'dist_in'):
        #    raise ValueError('no input distribution defined')
        #if ~(self.distribution_fit_ == self.dist_in_):
        #    raise ValueError('fit method has not been called on the right distribution')

        # calculate cdf
        if self.dist_in_ == 'uniform':
            return (X - self.param_in_1_)/(self.param_in_2_-self.param_in_1_)
        if self.dist_in_ == 'exponential':
            # calculate exponential cdf
            return X.apply(lambda x: 1-exp(-self.param_in_1_*x), axis=1)
        if self.dist_in_ == 'normal':
            # calculate normal cdf
            return X.apply(lambda x: .5*(1+erf((x-self.param_in_1_)/(self.param_in_2_*(2**.5)))), axis=1)

    def from_cdf(self, cdf):
        # cdf -> dist_out_
        # Check self.dist_out_ is defined and is in supported_dists
        self._check_output()
        #if ~hasattr(self, 'dist_out_'):
        #    raise ValueError('no output distribution defined')
        #if ~(self.dist_out_ in self.supported_dists_):
        #    raise ValueError('dist_in not supported')
        # project to epsilon:1-epsilon

        def shrink_cdf(x, epsilon=self.epsilon_):
            if x > (1-epsilon):
                x = 1-epsilon
            if x < epsilon:
                x = epsilon
            return x
        cdf = pd.DataFrame(cdf[0].apply(shrink_cdf))

        if self.dist_out_ == 'uniform':
            return cdf
        if self.dist_out_ == 'exponential':
            # calculate and return exponential distribution
            # return cdf.apply(lambda x: -1/self.param_in_1_*log(1-x), axis=1)
            return cdf.apply(lambda x: -log(1 - x), axis=1)
        if self.dist_out_ == 'normal':
            # calculate and return normal distribution
            # return cdf.apply(lambda x: self.param_in_1_ + self.param_in_2_ * 2**.5 * erfinv(2*x-1), axis=1)
            return cdf.apply(lambda x: 2 ** .5 * erfinv(2 * x - 1), axis=1)

    def change_in_out(self, dist_in=None, dist_out=None):
        # Validate inputs
        if not (dist_in in self.supported_dists_ or dist_in is None):
            raise ValueError('dist_in is not a supported distribution')
        if not (dist_out in self.supported_dists_ or dist_out is None):
            raise ValueError('dist_out is not a supported distribution')

        # Change dist_in_/dist_out_
        if not (dist_in is None):
            self._reset()
            self.dist_in_ = dist_in
        if not (dist_out is None):
            self.dist_out_ = dist_out

    def inverse_transform(self, X):
        # Check if has been fit is defined
        #if self.scale_ is None:
        #    raise ValueError('scale not defined')
        #scale = self.scale_
        #X = X.apply(lambda x: -1/scale*log(.5-.5*erf(x/(2**.5))))
        return X

    def _check_input(self):
        if not (hasattr(self, 'dist_in_')):
            raise ValueError('no input distribution defined')
        if not (self.dist_in_ in self.supported_dists_):
            raise ValueError('dist_in is not a supported distribution')

    def _check_output(self):
        if not (hasattr(self, 'dist_out_')):
            raise ValueError('no input distribution defined')
        if not (self.dist_in_ in self.supported_dists_):
            raise ValueError('dist_in is not a supported distribution')

    def _reset(self):
        # Reset variables
        self.distribution_fit_ = None
        if hasattr(self, 'param_in_1_'):
            del self.param_in_1_
        if hasattr(self, 'param_in_2_'):
            del self.param_in_2_

# testing
dt = DistributionTransformer(dist_in = 'uniform', dist_out = 'normal')
print("test")

# To do: fit, transform, inverse transform, to_cdf, from_cdf, reset
# next: from cdf -> find inverse cdf functions
# Could add streaming support for distributions which support streaming fits?