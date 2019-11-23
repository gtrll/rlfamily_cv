from abc import ABC, abstractmethod
import numpy as np
import math


class Scheduler(ABC):
    # Interface for schedulers.

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @property
    @abstractmethod
    def stepsize(self):
        pass


class PowerScheduler(object):
    """
        A helper class for calculating the stepsize (i.e. the regularization
        constant) for a weighted online learning problem.
    """

    def __init__(self, eta, k=None, c=1e-3, p=1.0, N=200, limit=None):
        # It computes stepsize = \eta / (1+c*sum_w / sqrt{n}) / eta_nor, where
        # eta_nor is a normalization factor so that different choices of p are
        # comparable XXX p and N are only used in eta normalization.
        self._c = c  # how fast the learning rate decays
        self._k = k if k is not None else 0.5  # 0.5 for CVX, and 0.0 for SCVX
        self._eta = eta  # nominal stepsize
        self._eta_nor = self._compute_eta_nor(p, self._k, self._c, N)  # the constant normalizer of eta
        self._limit = eta if limit is None else limit
        self._w = 1
        self.reset()

    def reset(self):
        self._sum_w = 0
        self._itr = 0

    def update(self, w=1.0):
        self._w = w
        self._sum_w += w
        self._itr += 1

    @property
    def stepsize(self):
        stepsize = self._eta / (1.0 + self._c * self._sum_w / np.sqrt(self._itr + 1e-8)) / self._eta_nor
        if stepsize * self._w > self._limit:
            stepsize = self._limit / self._w
        return stepsize

    @staticmethod
    def _compute_eta_nor(p, k, c, N):
        # Compute the normalization constant for lr, s.t. the area under the
        # scheduling with arbitrary p is equal to that with p = 0.
        nn = np.arange(1, N, 0.01)

        def area_under(_p, _k, _c):
            return np.sum(nn**_p / (1.0 + _c * np.sum(nn**_p) / np.sqrt(nn**_k + 1e-8)))
        return area_under(p, k, c) / area_under(0, 0.5, c)

# TODO double check this


class DoublingTrickScheduler(object):
    # Adaptive step size, adjusted per 2^expo iterations.
    # Compute the optimal steps size given the current estimate of G, to minimize the regret at iteration 2^expo,
    # when we reach 2^expo iteration, iteration counter and sum of g are reset.
    # Added convexity: G \frac{ \sqrt{\sum w_n^2} } {w_1 R}, and w_1 = 1, n = 2^expo, w_n = n^p
    # itr: is the # of iterations done so far!

    def __init__(self, eta, p, mode='max'):
        self._stats = .0  # certains stats about G
        self._itr = 0
        self._R = eta  # R is the eta constant
        self._expo = .0
        self._eta = .0
        self._p = p
        self._mode = mode

    @property
    def w(self):
        return (self._itr + 1) ** self._p

    def reset(self):
        self._stats = .0  # certains stats about G
        self._itr = 0
        self._expo = .0
        self._eta = .0

    def update(self, dualnorm):
        # Return True/False indicating whether step size is updated.
        self._itr += 1
        if self._mode == 'avg':
            self._stats += dualnorm
            G = self._stats / self._itr  # average
        elif self._mode == 'max':
            self._stats = max(self._stats, dualnorm)
            G = self._stats
        else:
            raise ValueError('Unknown G estimate mode')
        expo = math.log2(self._itr)
        if np.isclose(expo, self._expo):
            # Compute the optimal stepsize for the next 2**(expo+1) steps.
            sum_w_n_squared = np.sum((np.arange(2**(self._expo + 1)) + 1.0) ** (2 * self._p))
            self._eta = 1.0 / (G * np.sqrt(sum_w_n_squared) / self._R)
            # Reset.
            self._expo += 1  # expo + 1
            self._itr = 0
            self._stats = .0
            print('eta', self._eta)
            return True
        return False

    @property
    def stepsize(self):
        return self._eta
