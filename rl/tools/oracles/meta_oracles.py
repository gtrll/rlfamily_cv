from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg as la
import copy
from collections import deque
from rl.tools.utils.mvavg import ExpMvAvg
from rl.tools.oracles.oracle import Oracle
from rl.tools.utils import logz


class MetaOracle(Oracle):
    """These Oracles are built on other Oracle objects."""

    @abstractmethod
    def __init__(self, base_oracle, *args, **kwargs):
        """It should have attribute base_oracle or base_oracles."""


class DummyOracle(MetaOracle):

    def __init__(self, base_oracle, *args, **kwargs):
        self._base_oracle = copy.deepcopy(base_oracle)
        self._g = 0.

    def compute_loss(self):
        return 0.

    def compute_grad(self):
        return self._g

    def update(self, g=None, *args, **kwargs):
        assert g is not None
        self._base_oracle.update(*args, **kwargs)
        self._g = np.copy(g)


class LazyOracle(MetaOracle):
    """Function-based oracle based on moving average."""

    def __init__(self, base_oracle, beta):
        self._base_oracle = copy.deepcopy(base_oracle)
        self._beta = beta
        self._f = ExpMvAvg(None, beta)
        self._g = ExpMvAvg(None, beta)

    def update(self, *args, **kwargs):
        self._base_oracle.update(*args, **kwargs)
        self._f.update(self._base_oracle.compute_loss())
        self._g.update(self._base_oracle.compute_grad())

    def compute_loss(self):
        f = self._base_oracle.compute_loss()
        # if np.isfinite(f):
        #    self._f.replace(f)
        #    return self._f.val
        # else:
        return f

    def compute_grad(self):
        g = self._base_oracle.compute_grad()
        # if np.all(np.isfinite(g)):
        #     self._g.replace(g)
        #     return self._g.val
        # else:
        return g


class AdversarialOracle(LazyOracle):
    """For debugging purpose."""

    def __init__(self, base_oracle, beta):
        super().__init__(base_oracle, beta)
        self._max = None

    def compute_grad(self):
        g = super().compute_grad()
        if self._max is None:
            self._max = np.linalg.norm(g)
        else:
            self._max = max(self._max, np.linalg.norm(g))
        return -g / max(np.linalg.norm(g), 1e-5) * self._max


class WeightedAverageOracle(MetaOracle):
    """
        Weighted average over several recent oracles / a sliding window.
        Currently, these oracles have to be the same type.

        min \sum_n \norm{g_n - p_n}_2^2, n is the iteration number.
        g_n is the true grad, and p_n is the predicted one.
        p_n = V_n w, where the ith column of V_n is the grad predicted by the ith most recent oracle,
        and w is the weights.
        V_n has dimension p x s. p: size of the grad, s: size of weights.
        A: sum of V_n^T V_n, b: sum of V_n^T g_n, then w = A^{-1} b
    """

    def __init__(self, base_oracles, mode=0.1):
        # mode = 'average', average over recent base_oracles. 'recent': use the most recent one.
        # otherwise, mode is the forgetting factor.
        # self.reg_factor = 0.1  # regularization magnitude
        self.reg_factor = 1e-2  # regularization magnitude
        self._base_oracles = deque(base_oracles)  # will be ordered from most recent to least recent.
        self.n_base_oracles = len(base_oracles)
        self.n_valid_base_oracles = 0  # the number of valide base_oracles that have been updated
        self.mode = mode  # mode, or forget factor if it's float
        self.w, self.V, self.A, self.b = None, None, None, None
        self.dim = None

    def update(self, g=None, to_log=False, *args, **kwargs):
        assert g is not None
        # Compute V.
        if self.w is None:  # initialization (V is not needed)
            self.dim = g.shape[0]
            self.V = self._compute_V()  # XXX for logging
        else:
            assert self.V is not None  # make sure compute_grad has been queried
            pred_error_size = la.norm(np.dot(self.V, self.w) - g)

        # Update the most recent oracle using new samples (rotate right).
        oracle = self._base_oracles.pop()  # pop the most right element
        oracle.update(to_log=to_log, *args, **kwargs)
        self._base_oracles.appendleft(oracle)
        if self.n_valid_base_oracles < self.n_base_oracles:
            self.n_valid_base_oracles += 1
        # Regression using true grads
        if self.mode == 'average':
            self.w = np.zeros(self.n_base_oracles)
            self.w[:self.n_valid_base_oracles] = 1.0 / self.n_valid_base_oracles
        elif self.mode == 'recent':
            self.w = np.zeros(self.n_base_oracles)
            self.w[0] = 1.0
        else:
            if self.w is None:  # initialization. cst * 1/2 * (w - e_1)^2
                self.w = np.zeros(self.n_base_oracles)
                self.w[0] = 1.0
                self.A = (self.reg_factor * la.norm(g)**2 / self.n_base_oracles) * np.eye(self.n_base_oracles)
                self.b = np.dot(self.A, self.w)
            else:
                self.A = (1.0 - self.mode) * self.A + np.matmul(self.V.T, self.V)
                self.b = (1.0 - self.mode) * self.b + np.matmul(self.V.T, g)
                self.w = la.solve(self.A, self.b)
                self.w = np.clip(self.w, 0.0, 2.0)  # XXX

        if to_log:
            logz.log_tabular('min_weights', np.min(self.w))
            logz.log_tabular('max_weights', np.max(self.w))
            logz.log_tabular('norm_weights', la.norm(self.w))

        # Reset V.
        self.V = None

    def compute_loss(self):
        if self.w is None:
            return 0.
        else:
            loss = np.zeros([self.n_base_oracles])  # each column is a grad
            for i in range(self.n_valid_base_oracles):
                loss[i] = self._base_oracles[i].compute_loss()
            return np.dot(loss, self.w)

    def _compute_V(self):
        V = np.zeros([self.dim, self.n_base_oracles])  # each column is a grad
        for i in range(self.n_valid_base_oracles):
            V[:, i] = self._base_oracles[i].compute_grad()
        return V

    def compute_grad(self):
        if self.w is None:
            return 0.
        else:
            self.V = self._compute_V()
            return np.dot(self.V, self.w)
