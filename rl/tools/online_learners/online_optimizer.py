from abc import ABC, abstractmethod
import numpy as np
import copy

import scipy.optimize as spopt
from rl.tools.online_learners.base_algorithms import MirrorDescent, Adam, BaseAlgorithm
from rl.tools.online_learners.scheduler import PowerScheduler
from rl.tools.utils.misc_utils import cprint


class OnlineOptimizer(ABC):
    """
    An easy-to-use interface of BaseAlgorithm for solving weighted online learning problems.
    The weight is n^p by default.
    """

    def __init__(self, base_alg, p=0.0):
        assert isinstance(base_alg, BaseAlgorithm)
        self._base_alg = base_alg  # a BaseAlgorithm object
        self._itr = 0  # starts with 0
        self._p = p  # the rate of the weight

    def reset(self):
        self._itr = 0
        self._base_alg.reset()

    def _get_w(self, itr):  # NOTE this can be overloaded
        return itr**self._p

    @property
    def w(self):  # weighting for the loss sequence
        return self._get_w(self._itr)

    @property
    def x(self):  # decision
        return self._base_alg.project()

    @x.setter
    def x(self, val):
        self._set_x(val)

    def _set_x(self, val):  # NOTE this can be overloaded
        assert isinstance(self._base_alg, MirrorDescent)
        self._base_alg.set(val)

    @property
    def stepsize(self):  # effective stepsize taken (for debugging)
        return self.w * self._base_alg.stepsize

    @abstractmethod
    def update(self, *args, **kwargs):
        # self._itr += 1  # starts a new round
        # update the decision with g wrt w
        pass


class BasicOnlineOptimizer(OnlineOptimizer):
    """
        A online optimizer for adversarial linear problems.
    """

    def update(self, g, **kwargs):
        self._itr += 1  # starts a new round
        self._base_alg.adapt(g, self.w, **kwargs)
        self._base_alg.update(g, self.w)


