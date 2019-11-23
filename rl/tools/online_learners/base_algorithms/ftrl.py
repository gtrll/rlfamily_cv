import numpy as np

from rl.tools.online_learners.base_algorithm import BaseAlgorithm
import rl.tools.online_learners.prox as px
from rl.tools.utils.mvavg import ExpMvAvg

# TODO below needs to be updated according to the new Prox class


class FTRL(BaseAlgorithm):
    """ A simple adaptive FTRL with (adaptive) scalar learning rate"""

    def __init__(self, x0, prox, scheduler):
        super().__init__()
        self._scheduler = scheduler
        self._create_reg = lambda x, w: px.BregmanDivergences(prox, x, w)
        eta_0 = self._scheduler.stepsize
        self._sum_w_g = np.zeros_like(x0)  # h
        self._reg = self._create_reg(x0, 1 / eta_0)  # H

    def _project(self):
        # argmin_x  < sum_wg, x > + sum_n B_n(x|| x_n)
        return self._reg.proxstep(self._sum_w_g)

    def adapt(self, g, w):
        eta_n_minus_1 = self._scheduler.stepsize / self._G  # previous stepsize
        # compute the new stepsize
        self._scheduler.update(w)
        self._update_G(g)
        eta_n = self._scheduler.stepsize / self._G
        # update the regularization
        self._reg += self._create_reg(self.project(), (1.0 / eta_n - 1.0 / eta_n_minus_1))

    def _update(self, g, w):
        self._sum_w_g += w * g

    @property
    def _G(self):
        return 1.  # an upper bound of the gradient norm

    def _update_G(self, g):
        pass  # (optional) this can be updated

    @property
    def stepsize(self):
        return self._scheduler.stepsize / self._G


class FTRLMax(FTRL):

    def __init__(self, x0, prox, scheduler):
        super().__init__(x0, prox, scheduler)
        self._G_max = 1e-8  # an upper bound of the gradient norm

    @property
    def _G(self):
        return 1.0 if self._G_max <= 1e-8 else self._G_max

    def _update_G(self, g):
        self._G_max = max(norm(g), self._G_max)


class FTRLAvg(FTRL):

    def __init__(self, x0, prox, scheduler):
        super().__init__(x0, prox, scheduler)
        self._G_avg = ExpMvAvg(1e-8, 0.999)

    @property
    def _G(self):
        return self._G_avg.val

    def _update_G(self, g):
        self._G_avg.update(norm(g))


class FTRLDoublingTrick(BaseAlgorithm):
    # Constant step size during 2^expo iterations. expo increments.
    def __init__(self, x0, prox, eta, p=1.0):
        super().__init__()
        self._scheduler = DoublingTrickScheduler(eta, p)  # eta is an estimate for R
        self._create_reg = lambda x, w: px.BregmanDivergences(prox, x, w)
        self._x = x0  # the first reg is added after the first g is seen
        self._sum_w_g = arrayfun(np.zeros_like, x0)
        self._reg = None

    def _project(self):
        return self._reg.proxstep(self._sum_w_g)

    def adapt(self, g, w):
        # w should not be used!
        dualnorm = norm(g)
        if self._scheduler.update(dualnorm):
            eta = self._scheduler.stepsize
            self._reg = self._create_reg(self.project(), 1.0 / eta)
            self._sum_w_g = np.zeros_like(self.project())  # reset memory

    def _update(self, g, w):
        # w should not be used!
        self._sum_w_g += self._scheduler.w * g

    @property
    def stepsize(self):
        return self._scheduler.stepsize
