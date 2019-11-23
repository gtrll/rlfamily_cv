import pdb
import numpy as np
from rl.tools.online_learners import online_optimizer as OO
from rl.tools.online_learners import base_algorithms as BA
from rl.policies import Policy


class Reg(object):
    # regularization based on KL divergence between policies
    def __init__(self, refpol, varpol, default_damping=0.1):
        """
        refpol:
            reference point to compute gradient.
        varpol:
            variable policy, which has the variables to optimize over.
        """
        assert isinstance(refpol, Policy) and isinstance(varpol, Policy)
        self._damping0 = default_damping
        self.refpol = refpol
        self.varpol = varpol
        self.obs = None
        self.damping = None

    def kl(self, x):
        self.varpol.variable = x
        return self.refpol.kl(self.varpol, self.obs, reversesd=False)

    def fvp(self, g):
        return self.refpol.fvp(self.obs, g) + self.damping * g

    @property
    def std(self):
        return self.refpol.std if hasattr(self.refpol, 'std') else 1.0

    def assign(self, reg):
        assert type(self) == type(reg)
        self.refpol.assign(reg.refpol)
        self.obs = np.copy(reg.obs)
        self.damping = np.copy(reg.damping)
        self._damping0 = np.copy(reg._damping0)

    def update(self, obs):
        self.refpol.assign(self.varpol)
        self.obs = np.copy(obs)
        self.damping = self._damping0  # /np.mean(self.std**2.0)

    @property
    def initialized(self):
        return not self.obs is None


class rlOnlineOptimizer(object):
    pass


def _rlOnlineOptimizerDecorator(cls):
    """
    A decorator for reusing OO.OnlineOptimizers while making them use policy as the variable.
    This would create a new class that is both the base class in OO.OnlineOptimizer and rlOnlineOptimizer.
    """
    assert issubclass(cls, OO.OnlineOptimizer)

    class decorated_cls(cls, rlOnlineOptimizer):
        """ self._policy is only used as an interface variable for other rl
        codes, which can be modified on-the-fly, NOT the state of the optimzier
        """

        def __init__(self, policy, base_alg, p=0.0, damping=None, **kwargs):
            assert isinstance(policy, Policy) and isinstance(base_alg, BA.BaseAlgorithm)
            self._policy = policy
            super().__init__(base_alg, p=p, **kwargs)
            if self.has_fisher:
                if damping is not None:
                    self._reg = Reg(self._policy.copy('reg_pol'), self._policy, default_damping=damping)
                else:
                    self._reg = Reg(self._policy.copy('reg_pol'), self._policy)

        @property
        def has_fisher(self):
            return isinstance(self._base_alg, BA.SecondOrderUpdate)

        def update(self, g, ro=None, **kwargs):
            if self.has_fisher:
                assert ro is not None
                limit = 100000  # max number of obs use in computing kl
                obs = ro.obs
                if len(obs) > limit:
                    obs = obs[np.random.choice(len(ro.obs), limit, replace=False)]
                self._reg.update(obs)
                if isinstance(self._base_alg, BA.TrustRegionSecondOrderUpdate):
                    super().update(g, mvp=self._reg.fvp, dist_fun=self._reg.kl, **kwargs)
                elif isinstance(self._base_alg, BA.RobustAdaptiveSecondOrderUpdate):
                    # Has to go before AdaptiveSecondOrderUpdate
                    super().update(g, mvp=self._reg.fvp, dist_fun=self._reg.kl)
                elif isinstance(self._base_alg, BA.AdaptiveSecondOrderUpdate):
                    super().update(g, mvp=self._reg.fvp, **kwargs)
            else:
                super().update(g, **kwargs)
            self._policy.variable = self.x

        def _set_x(self, val):
            super()._set_x(val)
            self._policy.variable = self.x

        def assign(self, policy):
            # NOTE this does not change the state of the optimizer
            self._policy.assign(policy)

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


# =====================================================================================
# Here we define all rl-oriented classes for policy optimization
@_rlOnlineOptimizerDecorator
class BasicOnlineOptimizer(OO.BasicOnlineOptimizer):
    pass

