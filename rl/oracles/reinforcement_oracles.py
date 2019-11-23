import pdb
import math
import numpy as np
import functools

from scipy import linalg as la
from rl.tools.oracles import tfLikelihoodRatioOracle
from rl.oracles.oracle import rlOracle
from rl.policies import tfPolicy, tfGaussianPolicy
from rl.tools.normalizers import OnlineNormalizer
from rl.tools.utils.tf_utils import tfObject
from rl.tools.utils import logz
from rl.experimenter.rollout import RO
from rl.experimenter.generate_rollouts import generate_rollout
from rl.tools.function_approximators import online_compatible
from rl.tools.utils.misc_utils import timed, unflatten, cprint


class tfPolicyGradient(tfLikelihoodRatioOracle, rlOracle):
    """ A wrapper of tfLikelihoodRatioOracle for computing policy gradient of the type
            E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]
        where \pi' is specified in ae.
    """

    @tfObject.save_init_args()
    def __init__(self, policy, ae, nor, correlated=True, use_log_loss=False, normalize_weighting=False,
                 onestep_weighting=True, avg_type='avg', **kwargs):
        assert isinstance(policy, tfPolicy)
        assert isinstance(nor, OnlineNormalizer)
        rlOracle.__init__(self, policy)  # set policy as an attribute
        # Initialize the tfLikelihoodRatioOracle
        # we do not need to modify deepcopy lists because no new stateful objects are added.
        ph_args = [policy.ph_obs, policy.ph_acs]
        tfLikelihoodRatioOracle.__init__(
            self, policy.ts_vars, policy.ts_logp,
            ph_args=ph_args, nor=nor, correlated=correlated,
            use_log_loss=use_log_loss, normalize_weighting=normalize_weighting)
        # Define attributes for computing function values and weighting
        self._ae = ae
        self._onestep_weighting = onestep_weighting
        assert avg_type in ['avg', 'sum']
        self._avg_type = avg_type
        self._ro = None

    @property
    def _post_deepcopy_list(self):
        return tfLikelihoodRatioOracle._post_deepcopy_list.fget(self) + ['_ro']

    @property
    def ro(self):
        return self._ro

    def update_ae(self, ro, to_log=False, log_prefix=''):
        self._ae.update(ro, to_log=to_log, log_prefix=log_prefix)

    def update(self, ro, update_nor=False, shift_adv=False, to_log=False, log_prefix=''):
        """
            Args:
                ro: RO object representing the new information
                update_nor: whether to update the  control variate of tfLikelihoodRatioOracle
                shift_adv: whether to force the adv values to be positive. if float, it specifies the
                    amount to shift.
        """

        # Only implemented this version.
        assert shift_adv is False
        assert self._normalize_weighting is False
        assert self._use_log_loss is True
        assert self._avg_type == 'sum'
        assert self._onestep_weighting is False

        self._ro = ro  # save the ref to rollouts

        # Compute adv.
        advs, vfns = self._ae.advs(ro)  # adv has its own ref_policy
        adv = np.concatenate(advs)
        if shift_adv:  # XXX make adv non-negative
            assert self._use_log_loss
            if shift_adv is True:
                adv = adv - np.min(adv)
            else:
                adv = adv - np.mean(adv) + shift_adv
            self._nor.reset()  # defined in tfLikelihoodRatioOracle
            update_nor = False

        if not self._normalize_weighting:
            if self._avg_type == 'sum':  # rescale the problem if needed
                adv *= len(adv) / len(ro)

        # Update the loss function.
        if self._use_log_loss is True:
            #  - E_{ob} E_{ac ~ q | ob} [ w * log p(ac|ob) * adv(ob, ac) ]
            if self._onestep_weighting:  # consider importance weight
                w_or_logq = np.concatenate(self._ae.weights(ro, policy=self.policy))  # helper function
            else:
                w_or_logq = np.ones_like(adv)
        else:  # False or None
            #  - E_{ob} E_{ac ~ q | ob} [ p(ac|ob)/q(ac|ob) * adv(ob, ac) ]
            assert self._onestep_weighting
            w_or_logq = ro.lps

        if to_log:
            vfn = np.concatenate(vfns)
            logz.log_tabular('max_adv', np.amax(np.abs(adv)))
            logz.log_tabular('max_vfn', np.amax(np.abs(vfn)))

        # Update the tfLikelihoodRatioOracle.
        super().update(-adv, w_or_logq, [ro.obs, ro.acs], update_nor)  # loss is negative reward


class tfPolicyGradientSimpleCV(tfPolicyGradient):
    # Natural ordering CV.
    @tfObject.save_init_args()
    def __init__(self, policy, ae, nor,
                 correlated=True, use_log_loss=False, normalize_weighting=False, onestep_weighting=True, avg_type='avg',
                 sim_env=None, n_ac_samples=0, cv_type='nocv', stop_cv_step=1, theta=1.0,
                 quad_style='diff',
                 dyn_update_weights_type='one',
                 rw_update_weights_type='one',
                 var_env=None,
                 switch_at_itr=None,
                 cv_onestep_weighting=False,
                 **kwargs):
        # var_env: env for computing variance.
        # Only implemented this version for now.
        assert correlated is True  # update adv nor before normalizing adv, adv nor NOT used actually
        assert normalize_weighting is False
        assert use_log_loss is True
        assert avg_type == 'sum'
        assert onestep_weighting is False
        assert np.isclose(ae._pe.gamma, 1.0)  # undiscounted problem
        assert np.isclose(ae._pe.lambd, 1.0)  # telescoping sum, no GAE
        assert ae._v_target is not None  # vf should be on
        assert sim_env is not None  # current way of computing q
        assert nor is not None

        tfPolicyGradient.__init__(self, policy, ae, nor, correlated, use_log_loss,
                                  normalize_weighting, onestep_weighting, avg_type)
        self.sim_env = sim_env
        self.adv_nor = nor  # not used yet
        self.ac_dim = policy.y_dim
        self.ob_dim = policy.x_dim
        self.n_ac_samples = n_ac_samples
        self.delta = ae._pe.delta  # the discount factor used in vf definition
        self.ae = ae
        assert cv_type in ['nocv', 'state', 'new', 'quad']
        self.cv_type = cv_type
        self.stop_cv_step = stop_cv_step
        self.dyn_update_weights_type = dyn_update_weights_type
        self.rw_update_weights_type = rw_update_weights_type
        self.gen_ro = functools.partial(generate_rollout, env=var_env,
                                        pi=self.policy.pi, logp=None, min_n_samples=None)
        # extra decay
        self.theta = theta
        self.quad_style = quad_style
        self.cv_onestep_weighting = cv_onestep_weighting
        # For traj cv, first do several steps of state cv to warm up.
        self.switch_at_itr = switch_at_itr
        self.switched = False
        if self.switch_at_itr is not None:
            self.saved_cv_type = self.cv_type
            self.cv_type = 'state'  # switch back at iteration switch_at_itr

    def update(self, ro, update_nor=False, to_log=False, log_prefix='', itr=None, **kwargs):
        if (itr is not None and self.switch_at_itr is not None and
                itr >= self.switch_at_itr and not self.switched):
            cprint('Switch to fancy cv: {} from {}'.format(self.saved_cv_type, self.cv_type))
            self.cv_type = self.saved_cv_type
            self.switched = True
        self._ro = ro

    def compute_grad(self, ret_comps=False):
        mc, ac_os, tau_os = .0, .0, .0

        # Compute for all rollouts.
        # All are lists with length = number of rollouts.
        # qs_env, vs = self._ae.qfns(self._ro, ret_vfns=True)
        # vs = [v[:-1] for v in vs]  # leave out the last one

        # Do it for each rollout, in order to avoid memory overflow,
        # and the actual reason is to make coding easier.
        # pdb.set_trace()
        if self.cv_onestep_weighting:
            onestep_ws = self._ae.weights(self._ro, policy=self.policy)
        else:
            onestep_ws = np.ones(len(self._ro))
        for i, r in enumerate(self._ro.rollouts):
            decay = self.ae._pe.gamma * self.delta
            ws = decay ** np.arange(len(r))
            Ws = np.triu(la.circulant(ws).T, k=0)  # XXX WITHOUT the diagonal terms!!!!
            qs = np.ravel(np.matmul(Ws, r.rws[:, None]))
            gd = self.prepare_grad_data(r)
            mc += self.policy.nabla_logp_f(r.obs_short, r.acs, qs)
            # CV for the first action, state (action) dependent CV.
            ac_os += self.policy.nabla_logp_f(r.obs_short, r.acs,
                                              gd.qs * onestep_ws[i]) - gd.grad_exp_qs
            # CV for the future trajectory (for each t: \delta Q_{t+1} + ... + \delta^{step} Q_{t+step})
            # Note that it starts from t+1.
            if len(np.array(gd.Ws).shape) == 0:
                tau_cvs = gd.Ws*(gd.qs * onestep_ws[i] - gd.exp_qs)
            else:
                tau_cvs = np.ravel(np.matmul(gd.Ws, (gd.qs * onestep_ws[i]-gd.exp_qs)[:, None]))
            tau_os += self.policy.nabla_logp_f(r.obs_short, r.acs, tau_cvs)

        # Average.
        mc /= len(self._ro)
        ac_os /= len(self._ro)
        tau_os /= len(self._ro)

        g = - (mc - (ac_os + tau_os))  # gradient ascent
        if ret_comps:
            return g, mc, ac_os, tau_os
        else:
            return g

    def prepare_grad_data(self, r):
        # r: a rollout object
        class GradData(object):
            def __init__(self, qs, exp_qs, grad_exp_qs, decay=None, stop_cv_step=None):
                self.qs = qs  # T
                self.exp_qs = exp_qs  # T
                self.grad_exp_qs = grad_exp_qs  # d (already sum over the trajectory)
                if decay is not None:
                    ws = decay ** np.arange(len(r))
                    if stop_cv_step is not None:
                        ws[min(stop_cv_step, len(r)):] = 0
                    Ws = np.triu(la.circulant(ws).T, k=1)  # XXX WITHOUT the diagonal terms!!!!
                else:
                    Ws = 1.0
                self.Ws = Ws  # T * T

        if self.cv_type == 'nocv':
            qs = exp_qs = np.zeros(len(r))
            grad_exp_qs = 0.
            grad_data = GradData(qs, exp_qs, grad_exp_qs)

        elif self.cv_type == 'state':
            qs = exp_qs = np.ravel(self.ae._vfn.predict(r.obs_short))
            grad_exp_qs = 0.
            grad_data = GradData(qs, exp_qs, grad_exp_qs)

        elif self.cv_type == 'new':
            qs = self.compute_q(r.obs_short, r.acs, r.sts_short)
            # Sample extra actions for approximating the required expectations.
            # (repeat the same obs for many times consecutively)
            obs_exp = np.repeat(r.obs_short, self.n_ac_samples, axis=0)
            sts_exp = np.repeat(r.sts_short, self.n_ac_samples, axis=0)
            # sample the same randomness for all steps
            rand = np.random.normal(size=[self.n_ac_samples, self.ac_dim])
            rand = np.tile(rand, [len(r), 1])
            acs_exp = self.policy.pi_given_r(obs_exp, rand)
            qs_exp = self.compute_q(obs_exp, acs_exp, sts_exp)
            # Compute exp_qs
            exp_qs = np.reshape(qs_exp, [len(r), self.n_ac_samples])
            exp_qs = np.mean(exp_qs, axis=1)
            # Compute grad_exp_qs
            vs = np.ravel(self.ae._vfn.predict(r.obs_short))
            vs_exp = np.repeat(vs, self.n_ac_samples, axis=0)
            grad_exp_qs = self.policy.nabla_logp_f(obs_exp, acs_exp, qs_exp-vs_exp)
            grad_exp_qs /= self.n_ac_samples  # sum over problem horizon but average over actions
            grad_data = GradData(qs, exp_qs, grad_exp_qs, self.delta*self.theta, self.stop_cv_step)

        elif self.cv_type == 'quad':
            qs, info = self.compute_quad_q(r.obs_short, r.acs, r.sts_short, style=self.quad_style)
            assert isinstance(self.policy, tfGaussianPolicy)
            w = self.ae._pe.gamma ** np.arange(len(r))
            exp_qs, grad_exp_qs = self.policy.quad_exp(r.obs_short, info['A'], info['b'], info['c'], w)
            grad_data = GradData(qs, exp_qs, grad_exp_qs, self.delta*self.theta, self.stop_cv_step)

        else:
            raise ValueError('Unknown cv_type.')

        return grad_data

    # XXX
    def compute_quad_q(self, obs, acs, sts, style='diff'):
        # compute a quadratic q values given obs, and acs.
        # obs do not include the final obs.

        T = acs.shape[0]
        ac_dim = acs.shape[1]
        # Compute E_{s'|s,a}[V(s')]
        dyn = self.sim_env._predict.__self__  # dynamics model
        gamma = self.ae._pe.gamma
        decay = gamma*self.delta
        vfn = self.ae._vfn  # value model

        def aclip(a): return np.clip(a, *self.sim_env._action_clip)
        ms = self.policy.pid(obs)
        xs = np.hstack([obs, (ms)])  # input to the dynamics model (T * ob_dim)
        next_obs = dyn.predict(xs)  # T * ob_dim

        # define A, b, c
        if style == 'now':
            vs = np.ravel(vfn.predict(obs))
            grad_vs = vfn.grad(obs)  # T * ob_dim
            grad_vdyns = dyn.grad(xs, grad_ys=grad_vs)[:, -ac_dim:]  # derivative to actions (T * ob_dim)
            rw_A, rw_b, rw_c = self.sim_env._batch_reward(obs, sts)
            A = rw_A
            b = rw_b + decay*grad_vdyns  # T * ac_dim
            c = rw_c + decay*(vs-np.sum(grad_vdyns*ms, axis=1)+np.sum(grad_vs*(next_obs-obs), axis=1))  # T

        elif 'next' in style:
            next_dones = self.sim_env._batch_is_done(next_obs)
            next_vs = np.ravel(vfn.predict(next_obs))
            next_vs[next_dones] = 0.0
            grad_next_vs = vfn.grad(next_obs)  # T * ob_dim
            grad_next_vs[next_dones] = 0.0
            grad_next_v_dyns = dyn.grad(xs, grad_ys=grad_next_vs)[:, -ac_dim:]  # derivative to actions (T * ob_dim)
            rw_A, rw_b, rw_c = self.sim_env._batch_reward(obs, sts)

            GN = 0.
            if 'gn' in style:
                next_H = vfn.Hess(next_obs)  # T * ob_dim * ob_dim
                next_H[next_dones] = 0.0
                next_H_grad_dyns = dyn.grad_prod(xs, next_H)[:, -ac_dim:, :]  # T * ac_dim * ob_dim
                # T * ac_dim * ac_dim
                GN = dyn.grad_prod(xs,
                                   np.transpose(next_H_grad_dyns, axes=(0, 2, 1)))[:, -ac_dim:, :]
                assert GN.shape == (T, ac_dim, ac_dim)
                if len(rw_A.shape) == 2:
                    rw_A = rw_A[:, None, :]*np.eye(ac_dim)
                assert rw_A.shape == (T, ac_dim, ac_dim)

            A = rw_A + decay * GN
            b = rw_b + decay*grad_next_v_dyns  # T * ac_dim
            c = rw_c + decay*(next_vs-np.sum(grad_next_v_dyns*ms, axis=1))  # T

        elif 'diff' in style:
            vs = np.ravel(vfn.predict(obs))
            next_dones = self.sim_env._batch_is_done(next_obs)
            grad_next_vs = vfn.grad(next_obs)  # T * ob_dim
            grad_next_vs[next_dones] = 0.0
            grad_next_v_dyns = dyn.grad(xs, grad_ys=grad_next_vs)[:, -ac_dim:]  # derivative to actions (T * ac_dim)
            var = self.policy.std**2
            rw_A, rw_b, _ = self.sim_env._batch_reward(obs, sts, omit_c=True)
            GN = 0.
            if 'gn' in style:
                next_H = vfn.Hess(next_obs)  # T * ob_dim * ob_dim
                next_H[next_dones] = 0.0
                next_H_grad_dyns = dyn.grad_prod(xs, next_H)[:, -ac_dim:, :]  # T * ac_dim * ob_dim
                GN = dyn.grad_prod(xs, np.transpose(next_H_grad_dyns, axes=(0, 2, 1)))[
                    :, -ac_dim:, :]  # T * ac_dim * ac_dim
                assert GN.shape == (T, ac_dim, ac_dim)
                if len(rw_A.shape) == 2:
                    rw_A = rw_A[:, None, :]*np.eye(ac_dim)
                assert rw_A.shape == (T, ac_dim, ac_dim)

            A = rw_A + decay*GN
            if len(A.shape) == 2:
                Ams = A*ms  # T * ac_dim
                trAS = np.sum(A*var[None, :], axis=1)  # T
            else:
                Ams = np.squeeze(np.matmul(A, ms[:, :, None]), axis=2)
                trAS = np.trace(A*var, axis1=1, axis2=2)
            b = rw_b + decay*grad_next_v_dyns - Ams   # T * ac_dim
            c = vs - np.sum(b*ms, axis=1) + 0.5*np.sum(ms*Ams, axis=1) - 0.5*trAS

        # Compute qs and info
        if len(A.shape) == 2:  # diagonal
            qs = 0.5 * np.sum(acs*A*acs, axis=1)+np.sum(b*acs, axis=1)+c
        else:  # A is dense
            Aa = np.reshape(np.matmul(A, acs[:, :, None]), [T, ac_dim])
            qs = 0.5 * np.sum(acs*Aa, axis=1)+np.sum(b*acs, axis=1)+c

        info = {'A': A, 'b': b, 'c': c}
        return qs, info

    @online_compatible
    def sim_one_step(self, sts, acs):
        # Note that st is provided instead of ob.
        # Simulate a single step from state st with different actions acs.
        env = self.sim_env
        next_obs = np.zeros((len(sts), self.ob_dim))
        next_rws = np.zeros(len(sts))
        next_dones = np.zeros(len(sts), dtype=bool)

        def set_state(s):
            if hasattr(env, 'env'):
                env.env.set_state_vector(s)
            else:
                env.set_state(s)

        for i, st, ac in zip(range(len(sts)), sts, acs):
            env.reset()  # reset the env
            set_state(st)  # set the start state
            next_obs[i], next_rws[i], next_dones[i], _ = env.step(ac)

        return next_obs, next_rws, next_dones

    @online_compatible
    def compute_v(self, obs, dones=None):
        # V that considers padding
        vfns = np.ravel(self.ae._vfn.predict(obs))
        if dones is not None:
            vfns[dones] = self.ae._pe.default_v
        return vfns

    @online_compatible
    def compute_q(self, obs, acs, sts):
        # compute q values given obs, and acs.
        # obs do not include the final obs.
        assert sts is not None
        assert len(sts) == len(obs)
        assert len(sts) == len(acs)

        if hasattr(self.sim_env, '_predict'):
            # XXX Clipping.
            acs = np.clip(acs, *self.sim_env._action_clip)
            next_obs = self.sim_env._predict(np.hstack([obs, acs]))  # XXX use ob instead of st
            rws = self.sim_env._batch_reward(obs, sts, acs)
            next_dones = self.sim_env._batch_is_done(next_obs)
        else:
            # Note that in the rollouts, last ob and done is generated in the same env.step as here
            next_obs, rws, next_dones = self.sim_one_step(sts, acs)
        vfns = self.compute_v(next_obs, next_dones)
        qfns = rws + self.delta * vfns
        return qfns

    def compute_update_args(self, ro, weights_type, tar=None):
        if weights_type == 'T-t':
            def weight(l): return np.arange(l, 0.0, -1.0)
        elif weights_type == 'one':
            def weight(l): return np.ones(l)
        assert self.sim_env._action_clip is not None

        def clip(acs): return np.clip(acs, *self.sim_env._action_clip)  # low and high limits
        inputs = np.concatenate([np.hstack([r.obs[:-1], clip(r.acs)]) for r in ro.rollouts])
        if tar == 'dyn':
            targets = np.concatenate([r.obs[1:] for r in ro.rollouts])
        elif tar == 'rw':
            targets = np.expand_dims(ro.rws, axis=1)  # n x 1, unsqueeze
        else:
            raise ValueError('Unknow tar: {}'.format(tar))
        weights = np.concatenate([weight(len(r.acs)) for r in ro.rollouts])
        return inputs, targets, weights

    def update_dyn(self, ro, to_log=False):
        if (hasattr(self.sim_env, '_predict') and self.sim_env._predict is not None):
            inputs, targets, weights = self.compute_update_args(ro, self.dyn_update_weights_type,
                                                                tar='dyn')
            self.sim_env._predict.__self__.update(inputs, targets, weights, to_log=to_log)

    def update_rw(self, ro, to_log=False):
        if (hasattr(self.sim_env, '_rw_fun') and self.sim_env._rw_fun is not None):
            inputs, targets, weights = self.compute_update_args(ro, self.rw_update_weights_type,
                                                                tar='rw')
            self.sim_env._rw_fun.__self__.update(inputs, targets, weights, to_log=to_log)

    def log_sigmas(self, idx=100, n_ros=30, n_acs=30, n_taus=30, n_steps=None,
                               use_vf=False):
        # Estimate the vairance of G_idx for different cvs for comparison.
        # n_steps, rollout for max n_steps for tau.
        # use_vf: use value function to reduce the variance in estimate E_a E_tau NQ.

        # Collect samples.
        # Data structure:
        #   sts: 2d array.
        #   acs: 3d array.
        #   advs (advantage function): 3d array.
        #   N (log probability gradient): 3d array.


        # XXX
        # Use state baseline to reduce the variance of the estimates.
        ro = self.gen_ro(max_n_rollouts=n_ros, max_rollout_len=idx+1)
        sts = np.array([r.obs[idx] for r in ro.rollouts if len(r) > idx])
        n_sts = len(sts)

        if n_sts == 0:
            log = {
                'sigma_s_mc': .0,
                'sigma_a_mc': .0,
                'sigma_tau_mc': .0,
                'n_ros_in_total': n_sts * n_acs * n_taus,
                'n_sts': n_sts,
            }
        else:
            acs = self.policy.pi(np.repeat(sts, n_acs, axis=0))
            acs = np.reshape(acs, [n_sts, n_acs, -1])
            Q = np.zeros((n_ros, n_acs, n_taus))
            N_dim = len(self.policy.logp_grad(ro.obs[0], ro.acs[0]))
            N = np.zeros((n_ros, n_acs, N_dim))
            decay = self.ae._pe.gamma * self.delta
            for i, s in enumerate(sts):
                for j, a in enumerate(acs[i]):
                    # This should be the bottleneck!!
                    ro = self.gen_ro(max_n_rollouts=n_taus, max_rollout_len=n_steps,
                                     start_state=s, start_action=a)
                    N[i, j] = self.policy.logp_grad(s, a)
                    for k, r in enumerate(ro.rollouts):
                        q0 = ((decay ** np.arange(len(r))) * r.rws).sum()
                        Q[i, j, k] = q0

            # Fill the rest with zeros.
            if use_vf:
                V = np.zeros((n_ros))
                for i, s in enumerate(sts):
                    V[i] = self.ae._vfn.predict(s[None])[0]

            def compute_sigma_s(Q):
                E_tau_Q = np.mean(Q, axis=2)  # s x a
                if use_vf:
                    E_tau_Q -= np.expand_dims(V, axis=-1)  # s x 1
                E_tau_Q = np.expand_dims(E_tau_Q, axis=-1)  # s x a x 1
                E_a_tau_NQ = np.mean(E_tau_Q * N, axis=1)  # s x N
                E_s_a_tau_NQ = np.mean(E_a_tau_NQ, axis=0)  # N
                E_s_a_tau_NQ = np.expand_dims(E_s_a_tau_NQ, axis=0)  # 1 x N
                Var = np.mean(np.square(E_a_tau_NQ - E_s_a_tau_NQ), axis=0)  # N
                sigma = np.sqrt(np.sum(Var))

                return sigma

            def compute_sigma_a(Q):
                E_tau_Q = np.mean(Q, axis=2)  # s x a
                E_tau_Q = np.expand_dims(E_tau_Q, axis=-1)  # s x a x 1
                N_E_tau_Q = N * E_tau_Q  # s x a x N
                if use_vf:
                    N_E_tau_Q_for_E_a = N * (E_tau_Q - np.reshape(V, V.shape+(1, 1)))
                else:
                    N_E_tau_Q_for_E_a = N_E_tau_Q
                E_a_N_E_tau_Q = np.mean(N_E_tau_Q_for_E_a, axis=1)  # s x N
                E_a_N_E_tau_Q = np.expand_dims(E_a_N_E_tau_Q, axis=1)  # s x 1 x N
                Var = np.mean(np.square(N_E_tau_Q - E_a_N_E_tau_Q), axis=1)  # s x N
                sigma = np.sqrt(np.sum(np.mean(Var, axis=0)))

                return sigma

            def compute_sigma_tau(Q):
                E_tau_Q = np.mean(Q, axis=2)  # s x a
                E_tau_Q = np.expand_dims(E_tau_Q, axis=-1)  # s x a x 1
                Var = np.mean(np.square(Q - E_tau_Q), axis=2)  # s x a
                Var = np.expand_dims(Var, axis=-1)  # s x a x 1
                sigma = np.sqrt(np.sum(np.mean(np.square(N) * Var, axis=(0, 1))))
                return sigma

            log = {
                'sigma_s_mc': compute_sigma_s(Q),
                'sigma_a_mc': compute_sigma_a(Q),
                'sigma_tau_mc': compute_sigma_tau(Q),
                'n_ros_in_total': n_sts * n_acs * n_taus,
                'n_sts': n_sts,
            }

        for k, v in log.items():
            logz.log_tabular(k, v)

