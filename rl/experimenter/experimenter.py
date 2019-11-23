import pdb
import functools
import time
import numpy as np
from rl.algorithms import Algorithm
from rl.tools.utils.misc_utils import safe_assign, timed, cprint
from rl.tools.utils import logz
from rl.experimenter.generate_rollouts import generate_rollout


class Experimenter:

    def __init__(self, env, alg, gen_ro):
        self._env = env
        self._alg = safe_assign(alg, Algorithm)
        self._gen_ro_raw = gen_ro
        self._gen_ro = functools.partial(gen_ro, pi=self._alg.pi_ro, logp=self._alg.logp)
        self._ndata = 0  # number of data points seen

    def gen_ro(self, log_prefix='', to_log=False):
        ro = self._gen_ro()
        self._ndata += ro.n_samples
        if to_log:
            log_rollout_info(ro, prefix=log_prefix)
            logz.log_tabular(log_prefix + 'NumberOfDataPoints', self._ndata)
        return ro

    def run_alg(self, n_itrs, save_policy=None, save_policy_fun=None, save_freq=None,
                pretrain=True, rollout_kwargs=None, **other_pretrain_kwargs):
        start_time = time.time()
        if pretrain:  # algorithm-specific
            if rollout_kwargs is None:
                gr = self._gen_ro_raw
            elif (rollout_kwargs['max_n_rollouts'] is None and
                  rollout_kwargs['min_n_samples'] is None):
                gr = self._gen_ro_raw
            else:
                gr = functools.partial(generate_rollout, env=self._env, **rollout_kwargs)
            self._alg.pretrain(gr, **other_pretrain_kwargs)

        # Main loop
        for itr in range(n_itrs):
            logz.log_tabular("Time", time.time() - start_time)
            logz.log_tabular("Iteration", itr)
            with timed('Generate env rollouts'):
                ro = self.gen_ro(to_log=True)
            # algorithm-specific
            self._alg.update(ro, gen_env_ro=self._gen_ro)
            logz.dump_tabular()  # dump log
            if save_policy and isinstance(save_freq, int) and itr % save_freq == 0:
                save_policy_fun('{}'.format(itr))

        # Save the final policy.
        if save_policy:
            save_policy_fun('final')
            cprint('Final policy has been saved.')


def log_rollout_info(ro, prefix=''):
    # print('Logging rollout info')
    if not hasattr(log_rollout_info, "total_n_samples"):
        log_rollout_info.total_n_samples = {}  # static variable
    if prefix not in log_rollout_info.total_n_samples:
        log_rollout_info.total_n_samples[prefix] = 0
    sum_of_rewards = [rollout.rws.sum() for rollout in ro.rollouts]
    rollout_lens = [len(rollout) for rollout in ro.rollouts]
    n_samples = sum(rollout_lens)
    log_rollout_info.total_n_samples[prefix] += n_samples
    logz.log_tabular(prefix + "NumSamplesThisBatch", n_samples)
    logz.log_tabular(prefix + "NumberOfRollouts", len(ro))
    logz.log_tabular(prefix + "TotalNumSamples", log_rollout_info.total_n_samples[prefix])
    logz.log_tabular(prefix + "MeanSumOfRewards", np.mean(sum_of_rewards))
    logz.log_tabular(prefix + "StdSumOfRewards", np.std(sum_of_rewards))
    logz.log_tabular(prefix + "MaxSumOfRewards", np.max(sum_of_rewards))
    logz.log_tabular(prefix + "MinSumOfRewards", np.min(sum_of_rewards))
    logz.log_tabular(prefix + "MeanRolloutLens", np.mean(rollout_lens))
    logz.log_tabular(prefix + "StdRolloutLens", np.std(rollout_lens))
    logz.log_tabular(prefix + "MeanOfRewards", np.sum(sum_of_rewards) / (n_samples + len(sum_of_rewards)))
