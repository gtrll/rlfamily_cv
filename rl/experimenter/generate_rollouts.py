import time
import numpy as np
import pdb
from rl.experimenter.rollout import RO, Rollout


def generate_rollout(pi, logp, env,
                     min_n_samples, max_n_rollouts, max_rollout_len,
                     with_animation=False, ac_rand=None,
                     start_state=None, start_action=None, init_step=0,
                     min_rollout_len=None, pir=None):
    """
    Collect rollouts until we have enough samples. All rollouts are COMPLETE in
    that they never end prematurely: they end either when done is true or
    max_rollout_len is reached.

    Args:
        pi: a function that maps ob to ac, or to ac + randomness, depending on the dimension of the
            return. If ac_rand is not None, ac_rand is taken as an input to pi.
        logp is either None or a function that maps (obs, acs) to log probabilities
        env: the environment
        max_rollout_len: maximal length of a rollout
        min_rollout_len: minimal length of a rollout
        min_n_samples: minimal number of samples to collect
        max_n_rollouts: maximal number of rollouts
        with_animation: display animiation of the first rollout
        ac_rand: the sequence of randomness applied for each rollout when generating actions.
        start_state: reset env to this state for each rollout
        start_action: apply this action as the first action
    """
    ac_dim = len(env.action_space.low)
    if max_rollout_len is None:
        max_rollout_len = env._max_episode_steps
    else:
        max_rollout_len = min(env._max_episode_steps, max_rollout_len)
    if min_rollout_len is not None:
        assert max_rollout_len >= min_rollout_len

    def get_state():
        if hasattr(env, 'env'):
            return env.env.state  # openai gym env, which is a TimeLimit object
        else:
            return env.state

    def set_state(s):
        if hasattr(env, 'env'):
            ob = env.env.set_state_vector(s)
        else:
            ob = env.set_state(s)
        return ob

    def get_action(ob, steps):
        # If start action is provided, ac_rand is not used.
        if steps == init_step and start_action is not None:
            ac = start_action
        elif ac_rand is not None:
            if len(ac_rand) > steps - init_step:
                ac = pi(ob, ac_rand[steps-init_step])
            else:
                ac = pir(ob)
        else:
            ac = pi(ob)

        # If ac randomness is appended to ac, split into ac and ep part.
        if ac.size > ac_dim:
            ac, ep = np.split(ac, [ac_dim])
        else:
            ep = np.zeros_like(ac)  # just placeholder

        return ac, ep

    def reset():
        ob = env.reset()  # reset the env
        if start_state is not None:
            ob = set_state(start_state)  # set the start state
        st = get_state()
        return ob, st

    ros = []
    obs, acs, eps, rws, sts = [], [], [], [], []
    steps = init_step
    n_samples = 0
    ob, st = reset()  # st should be paired with ob
    while True:
        # Animate.
        if len(ros) == 0 and with_animation:
            env.render()
            time.sleep(0.05)
        # Append.
        ac, ep = get_action(ob, steps)
        obs.append(ob)
        acs.append(ac)
        eps.append(ep)
        sts.append(st)
        # Step.
        ob, rw, done, _ = env.step(ac)
        st = get_state()
        rws.append(rw)
        steps += 1
        # Done with one rollout?
        reached_step_limit = max_rollout_len is not None and steps >= max_rollout_len
        if reached_step_limit or done:
            obs.append(ob)  # an additional ob
            sts.append(st)
            ro = Rollout(obs, acs, eps, rws, sts, done, logp)
            # Is the rollout long enough?
            if min_rollout_len is None or len(ro) >= min_rollout_len:
                n_samples += len(ro)
                ros.append(ro)
            enough_sps = min_n_samples is not None and n_samples >= min_n_samples
            enough_ros = max_n_rollouts is not None and len(ros) >= max_n_rollouts
            # Done?
            if enough_sps or enough_ros:
                break
            else:
                obs, acs, eps, rws, sts = [], [], [], [], []
                steps = init_step
                ob, st = reset()
    return RO(ros)
