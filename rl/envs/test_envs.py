import numpy as np
import pdb
from rl import envs
from rl.tools.utils.misc_utils import cprint
from rl.tools.utils.misc_utils import timed
# from rl.experimenter.generate_rollouts import Roller, generate_rollout
from rl.experimenter.generate_rollouts import generate_rollout


def time_batch_env(envid, n_envs):
    seed = 0
    n_ro = 5000
    e = envs.create_env(envid, seed)

    def pi(obs):
        ac = e.action_space.sample()
        ac = [ac for _ in range(len(obs))]
        return ac

    # env = envs.create_batch_env(envid, seed, 1, use_ext_proc=False)
    # roller = Roller(env, min_n_samples=None, max_n_rollouts=n_ro, max_rollout_len=None)
    # with timed('1 env generate {} ros'.format(n_ro)):
    #     roller.gen_ro(pi=pi, logp=None)

    # env = envs.create_batch_env(envid, seed, n_envs, use_ext_proc=True)
    # roller = Roller(env, min_n_samples=None, max_n_rollouts=n_ro, max_rollout_len=None)
    # with timed('{} envs parallel generating {} ros'.format(n_envs, n_ro)):
    #     roller.gen_ro(pi=pi, logp=None)

    e = envs.create_batch_env(envid, seed, 1, use_ext_proc=False)
    with timed(''):
        generate_rollout(lambda ob: e.action_space.sample(),
                         None, e, min_n_samples=None, max_n_rollouts=n_ro, max_rollout_len=None)


def test_batch_env(envid, n_envs):
    seed = 0
    n_ro = 10
    e = envs.create_env(envid, seed)
    e_ = envs.create_batch_env(envid, seed, n_envs, use_ext_proc=True)
    isclose_kwargs = {'atol': 1e-4, 'rtol': 1e-4}
    for _ in range(n_ro):
        e.reset()
        e_.reset()
        while True:
            a = e.action_space.sample()
            obs, rew, done, _ = e.step(a)
            obs_, rew_, done_, _ = e_.step(a[None])
            assert np.allclose(obs, obs_, **isclose_kwargs)
            assert np.isclose(rew, rew_, **isclose_kwargs)
            assert done == done_
            if done:
                break


def test_env(envid):
    cprint('Testing env: {}'.format(envid))
    seed = 10
    n_ro = 50
    env = envs.create_env(envid, seed)
    sim_env = envs.create_sim_env(env, seed, 0.0)
    isclose_kwargs = {'atol': 1e-4, 'rtol': 1e-4}

    for _ in range(n_ro):
        obs = env.reset()
        sim_env.reset()
        # print('============================================================')
        # print(env.env.state)
        # print(sim_env.state)
        sim_env.set_state(env.env.state)
        if isinstance(sim_env, envs.Reacher):
            sim_env.target = env.env.target
        assert np.allclose(env.env.state, sim_env.state)
        while True:
            # a = env.action_space.sample()
            a = np.random.normal(size=env.action_space.sample().shape)  # to have samples outside of range
            rew_sim2 = sim_env._batch_reward(obs[None], sim_env.state[None], a[None])[0]
            obs, rew, done, _ = env.step(a)
            obs_sim, rew_sim, done_sim, _ = sim_env.step(a)
            done_sim2 = sim_env._batch_is_done(obs[None])[0]
            # print(env.env.state)
            # print(sim_env.state)
            assert np.allclose(env.env.state, sim_env.state, **isclose_kwargs)
            assert np.allclose(obs, obs_sim, **isclose_kwargs)
            assert np.isclose(rew, rew_sim, **isclose_kwargs)
            assert np.isclose(rew, rew_sim2, **isclose_kwargs)
            assert done == done_sim
            assert done == done_sim2
            if done:
                break

    sim_env_inacc = envs.create_sim_env(env, seed, 0.2)
    sim_env_inacc.reset()
    a = env.action_space.sample()
    obs, rew, done, _ = sim_env_inacc.step(a)


if __name__ == '__main__':
    # envids = ['DartCartPole-v1', 'DartHopper-v1', 'DartSnake7Link-v1', 'DartWalker3d-v1']
    # envids = ['DartWalker3d-v1']
    # envids = ['DartSnake7Link-v1']
    # envids = ['DartCartPole-v1']
    # envids = ['DartHopper-v1']
    # envids = ['DartDog-v1']
    envids = ['DartReacher3d-v1']
    n_envs = 1
    for envid in envids:
        test_env(envid)
        # test_batch_env(envid, n_envs)
        # time_batch_env(envid, n_envs)
