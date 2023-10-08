'''Register environments.'''

from utils.registration import register

register(idx='quadrotor',
         entry_point='model.quadrotor:Quadrotor',
         config_entry_point='model:quadrotor.yaml')

'''Initializing vectorized environments.'''

import random

import torch
import numpy as np

from env.vectorized_env.dummy_vec_env import DummyVecEnv
from env.vectorized_env.subproc_vec_env import SubprocVecEnv


def make_env_fn(env_func,
                env_config,
                seed,
                rank):
    '''Higher-order function for env init func.

    Args:
        env_func (function): partial function that can accept args.
        env_config (dict): env init args.
        seed (int): base seed.
        rank (int): unique seed for each parallel env.

    Returns:
        _thunk (func): env-constructing func.
    '''

    def _thunk():
        # Do not set seed i if 0 (e.g. for evaluation).
        if seed is not None:
            e_seed = seed + rank
            random.seed(e_seed)
            np.random.seed(e_seed)
            torch.manual_seed(e_seed)
            env = env_func(seed=e_seed, **env_config)
        else:
            env = env_func(**env_config)
        return env
    return _thunk


def make_vec_envs(env_func,
                  env_configs=None,
                  batch_size=1,
                  n_processes=1,
                  seed=None):
    '''Produce envs with parallel rollout abilities.

    Args:
        env_func (function): partial function that can accept args.
        env_config (dict): non-shareable args for each env.
        batch_size (int): total num of parallel envs.
        n_processes (int): num of parallel workers to run envs.
        seed (int): base seed for the run.

    Returns:
        VecEnv (Vectorized env): (wrapped) parallel envs.
    '''
    if env_configs is None:
        env_configs = [{}] * batch_size
    env_fns = [make_env_fn(env_func, env_configs[i], seed, i) for i in range(batch_size)]
    if n_processes > 1:
        return SubprocVecEnv(env_fns, n_workers=n_processes)
    else:
        # E.g. can use in evaluation (with seed -1).
        return DummyVecEnv(env_fns)
