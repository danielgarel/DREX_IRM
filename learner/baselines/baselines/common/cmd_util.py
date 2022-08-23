"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import retro_wrappers

# def make_vec_env(env_id, n_envs=1, seed=None, start_index=0,
#                  monitor_dir=None, wrapper_class=None,
#                  env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None):
#     """
#     Create a wrapped, monitored `VecEnv`.
#     By default it uses a `DummyVecEnv` which is usually faster
#     than a `SubprocVecEnv`.
#     :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
#     :param n_envs: (int) the number of environments you wish to have in parallel
#     :param seed: (int) the initial seed for the random number generator
#     :param start_index: (int) start rank index
#     :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
#         If None, no file will be written, however, the env will still be wrapped
#         in a Monitor wrapper to provide additional information about training.
#     :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
#         This can also be a function with single argument that wraps the environment in many things.
#     :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
#     :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
#     :param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.
#     :return: (VecEnv) The wrapped environment
#     """
#     env_kwargs = {} if env_kwargs is None else env_kwargs
#     vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
#
#     def make_env(rank):
#         def _init():
#             if isinstance(env_id, str):
#                 env = gym.make(env_id)
#                 if len(env_kwargs) > 0:
#                     warnings.warn("No environment class was passed (only an env ID) so `env_kwargs` will be ignored")
#             else:
#                 env = env_id(**env_kwargs)
#             if seed is not None:
#                 env.seed(seed + rank)
#                 env.action_space.seed(seed + rank)
#             # Wrap the env in a Monitor wrapper
#             # to have additional training information
#             monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
#             # Create the monitor folder if needed
#             if monitor_path is not None:
#                 os.makedirs(monitor_dir, exist_ok=True)
#             env = Monitor(env, filename=monitor_path)
#             # Optionally, wrap the environment with the provided wrapper
#             if wrapper_class is not None:
#                 env = wrapper_class(env)
#             return env
#         return _init
#
#     # No custom VecEnv is passed
#     if vec_env_cls is None:
#         # Default: use a DummyVecEnv
#         vec_env_cls = DummyVecEnv
#
#     return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def make_vec_env(env_id, env_type, num_env, seed, env_kwargs={'xml_file': 'hopper.xml'}, wrapper_kwargs=None, start_index=0, reward_scale=1.0, gamestate=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            subrank = rank,
            seed=seed,
            env_kwargs=env_kwargs,
            reward_scale=reward_scale,
            gamestate=gamestate,
            wrapper_kwargs=wrapper_kwargs
        )

    set_global_seeds(seed)
    if num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])


def make_env(env_id, env_type, subrank=0, seed=None, env_kwargs={}, reward_scale=1.0, gamestate=None, wrapper_kwargs={}):
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    if env_type == 'atari':
        env = make_atari(env_id)
    elif env_type == 'retro':
        import retro
        gamestate = gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
    else:
        # print('inside the cmd_util env_kwawrgs: ', env_kwargs)
        # print('inside the cmd_util env_kwawrgs: ', env_id)

        # if env_kwargs == None:
        #     env_kwargs = {}
        print('in the make_env: ', env_kwargs)
        env = gym.make(env_id, **env_kwargs)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)
    elif env_type == 'retro':
        env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)

    return env


def make_mujoco_env(env_id, seed, env_kwargs ={'xml_file': 'hopper.xml'}, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id, **env_kwargs)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    # if env_kwargs==None:env_kwargs={}
    env = gym.make(env_id)#, **env_kwargs)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--custom_reward', default='')
    parser.add_argument('--custom_reward_kwargs', default='{}')

    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dicitonary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval

# import os
# try:
#     from mpi4py import MPI
# except ImportError:
#     MPI = None
#
# import gym
# # from gym.wrappers import FlattenObservation, FilterObservation
# from baselines import logger
# from baselines.bench import Monitor
# from baselines.common import set_global_seeds
# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common import retro_wrappers
# # from baselines.common.wrappers import ClipActionsWrapper
#
# def make_vec_env(env_id, env_type, num_env, seed,
#                  wrapper_kwargs=None,
#                  env_kwargs=None,
#                  start_index=0,
#                  reward_scale=1.0,
#                  flatten_dict_observations=True,
#                  gamestate=None,
#                  initializer=None,
#                  force_dummy=False):
#     """
#     Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
#     """
#     wrapper_kwargs = wrapper_kwargs or {}
#     env_kwargs = env_kwargs or {}
#     mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
#     seed = seed + 10000 * mpi_rank if seed is not None else None
#     logger_dir = logger.get_dir()
#     def make_thunk(rank, initializer=None):
#         return lambda: make_env(
#             env_id=env_id,
#             env_type=env_type,
#             mpi_rank=mpi_rank,
#             subrank=rank,
#             seed=seed,
#             reward_scale=reward_scale,
#             gamestate=gamestate,
#             flatten_dict_observations=flatten_dict_observations,
#             wrapper_kwargs=wrapper_kwargs,
#             env_kwargs=env_kwargs,
#             logger_dir=logger_dir,
#             initializer=initializer
#         )
#
#     set_global_seeds(seed)
#     if not force_dummy and num_env > 1:
#         return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
#     else:
#         return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])
#
#
# def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
#     if initializer is not None:
#         initializer(mpi_rank=mpi_rank, subrank=subrank)
#
#     wrapper_kwargs = wrapper_kwargs or {}
#     env_kwargs = env_kwargs or {}
#     if ':' in env_id:
#         import re
#         import importlib
#         module_name = re.sub(':.*','',env_id)
#         env_id = re.sub('.*:', '', env_id)
#         importlib.import_module(module_name)
#     if env_type == 'atari':
#         env = make_atari(env_id)
#     elif env_type == 'retro':
#         import retro
#         gamestate = gamestate or retro.State.DEFAULT
#         env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
#     else:
#         env = gym.make(env_id, **env_kwargs)
#
#     # if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
#     #     env = FlattenObservation(env)
#
#     env.seed(seed + subrank if seed is not None else None)
#     env = Monitor(env,
#                   logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
#                   allow_early_resets=True)
#
#
#     if env_type == 'atari':
#         env = wrap_deepmind(env, **wrapper_kwargs)
#     elif env_type == 'retro':
#         if 'frame_stack' not in wrapper_kwargs:
#             wrapper_kwargs['frame_stack'] = 1
#         env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)
#
#     # if isinstance(env.action_space, gym.spaces.Box):
#     #     env = ClipActionsWrapper(env)
#
#     if reward_scale != 1:
#         env = retro_wrappers.RewardScaler(env, reward_scale)
#
#     return env
#
#
# def make_mujoco_env(env_id, seed, reward_scale=1.0):
#     """
#     Create a wrapped, monitored gym.Env for MuJoCo.
#     """
#     rank = MPI.COMM_WORLD.Get_rank()
#     myseed = seed  + 1000 * rank if seed is not None else None
#     set_global_seeds(myseed)
#     env = gym.make(env_id)
#     logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
#     env = Monitor(env, logger_path, allow_early_resets=True)
#     env.seed(seed)
#     if reward_scale != 1.0:
#         from baselines.common.retro_wrappers import RewardScaler
#         env = RewardScaler(env, reward_scale)
#     return env
#
# def make_robotics_env(env_id, seed, rank=0):
#     """
#     Create a wrapped, monitored gym.Env for MuJoCo.
#     """
#     set_global_seeds(seed)
#     env = gym.make(env_id)
#     # env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
#     env = Monitor(
#         env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
#         info_keywords=('is_success',))
#     env.seed(seed)
#     return env
#
# def arg_parser():
#     """
#     Create an empty argparse.ArgumentParser.
#     """
#     import argparse
#     return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
# def atari_arg_parser():
#     """
#     Create an argparse.ArgumentParser for run_atari.py.
#     """
#     print('Obsolete - use common_arg_parser instead')
#     return common_arg_parser()
#
# def mujoco_arg_parser():
#     print('Obsolete - use common_arg_parser instead')
#     return common_arg_parser()
#
# def common_arg_parser():
#     """
#     Create an argparse.ArgumentParser for run_mujoco.py.
#     """
#     parser = arg_parser()
#     parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
#     parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
#     parser.add_argument('--seed', help='RNG seed', type=int, default=None)
#     parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
#     parser.add_argument('--num_timesteps', type=float, default=1e6),
#     parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
#     parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
#     parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
#     parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
#     parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
#     parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
#     parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
#     parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
#     parser.add_argument('--play', default=False, action='store_true')
#     return parser
#
# def robotics_arg_parser():
#     """
#     Create an argparse.ArgumentParser for run_mujoco.py.
#     """
#     parser = arg_parser()
#     parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
#     parser.add_argument('--seed', help='RNG seed', type=int, default=None)
#     parser.add_argument('--num-timesteps', type=int, default=int(1e6))
#     return parser
#
#
# def parse_unknown_args(args):
#     """
#     Parse arguments not consumed by arg parser into a dictionary
#     """
#     retval = {}
#     preceded_by_key = False
#     for arg in args:
#         if arg.startswith('--'):
#             if '=' in arg:
#                 key = arg.split('=')[0][2:]
#                 value = arg.split('=')[1]
#                 retval[key] = value
#             else:
#                 key = arg[2:]
#                 preceded_by_key = True
#         elif preceded_by_key:
#             retval[key] = arg
#             preceded_by_key = False
#
#     return retval