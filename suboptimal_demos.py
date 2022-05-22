import yaml
import gym
import numpy as np
import torch.nn as nn
import os
import tensorflow as tf
import datetime
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO as PPOSB
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from itertools import count


def process_sb_args(sb_args):
    def without_keys(d, *keys):
        return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))

    sb_args = without_keys(sb_args, 'n_envs', 'n_timesteps', 'policy',
                           'env_wrapper', 'normalize')

    # # process policy_kwargs str
    if 'policy_kwargs' in sb_args.keys():
        if isinstance(sb_args['policy_kwargs'], str):
            sb_args['policy_kwargs'] = eval(sb_args['policy_kwargs'])

    # process schedules
    for key in ["learning_rate", "clip_range", "clip_range_vf"]:
        if key not in sb_args:
            continue
        if isinstance(sb_args[key], str):
            schedule, initial_value = sb_args[key].split("_")
            initial_value = float(initial_value)
            sb_args[key] = linear_schedule(initial_value)

    return sb_args


def format_name_string(name_string):
    name_string = name_string.replace('{', '_').replace('}', '').replace(' ', '').replace("'xml_file'", '')
    name_string = name_string.replace("'", "").replace(":", "").replace('/', '')
    name_string = name_string.replace(".xml", "")

    return name_string


def save_expert_traj(env, model, spec_kwargs, cost, extra_reward_threshold=0,
                     nr_trajectories=1, pl_model_file=None):
    num_steps = 0
    expert_traj = []
    expert_traj_extra = []

    if pl_model_file is not None:
        print(pl_model_file)

    for i_episode in count():
        print(i_episode)
        ob = env.reset()
        done = False
        total_reward = 0
        episode_traj = []
        stacked_vec = []

        # Adding the lines below to match the dataset.pkl from CORL - DG:
        observations = []
        actions = []
        rewards = []

        while not done:
            ac, _states = model.predict(ob)
            # print(env)
            # if not isinstance(ac,list):
            # ac = np.array([ac])
            next_ob, reward, done, _ = env.step(ac)
            ob = next_ob
            total_reward += reward
            # if len(ob.shape) != len(ac.shape):
            # print("shape mismatch")
            # ob = np.squeeze(ob)
            observations.append(np.squeeze(np.array(ob)))
            actions.append(np.squeeze(np.array(ac)))
            rewards.extend(reward)
            #             stacked_vec = np.hstack([np.squeeze(ob), np.squeeze(ac), reward, done])
            #             stacked_vec = (np.concatenate(ob,axis=0),np.concatenate(ac,axis=0),np.concatenate(reward,axis=0))
            #             expert_traj.append(stacked_vec)
            #             episode_traj.append(stacked_vec)
            num_steps += 1

        stacked_vec.append(np.array(observations))
        stacked_vec.append(np.array(actions))
        stacked_vec.append(rewards)
        expert_traj.append(stacked_vec)
        #         episode_traj.append(stacked_vec)

        print("episode:", i_episode, "reward:", total_reward,
              "extra threshold", extra_reward_threshold)

        #         if total_reward > extra_reward_threshold:
        #             expert_traj_extra.extend(episode_traj)

        if i_episode == nr_trajectories - 1:
            break

    filename = env_name + format_name_string(str(spec_kwargs)) + str(cost)
    if pl_model_file is not None:
        filename = filename + pl_model_file
    demo_dir = './demos'
    if not os.path.exists(demo_dir):
        os.mkdir(demo_dir)
        os.mkdir(os.path.join(demo_dir, 'preference_learning'))
        os.mkdir(os.path.join(demo_dir, 'preference_learning', env_name + spec_kwargs + str(cost)))

    #     expert_traj = np.stack(expert_traj)

    if pl_model_file is not None:
        with open(os.path.join(demo_dir, 'preference_learning/' + filename + "_expert_traj.pkl"), 'wb') as f:
            pickle.dump(expert_traj, f)
    else:
        with open(os.path.join(demo_dir, filename + "_expert_traj.pkl"), 'wb') as f:
            pickle.dump(expert_traj, f)
        # np.save(os.path.join(opt.demo_dir, filename + "_expert_traj.npy"), noisy_traj)

    # if pl_model_file is not None:
    #     np.save(os.path.join(opt.demo_dir, 'preference_learning/' + filename + "_expert_traj.npy"), expert_traj)
    # else:
    #     np.save(os.path.join(opt.demo_dir, filename + "_expert_traj.npy"), expert_traj)


#     if len(expert_traj_extra) > 0 and pl_model_file is not None:
#         expert_traj_extra = np.stack(expert_traj_extra)
#         with open(os.path.join(demo_dir, filename + "_expert_traj_extra.pkl"), 'wb') as f:
#             pickle.dump(expert_traj_extra, f)
#         #np.save(os.path.join(opt.demo_dir, filename + "_expert_traj_extra.npy"), expert_traj_extra)


# This notebook is used to create the suboptimal trajectories
with open('config_sb3.yaml') as parameters:
    sb_args = yaml.safe_load(parameters)['Hopper-v3']
n_envs = sb_args['n_envs']
n_timesteps = int(sb_args['n_timesteps'])
policy = sb_args['policy']

env_name = 'Hopper-v3'
sb_args = process_sb_args(sb_args)

# Exploring option 1 with foot friction and reward
# Specifically create a spurious correlation between ctrl_cost_weight and friction. That is the penalty of the hopper taking too large actions.
# env_kwargs = [{'xml_file': 'hopper.xml'}, {'xml_file': 'hopper_foot_mu1.xml'}, {'xml_file': 'hopper_foot_mu3.xml'}]
env_kwargs = [{'xml_file': 'hopper_foot_mu1.xml'}, {'xml_file': 'hopper_foot_mu3.xml'}]

# default ctrl_cost_weight=0.001 and default foot_mu=2

pref_ckpt_dir = './sb_models/ckpt_' + env_name
seed = 0

if not os.path.exists('demos/preference_learning'):
    os.mkdir('demos/preference_learning')


def make_env(rank, ctrl_cost):
    def _thunk():
        env = gym.make(env_name, ctrl_cost_weight=ctrl_cost, **spec)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        #         env = apply_wrappers(env, **wrapper_kwargs)
        return env

    return _thunk


for spec, costs in zip(env_kwargs, [[0.01, 0.008], [0.1, 0.17]]):
    for cost in costs:
        pref_ckpt_dir = './sb_models/ckpt_' + env_name + format_name_string(str(spec)) + '_' + str(cost)

        print(spec, cost)
        envs = [make_env(i, cost) for i in range(1)]

        env = DummyVecEnv(envs)

        if not os.path.exists('./sb_models'):
            os.mkdir('sb_models')
        if not os.path.exists('./sb_models/ckpt'):
            os.mkdir('sb_models/ckpt')

        log_path = os.path.join('exp_output',
                                'ppo_sb_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(cost))
        model = PPOSB(policy, env, **sb_args, tensorboard_log=log_path, verbose=0)

        model_filename = "sb_models/ppo2_" + env_name + format_name_string(str(spec)) + '_ctrl_cost_weight_' + str(cost)

        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=pref_ckpt_dir,
                                                 name_prefix='ppo_model')
        print("-" * 100)
        print(">>> SB Training for preference learning using checkpoint callback")
        print("-" * 100)
        model.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
        #     model = PPOSB(policy, env, **sb_args)#, tensorboard_log=log_path)
        model.save(model_filename)

for spec, costs in zip(env_kwargs, [[0.01, 0.008], [0.1, 0.17]]):
    for cost in costs:
        model_dir = './sb_models/ckpt_' + env_name + format_name_string(str(spec)) + '_' + str(cost)

        for model_file in os.listdir(model_dir):
            model = PPOSB.load(os.path.join(model_dir, model_file))
            save_expert_traj(env, model, spec,cost, pl_model_file=model_file)
