import sys
import os
from pathlib import Path
import argparse
import pickle
from functools import partial
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True, disable=eval(os.environ.get("DISABLE_TQDM", 'False')))

import gym

from bc_noise_dataset import BCNoisePreferenceDataset
from utils import RewardNet, Model


def format_name_string(name_string):
    name_string = name_string.replace('{', '_').replace('}', '').replace(' ', '').replace("'xml_file'", '')
    name_string = name_string.replace("'", "").replace(":", "").replace('/', '')
    name_string = name_string.replace(".xml", "")
    name_string = name_string.replace("_hopper_foot_", "")

    return name_string


def format_name_cost(name_cost):
    name_cost = name_cost.replace('.', '')

    return name_cost


def train_reward(args):
    # set random seed
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    log_dir = Path(args.log_dir) / 'trex'
    log_dir.mkdir(parents=True, exist_ok='temp' in args.log_dir)

    with open(str(log_dir / 'args.txt'), 'w') as f:
        f.write(str(args))

    envs = []
    model = None
    ob_dims = None
    ac_dims = None
    datasets = []
    env_kwargs = [{'xml_file': 'hopper_foot_mu1.xml'}, {'xml_file': 'hopper_foot_mu3.xml'}]
    loss = []
    acc = []
    irm_loss = []

    for spec, costs in zip(env_kwargs, [[0.01, 0.008], [0.1, 0.17]]):
        for cost in costs:
            env = gym.make(args.env_id, ctrl_cost_weight=args.ctrl_cost, **spec)
            env.seed(args.seed)
            envs.append(env)

            ob_dims = env.observation_space.shape[-1]
            ac_dims = env.action_space.shape[-1]

            dataset = BCNoisePreferenceDataset(env, args.max_steps, args.min_noise_margin)
            friction_name = format_name_string(str(spec))
            cost_name = format_name_cost(str(cost))
            noise_injected_traj = f'./log/drex/hopper/noisy_{friction_name}penalty{cost_name}/prebuilt.pkl'
            loaded = dataset.load_prebuilt(noise_injected_traj)
            datasets.append(dataset)
            assert loaded

    with tf.variable_scope('model'):
        net = RewardNet(args.include_action, ob_dims, ac_dims, num_layers=args.num_layers,
                        embedding_dims=args.embedding_dims)
        model = Model(net, batch_size=64)

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    for dataset in datasets:
        D = dataset.sample(args.D, include_action=args.include_action)

        model.train(D, iter=args.iter, l2_reg=args.l2_reg, irm_coeff=args.irm_coeff, noise_level=args.noise, debug=True)
        if args.irm_coeff>0:

        model.saver.save(sess, os.path.join(str(log_dir), 'model.ckpt'), write_meta_graph=False)

    sess.close()


def eval_reward(args):
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    envs = []
    model = None
    ob_dims = None
    ac_dims = None
    datasets = []
    seen_trajs_envs = []
    env_kwargs = [{'xml_file': 'hopper_foot_mu1.xml'}, {'xml_file': 'hopper_foot_mu3.xml'}]

    for spec, costs in zip(env_kwargs, [[0.01, 0.008], [0.1, 0.17]]):
        for cost in costs:
            env = gym.make(args.env_id, ctrl_cost_weight=args.ctrl_cost, **spec)
            env.seed(args.seed)
            envs.append(env)

            ob_dims = env.observation_space.shape[-1]
            ac_dims = env.action_space.shape[-1]

            dataset = BCNoisePreferenceDataset(env, args.max_steps, args.min_noise_margin)
            friction_name = format_name_string(str(spec))
            cost_name = format_name_cost(str(cost))
            noise_injected_traj = f'./log/drex/hopper/noisy_{friction_name}penalty{cost_name}/prebuilt.pkl'
            loaded = dataset.load_prebuilt(noise_injected_traj)
            datasets.append(dataset)
            assert loaded

            # Load Seen Trajs
            seen_trajs = [
                (obs, actions, rewards) for _, trajs in dataset.trajs for obs, actions, rewards in trajs
            ]
            seen_trajs_envs.append(seen_trajs)

            # Load Unseen Trajectories
            if args.unseen_trajs:
                with open(args.unseen_trajs, 'rb') as f:
                    unseen_trajs = pickle.load(f)
            else:
                unseen_trajs = []

            # Load Demo Trajectories used for BC
            cost_str = str(cost)
            example_traj = f'./demos/suboptimal_demos/hopper/Hopper_intervened/{args.env_id}_hopper_foot_{friction_name}{cost_str}ppo_model_860000_steps.zip_expert_traj.pkl'
            with open(example_traj, 'rb') as f:
                bc_trajs = pickle.load(f)

    # Load T-REX Reward Model
    graph = tf.Graph()
    config = tf.ConfigProto()  # Run on CPU
    config.gpu_options.allow_growth = True

    with graph.as_default():
        with tf.variable_scope('model'):
            net = RewardNet(args.include_action, env.observation_space.shape[-1], env.action_space.shape[-1],
                            num_layers=args.num_layers, embedding_dims=args.embedding_dims)

            model = Model(net, batch_size=1)

    sess = tf.Session(graph=graph, config=config)

    with sess.as_default():
        model.saver.restore(sess, os.path.join(args.log_dir, 'trex', 'model.ckpt'))

    # Calculate Predicted Returns
    def _get_return(obs, acs):
        with sess.as_default():
            return model.get_reward(obs, acs)

    seen = [1] * len(seen_trajs) + [0] * len(unseen_trajs) + [2] * len(bc_trajs)
    gt_returns, pred_returns = [], []

    for obs, actions, rewards in seen_trajs + unseen_trajs + bc_trajs:
        gt_returns.append(np.sum(rewards))
        pred_returns.append(_get_return(obs, actions))
    sess.close()

    # # Draw Result
    # def _draw(gt_returns, pred_returns, seen, figname=False):
    #     """
    #     gt_returns: [N] length
    #     pred_returns: [N] length
    #     seen: [N] length
    #     """
    #     import matplotlib
    #     matplotlib.use('agg')
    #     import matplotlib.pylab
    #     from matplotlib import pyplot as plt
    #     from imgcat import imgcat
    #
    #     matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    #     plt.style.use('ggplot')
    #     params = {
    #         'text.color': 'black',
    #         'axes.labelcolor': 'black',
    #         'xtick.color': 'black',
    #         'ytick.color': 'black',
    #         'legend.fontsize': 'xx-large',
    #         # 'figure.figsize': (6, 5),
    #         'axes.labelsize': 'xx-large',
    #         'axes.titlesize': 'xx-large',
    #         'xtick.labelsize': 'xx-large',
    #         'ytick.labelsize': 'xx-large'}
    #     matplotlib.pylab.rcParams.update(params)
    #
    #     def _convert_range(x, minimum, maximum, a, b):
    #         return (x - minimum) / (maximum - minimum) * (b - a) + a
    #
    #     def _no_convert_range(x, minimum, maximum, a, b):
    #         return x
    #
    #     convert_range = _convert_range
    #     # convert_range = _no_convert_range
    #
    #     gt_max, gt_min = max(gt_returns), min(gt_returns)
    #     pred_max, pred_min = max([sum(rewards) for rewards in pred_returns]), min([sum(rewards) for rewards in pred_returns])
    #     max_observed = np.max(gt_returns[np.where(seen != 1)])
    #
    #     # Draw P
    #     fig, ax = plt.subplots()
    #
    #     ax.plot(gt_returns[np.where(seen == 0)],
    #             [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 0)]],
    #             'go')  # unseen trajs
    #     ax.plot(gt_returns[np.where(seen == 1)],
    #             [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 1)]],
    #             'bo')  # seen trajs for T-REX
    #     ax.plot(gt_returns[np.where(seen == 2)],
    #             [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 2)]],
    #             'ro')  # seen trajs for BC
    #
    #     ax.plot([gt_min - 5, gt_max + 5], [gt_min - 5, gt_max + 5], 'k--')
    #     # ax.plot([gt_min-5,max_observed],[gt_min-5,max_observed],'k-', linewidth=2)
    #     # ax.set_xlim([gt_min-5,gt_max+5])
    #     # ax.set_ylim([gt_min-5,gt_max+5])
    #     ax.set_xlabel("Ground Truth Returns")
    #     ax.set_ylabel("Predicted Returns (normalized)")
    #     fig.tight_layout()
    #
    #     plt.savefig(figname)
    #     plt.close()
    #
    # save_path = os.path.join(args.log_dir, 'gt_vs_pred_rewards.pdf')
    # _draw(np.array(gt_returns), np.array(pred_returns), np.array(seen), save_path)


def train_rl(args):
    # Train an agent
    import pynvml as N
    import subprocess, multiprocessing
    ncpu = multiprocessing.cpu_count()
    N.nvmlInit()
    ngpu = N.nvmlDeviceGetCount()

    log_dir = Path(args.log_dir) / 'rl'
    log_dir.mkdir(parents=True, exist_ok='temp' in args.log_dir)

    model_dir = os.path.join(args.log_dir, 'trex')

    kwargs = {
        "model_dir": os.path.abspath(model_dir),
        "ctrl_coeff": args.ctrl_coeff,
        "alive_bonus": 0.
    }

    procs = []
    for i in range(args.rl_runs):
        # Prepare Command
        template = 'python -m baselines.run --alg=ppo2 --env={env} --num_env={nenv} --num_timesteps={num_timesteps} --save_interval={save_interval} --custom_reward {custom_reward} --custom_reward_kwargs="{kwargs}" --gamma {gamma} --seed {seed}'

        cmd = template.format(
            env=args.env_id,
            nenv=1,  # ncpu//ngpu,
            num_timesteps=args.num_timesteps,
            save_interval=args.save_interval,
            custom_reward='preference_normalized_v3',
            gamma=args.gamma,
            seed=i,
            kwargs=str(kwargs)
        )

        # Prepare Log settings through env variables
        env = os.environ.copy()
        env["OPENAI_LOGDIR"] = os.path.join(str(log_dir.resolve()), 'run_%d' % i)
        if i == 0:
            env["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'
            p = subprocess.Popen(cmd, cwd='./learner/baselines', stdout=subprocess.PIPE, env=env, shell=True)
        else:
            env["OPENAI_LOG_FORMAT"] = 'log,csv,tensorboard'
            p = subprocess.Popen(cmd, cwd='./learner/baselines', env=env, shell=True)

        # run process
        procs.append(p)

    for line in procs[0].stdout:
        print(line.decode(), end='')

    for p in procs[1:]:
        p.wait()


def eval_rl(args):
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    from utils import PPO2Agent, gen_traj

    env = gym.make(args.env_id)
    env.seed(args.seed)

    def _get_perf(agent, num_eval=20):
        V = []
        for _ in range(num_eval):
            _, _, R = gen_traj(env, agent, -1)
            V.append(np.sum(R))
        return V

    with open(os.path.join(args.log_dir, 'rl_results_clip_action.txt'), 'w') as f:
        # Load T-REX learned agent
        agents_dir = Path(os.path.abspath(os.path.join(args.log_dir, 'rl')))

        trained_steps = sorted(list(set([path.name for path in agents_dir.glob('run_*/checkpoints/?????')])))
        for step in trained_steps[::-1]:
            perfs = []
            for i in range(args.rl_runs):
                path = agents_dir / ('run_%d' % i) / 'checkpoints' / step

                if path.exists() == False:
                    continue

                agent = PPO2Agent(env, 'mujoco', str(path), stochastic=True)
                agent_perfs = _get_perf(agent)
                print('[%s-%d] %f %f' % (step, i, np.mean(agent_perfs), np.std(agent_perfs)))
                print('[%s-%d] %f %f' % (step, i, np.mean(agent_perfs), np.std(agent_perfs)), file=f)

                perfs += agent_perfs
            print('[%s] %f %f %f %f' % (step, np.mean(perfs), np.std(perfs), np.max(perfs), np.min(perfs)))
            print('[%s] %f %f %f %f' % (step, np.mean(perfs), np.std(perfs), np.max(perfs), np.min(perfs)), file=f)

            f.flush()


if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help='seed for the experiments')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--env_id', required=True, help='Select the environment to run')
    parser.add_argument('--mode', default='train_reward',
                        choices=['all', 'train_reward', 'eval_reward', 'train_rl', 'eval_rl'])
    parser.add_argument('--ctrl_cost', required=False, help='Select the environment to run', default=0.001, type=float)
    parser.add_argument('--spec', required=False, help='Intervened environment to run',
                        default="{'xml_file': 'hopper.xml'}")
    # Args for T-REX
    ## Dataset setting
    # parser.add_argument('--noise_injected_trajs', default='')
    parser.add_argument('--unseen_trajs', default='', help='used for evaluation only')
    # parser.add_argument('--bc_trajs', default='', help='used for evaluation only')
    parser.add_argument('--D', default=5000, type=int, help='|D| in the preference paper')
    parser.add_argument('--max_steps', default=50, type=int, help='maximum length of subsampled trajecotry')
    parser.add_argument('--min_noise_margin', default=0.3, type=float, help='')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    ## Network setting
    parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--irm_coeff', default=0, type=float, help='irm coefficient size')
    parser.add_argument('--noise', default=0.0, type=float,
                        help='noise level to add on training label (another regularization)')
    parser.add_argument('--iter', default=3000, type=int, help='# trainig iters')
    parser.add_argument('--num_train_envs', default=4, type=int,
                        help='number of intervened training environments to train on')

    # Args for PPO
    parser.add_argument('--rl_runs', default=3, type=int)
    parser.add_argument('--num_timesteps', default=int(1e6), type=int)
    parser.add_argument('--save_interval', default=20, type=int)
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    args = parser.parse_args()

    if args.mode == 'train_reward':
        train_reward(args)
        tf.reset_default_graph()
        eval_reward(args)
    elif args.mode == 'eval_reward':
        eval_reward(args)
    elif args.mode == 'train_rl':
        train_rl(args)
        tf.reset_default_graph()
        eval_rl(args)
    elif args.mode == 'eval_rl':
        eval_rl(args)
    else:
        assert False
