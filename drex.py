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
import json

### DG - WANDB for logging ###
import wandb

# wandb.init(project="DREX-IRM_vis_final", entity="dgarellick", name = f'Hopper_{args.spec}_{args.irm_coeff}',settings=wandb.Settings(start_method='fork'))
##############################

tqdm = partial(std_tqdm, dynamic_ncols=True, disable=eval(os.environ.get("DISABLE_TQDM", 'False')))

import gym

from bc_noise_dataset import BCNoisePreferenceDataset
from utils import RewardNet, Model

'''
Need to add the tensorboard logging for all the losses.
'''


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
    if args.irm_coeff > 0:
        log_dir = Path(args.log_dir) / f"trex_irm_{args.irm_coeff}"
    else:
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
    # if args.irm_coeff>0:
    #     losses = []
    #     accuracies = []
    #     irm_penalties = []

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

    # writer = tf.summary.FileWriter('./graphs', sess.graph) ADD THIS TO SESSION

    sess.run(init_op)

    for dataset in datasets:
        D = dataset.sample(args.D, include_action=args.include_action)

        # if args.irm_coeff>0:
        #     loss, acc, irm_penalty = model.train(D, iter=args.iter, l2_reg=args.l2_reg, irm_coeff=args.irm_coeff, noise_level=args.noise, debug=True)
        #     losses.append(loss)
        #     accuracies.append(acc)
        #     irm_penalties.append(irm_penalty)

        # else:
        model.train(D, iter=args.iter, l2_reg=args.l2_reg, irm_coeff=args.irm_coeff, noise_level=args.noise, debug=True)

        model.saver.save(sess, os.path.join(str(log_dir), 'model.ckpt'), write_meta_graph=False)
    #
    # if args.irm_coeff>0:
    #     train_loss = tf.stack(losses).mean()
    #     train_acc = tf.stack(accuracies).mean()
    #     train_penalty = tf.stack(irm_penalties).mean()
    #     total_loss = train_loss + args.irm_coeff * train_penalty
    #     if args.irm_coeff>1.0:
    #         total_loss /= args.irm_coeff

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
        if args.irm_coeff > 0:
            model.saver.restore(sess, os.path.join(args.log_dir, f"trex_irm_{args.irm_coeff}", 'model.ckpt'))
        else:
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

    # Draw Result
    def _draw(gt_returns, pred_returns, seen, figname=False):
        """
        gt_returns: [N] length
        pred_returns: [N] length
        seen: [N] length
        """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pylab
        from matplotlib import pyplot as plt
        from imgcat import imgcat

        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        plt.style.use('ggplot')
        params = {
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'legend.fontsize': 'xx-large',
            # 'figure.figsize': (6, 5),
            'axes.labelsize': 'xx-large',
            'axes.titlesize': 'xx-large',
            'xtick.labelsize': 'xx-large',
            'ytick.labelsize': 'xx-large'}
        matplotlib.pylab.rcParams.update(params)

        def _convert_range(x, minimum, maximum, a, b):
            return (x - minimum) / (maximum - minimum) * (b - a) + a

        def _no_convert_range(x, minimum, maximum, a, b):
            return x

        convert_range = _convert_range
        # convert_range = _no_convert_range

        gt_max, gt_min = max(gt_returns), min(gt_returns)
        pred_max, pred_min = max([sum(rewards) for rewards in pred_returns]), min(
            [sum(rewards) for rewards in pred_returns])
        max_observed = np.max(gt_returns[np.where(seen != 1)])

        # Draw P
        fig, ax = plt.subplots()

        # ax.plot(gt_returns[np.where(seen == 0)],
        #         [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 0)]],
        #         'go')  # unseen trajs

        y_seen = [convert_range(sum(p), pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 1)]]
        y_BC = [convert_range(sum(p), pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 2)]]

        ax.plot(gt_returns[np.where(seen == 1)],
                list(y_seen),
                'bo')  # [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 1)]],'bo')  # seen trajs for T-REX
        ax.plot(gt_returns[np.where(seen == 2)],
                list(y_BC),
                'ro')  # [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pred_returns[np.where(seen == 2)]],'ro')  # seen trajs for BC

        ax.plot([gt_min - 5, gt_max + 5], [gt_min - 5, gt_max + 5], 'k--')
        # ax.plot([gt_min-5,max_observed],[gt_min-5,max_observed],'k-', linewidth=2)
        # ax.set_xlim([gt_min-5,gt_max+5])
        # ax.set_ylim([gt_min-5,gt_max+5])
        ax.set_xlabel("Ground Truth Returns")
        ax.set_ylabel("Predicted Returns (normalized)")
        fig.tight_layout()

        plt.savefig(figname)
        plt.close()

    save_path = os.path.join(args.log_dir, f'gt_vs_pred_rewards_{args.irm_coeff}.pdf')
    _draw(np.array(gt_returns), np.array(pred_returns), np.array(seen), save_path)


def train_rl(args):
    # Train an agent
    import pynvml as N
    import subprocess, multiprocessing
    ncpu = multiprocessing.cpu_count()
    N.nvmlInit()
    ngpu = N.nvmlDeviceGetCount()

    name_temp = str(args.env_kwargs)
    print('name_temp: ', name_temp)
    if args.irm_coeff > 0:
        log_dir = Path(args.log_dir) / f'rl_{args.irm_coeff}_{name_temp}_seed_{args.seed}'
    else:
        log_dir = Path(args.log_dir) / f'rl_{name_temp}_seed_{args.seed}'

    log_dir.mkdir(parents=True, exist_ok='temp' in args.log_dir)

    if args.irm_coeff > 0:
        model_dir = os.path.join(args.log_dir, f"trex_irm_{args.irm_coeff}")
    else:
        model_dir = os.path.join(args.log_dir, f'trex')

    kwargs = {
        "model_dir": os.path.abspath(model_dir),
        "ctrl_coeff": args.ctrl_coeff,
        "alive_bonus": 0.
    }

    procs = []
    print('env_kwargs: ', args.env_kwargs)
    for i in range(args.rl_runs):
        # Prepare Command
        template = 'python -m baselines.run --alg=ppo2 --save_path={save_path} --env={env}  --num_env={nenv} --num_timesteps={num_timesteps} --save_interval={save_interval} --custom_reward {custom_reward} --custom_reward_kwargs="{kwargs}" --gamma {gamma} --seed {seed}'
        cmd = template.format(
            env=args.env_id,
            save_path=f'model/mu2/{args.env_id}_irm_{args.irm_coeff}_seed_{args.seed}',
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
    name_temp = str(args.env_kwargs)

    from utils import PPO2Agent, gen_traj

    print('evaluating reward RL trained or Reward')

    env = gym.make(args.env_id, **args.env_kwargs)
    env.seed(args.seed)

    def _get_perf(agent, num_eval=20):
        V = []
        sum_ctrl_cost = []
        sum_healthy_reward = []
        sum_forward_reward = []
        # ctrl_cost = []
        for _ in range(num_eval):
            _, _, R, control_cost, healthy_reward, forw_rew = gen_traj(env, agent, -1)
            V.append(np.sum(R))
            sum_ctrl_cost.append(np.sum(control_cost))
            sum_healthy_reward.append(np.sum(healthy_reward))
            sum_forward_reward.append(np.sum(forw_rew))
            # ctrl_cost.append(control_cost)
        return V, sum_ctrl_cost, sum_healthy_reward, sum_forward_reward  # , ctrl_cost

    with open(os.path.join(args.log_dir, 'rl_results_clip_action.txt'), 'w') as f:
        # Load T-REX learned agent
        if args.irm_coeff > 0:
            agents_dir = Path(
                os.path.abspath(os.path.join(args.log_dir, f'rl_{args.irm_coeff}_{name_temp}_seed_{args.seed}')))
        else:
            agents_dir = Path(os.path.abspath(os.path.join(args.log_dir, f'rl_{name_temp}_seed_{args.seed}')))

        trained_steps = sorted(list(set([path.name for path in agents_dir.glob('run_*/checkpoints/?????')])))
        for step in trained_steps[::-1]:
            perfs = []
            total_ctrl_costs = []
            healthy = []
            forward_reward = []
            for i in range(args.rl_runs):
                path = agents_dir / ('run_%d' % i) / 'checkpoints' / step

                if path.exists() == False:
                    continue

                agent = PPO2Agent(env, 'mujoco', str(path), stochastic=True)
                agent_perfs, cum_ctrl_cost, healthy_reward, sum_forward_reward = _get_perf(agent)
                wandb.log({'cummulative_reward': np.mean(agent_perfs), 'mean_ctrl_cost': np.mean(cum_ctrl_cost),
                           'std_ctrl_cost': np.std(cum_ctrl_cost), 'healthy reward': np.mean(healthy_reward),
                           'forward_reward': np.mean(sum_forward_reward)})
                print('[%s-%d] %f %f' % (step, i, np.mean(agent_perfs), np.std(agent_perfs)))
                print('[%s-%d] %f %f' % (step, i, np.mean(agent_perfs), np.std(agent_perfs)), file=f)

                perfs += agent_perfs
                total_ctrl_costs += cum_ctrl_cost
                healthy += healthy_reward
                forward_reward += sum_forward_reward
            print('[%s] %f %f %f %f' % (step, np.mean(perfs), np.std(perfs), np.max(perfs), np.min(perfs)))
            print('[%s] %f %f %f %f' % (step, np.mean(perfs), np.std(perfs), np.max(perfs), np.min(perfs)), file=f)
            wandb.log({'mean_cum_reward': np.mean(perfs), 'max_perf': np.max(perfs), 'min_perfs': np.min(perfs),
                       'mean_total_ctrl_cost': np.mean(total_ctrl_costs), 'mean total healthy': np.mean(healthy),
                       'mean forward reward': np.mean(forward_reward)})
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
    parser.add_argument('--env_kwargs', required=False, help='Run on intervened environment to run',
                        default={'xml_file': 'hopper_foot_mu5.xml'})
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

    # if args.env_kwargs == True:
    #     args.env_kwargs = {'xml_file': 'hopper_foot_mu5.xml'}
    # else:
    #     args.env_kwargs = {'xml_file': 'hopper.xml'}

    wandb.init(project="DREX-IRM_FINAL", entity="dgarellick",
               name=f'Hopper_{str(args.env_kwargs)}_{args.irm_coeff}',
               settings=wandb.Settings(start_method='fork'))

    if args.mode == 'train_reward':
        train_reward(args)
        tf.reset_default_graph()
        eval_reward(args)
    elif args.mode == 'eval_reward':
        eval_reward(args)
    elif args.mode == 'train_rl':
        print('in training rl: ',args.env_kwargs)
        train_rl(args)
        tf.reset_default_graph()
        eval_rl(args)
    elif args.mode == 'eval_rl':
        eval_rl(args)
    else:
        assert False
# ====
# import sys
# import os
# from pathlib import Path
# import argparse
# import pickle
# from functools import partial
# from pathlib import Path
# import numpy as np
# import tensorflow as tf
# from tqdm import tqdm as std_tqdm
# tqdm = partial(std_tqdm, dynamic_ncols=True, disable=eval(os.environ.get("DISABLE_TQDM", 'False')))
#
# import gym
#
# from bc_noise_dataset import BCNoisePreferenceDataset
# from utils import RewardNet, Model
#
# def train_reward(args):
#     # set random seed
#     np.random.seed(args.seed)
#     tf.random.set_random_seed(args.seed)
#
#     log_dir = Path(args.log_dir)/'trex'
#     log_dir.mkdir(parents=True,exist_ok='temp' in args.log_dir)
#
#     with open(str(log_dir/'args.txt'),'w') as f:
#         f.write( str(args) )
#
#     env = gym.make(args.env_id)
#     env.seed(args.seed)
#
#     ob_dims = env.observation_space.shape[-1]
#     ac_dims = env.action_space.shape[-1]
#
#     dataset = BCNoisePreferenceDataset(env,args.max_steps,args.min_noise_margin)
#
#     loaded = dataset.load_prebuilt(args.noise_injected_trajs)
#     assert loaded
#
#     models = []
#     for i in range(args.num_models):
#         with tf.variable_scope('model_%d'%i):
#             net = RewardNet(args.include_action,ob_dims,ac_dims,num_layers=args.num_layers,embedding_dims=args.embedding_dims)
#             model = Model(net,batch_size=64)
#             models.append(model)
#
#     ### Initialize Parameters
#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#     # Training configuration
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.InteractiveSession()
#
#     sess.run(init_op)
#
#     for i,model in enumerate(models):
#         D = dataset.sample(args.D,include_action=args.include_action)
#
#         model.train(D,iter=args.iter,l2_reg=args.l2_reg,noise_level=args.noise,debug=True)
#
#         model.saver.save(sess,os.path.join(str(log_dir),'model_%d.ckpt'%(i)),write_meta_graph=False)
#
#     sess.close()
#
# def eval_reward(args):
#     np.random.seed(args.seed)
#     tf.random.set_random_seed(args.seed)
#
#     env = gym.make(args.env_id)
#     env.seed(args.seed)
#
#     dataset = BCNoisePreferenceDataset(env)
#
#     loaded = dataset.load_prebuilt(args.noise_injected_trajs)
#     assert loaded
#
#     # Load Seen Trajs
#     seen_trajs = [
#         (obs,actions,rewards) for _,trajs in dataset.trajs for obs,actions,rewards in trajs
#     ]
#
#     # Load Unseen Trajectories
#     if args.unseen_trajs:
#         with open(args.unseen_trajs,'rb') as f:
#             unseen_trajs = pickle.load(f)
#     else:
#         unseen_trajs = []
#
#     # Load Demo Trajectories used for BC
#     with open(args.bc_trajs,'rb') as f:
#         bc_trajs = pickle.load(f)
#
#     # Load T-REX Reward Model
#     graph = tf.Graph()
#     config = tf.ConfigProto() # Run on CPU
#     config.gpu_options.allow_growth = True
#
#     with graph.as_default():
#         models = []
#         for i in range(args.num_models):
#             with tf.variable_scope('model_%d'%i):
#                 net = RewardNet(args.include_action,env.observation_space.shape[-1],env.action_space.shape[-1],num_layers=args.num_layers,embedding_dims=args.embedding_dims)
#
#                 model = Model(net,batch_size=1)
#                 models.append(model)
#
#     sess = tf.Session(graph=graph,config=config)
#     for i,model in enumerate(models):
#         with sess.as_default():
#             model.saver.restore(sess,os.path.join(args.log_dir,'trex','model_%d.ckpt'%i))
#
#     # Calculate Predicted Returns
#     def _get_return(obs,acs):
#         with sess.as_default():
#             return np.sum([model.get_reward(obs,acs) for model in models]) / len(models)
#
#     seen = [1] * len(seen_trajs) + [0] * len(unseen_trajs) + [2] * len(bc_trajs)
#     gt_returns, pred_returns = [], []
#
#     for obs,actions,rewards in seen_trajs+unseen_trajs+bc_trajs:
#         gt_returns.append(np.sum(rewards))
#         pred_returns.append(_get_return(obs,actions))
#     sess.close()
#
#
#
#     # Draw Result
#     def _draw(gt_returns,pred_returns,seen,figname=False):
#         """
#         gt_returns: [N] length
#         pred_returns: [N] length
#         seen: [N] length
#         """
#         import matplotlib
#         matplotlib.use('agg')
#         import matplotlib.pylab
#         from matplotlib import pyplot as plt
#         from imgcat import imgcat
#
#         matplotlib.rcParams.update(matplotlib.rcParamsDefault)
#         plt.style.use('ggplot')
#         params = {
#             'text.color':'black',
#             'axes.labelcolor':'black',
#             'xtick.color':'black',
#             'ytick.color':'black',
#             'legend.fontsize': 'xx-large',
#             #'figure.figsize': (6, 5),
#             'axes.labelsize': 'xx-large',
#             'axes.titlesize':'xx-large',
#             'xtick.labelsize':'xx-large',
#             'ytick.labelsize':'xx-large'}
#         matplotlib.pylab.rcParams.update(params)
#
#         def _convert_range(x,minimum, maximum,a,b):
#             return (x - minimum)/(maximum - minimum) * (b - a) + a
#
#         def _no_convert_range(x,minimum, maximum,a,b):
#             return x
#
#         convert_range = _convert_range
#         #convert_range = _no_convert_range
#
#         gt_max,gt_min = max(gt_returns),min(gt_returns)
#         pred_max,pred_min = max(pred_returns),min(pred_returns)
#         max_observed = np.max(gt_returns[np.where(seen!=1)])
#
#         # Draw P
#         fig,ax = plt.subplots()
#
#         # ax.plot(gt_returns[np.where(seen==0)],
#         #         [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen==0)]], 'go') # unseen trajs
#         ax.plot(gt_returns[np.where(seen==1)],
#                 [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen==1)]], 'bo') # seen trajs for T-REX
#         ax.plot(gt_returns[np.where(seen==2)],
#                 [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen==2)]], 'ro') # seen trajs for BC
#
#         ax.plot([gt_min-5,gt_max+5],[gt_min-5,gt_max+5],'k--')
#         #ax.plot([gt_min-5,max_observed],[gt_min-5,max_observed],'k-', linewidth=2)
#         #ax.set_xlim([gt_min-5,gt_max+5])
#         #ax.set_ylim([gt_min-5,gt_max+5])
#         ax.set_xlabel("Ground Truth Returns")
#         ax.set_ylabel("Predicted Returns (normalized)")
#         fig.tight_layout()
#
#         plt.savefig(figname)
#         plt.close()
#
#     save_path = os.path.join(args.log_dir,'gt_vs_pred_rewards.pdf')
#     _draw(np.array(gt_returns),np.array(pred_returns),np.array(seen),save_path)
#
# def train_rl(args):
#     # Train an agent
#     import pynvml as N
#     import subprocess, multiprocessing
#     ncpu = multiprocessing.cpu_count()
#     N.nvmlInit()
#     ngpu = N.nvmlDeviceGetCount()
#
#     log_dir = Path(args.log_dir)/'rl'
#     log_dir.mkdir(parents=True,exist_ok='temp' in args.log_dir)
#
#     model_dir = os.path.join(args.log_dir,'trex')
#
#     kwargs = {
#         "model_dir":os.path.abspath(model_dir),
#         "ctrl_coeff":args.ctrl_coeff,
#         "alive_bonus": 0.
#     }
#
#     procs = []
#     for i in range(args.rl_runs):
#         # Prepare Command
#         template = 'python -m baselines.run --alg=ppo2 --env={env} --num_env={nenv} --num_timesteps={num_timesteps} --save_interval={save_interval} --custom_reward {custom_reward} --custom_reward_kwargs="{kwargs}" --gamma {gamma} --seed {seed}'
#
#         cmd = template.format(
#             env=args.env_id,
#             nenv=1, #ncpu//ngpu,
#             num_timesteps=args.num_timesteps,
#             save_interval=args.save_interval,
#             custom_reward='preference_normalized_v2',
#             gamma=args.gamma,
#             seed=i,
#             kwargs=str(kwargs)
#         )
#
#         # Prepare Log settings through env variables
#         env = os.environ.copy()
#         env["OPENAI_LOGDIR"] = os.path.join(str(log_dir.resolve()),'run_%d'%i)
#         if i == 0:
#             env["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'
#             p = subprocess.Popen(cmd, cwd='./learner/baselines', stdout=subprocess.PIPE, env=env, shell=True)
#         else:
#             env["OPENAI_LOG_FORMAT"] = 'log,csv,tensorboard'
#             p = subprocess.Popen(cmd, cwd='./learner/baselines', env=env, shell=True)
#
#         # run process
#         procs.append(p)
#
#     for line in procs[0].stdout:
#         print(line.decode(),end='')
#
#     for p in procs[1:]:
#         p.wait()
#
# def eval_rl(args):
#     np.random.seed(args.seed)
#     tf.random.set_random_seed(args.seed)
#
#     from utils import PPO2Agent, gen_traj
#
#     env = gym.make(args.env_id)
#     env.seed(args.seed)
#
#     def _get_perf(agent, num_eval=20):
#         V = []
#         for _ in range(num_eval):
#             _,_,R = gen_traj(env,agent,-1)
#             V.append(np.sum(R))
#         return V
#
#     with open(os.path.join(args.log_dir,'rl_results_clip_action.txt'),'w') as f:
#         # Load T-REX learned agent
#         agents_dir = Path(os.path.abspath(os.path.join(args.log_dir,'rl')))
#
#         trained_steps = sorted(list(set([path.name for path in agents_dir.glob('run_*/checkpoints/?????')])))
#         for step in trained_steps[::-1]:
#             perfs = []
#             for i in range(args.rl_runs):
#                 path = agents_dir/('run_%d'%i)/'checkpoints'/step
#
#                 if path.exists() == False:
#                     continue
#
#                 agent = PPO2Agent(env,'mujoco',str(path),stochastic=True)
#                 agent_perfs = _get_perf(agent)
#                 print('[%s-%d] %f %f'%(step,i,np.mean(agent_perfs),np.std(agent_perfs)))
#                 print('[%s-%d] %f %f'%(step,i,np.mean(agent_perfs),np.std(agent_perfs)),file=f)
#
#                 perfs += agent_perfs
#             print('[%s] %f %f %f %f'%(step,np.mean(perfs),np.std(perfs),np.max(perfs),np.min(perfs)))
#             print('[%s] %f %f %f %f'%(step,np.mean(perfs),np.std(perfs),np.max(perfs),np.min(perfs)),file=f)
#
#             f.flush()
#
# if __name__ == "__main__":
#     # Required Args (target envs & learners)
#     parser = argparse.ArgumentParser(description=None)
#     parser.add_argument('--seed', default=0, type=int, help='seed for the experiments')
#     parser.add_argument('--log_dir', required=True)
#     parser.add_argument('--env_id', required=True, help='Select the environment to run')
#     parser.add_argument('--mode', default='train_reward',choices=['all','train_reward','eval_reward','train_rl','eval_rl'])
#     # Args for T-REX
#     ## Dataset setting
#     parser.add_argument('--noise_injected_trajs', default='')
#     parser.add_argument('--unseen_trajs', default='', help='used for evaluation only')
#     parser.add_argument('--bc_trajs', default='', help='used for evaluation only')
#     parser.add_argument('--D', default=5000, type=int, help='|D| in the preference paper')
#     parser.add_argument('--max_steps', default=50, type=int, help='maximum length of subsampled trajecotry')
#     parser.add_argument('--min_noise_margin', default=0.3, type=float, help='')
#     parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
#     ## Network setting
#     parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
#     parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
#     parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
#     parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
#     parser.add_argument('--irm_coeff', default=0, type=float, help='irm regularization size')
#     parser.add_argument('--noise', default=0.0, type=float, help='noise level to add on training label (another regularization)')
#     parser.add_argument('--iter', default=3000, type=int, help='# trainig iters')
#     # Args for PPO
#     parser.add_argument('--rl_runs', default=3, type=int)
#     parser.add_argument('--num_timesteps', default=int(1e6), type=int)
#     parser.add_argument('--save_interval', default=20, type=int)
#     parser.add_argument('--ctrl_coeff', default=0.0, type=float)
#     parser.add_argument('--gamma', default=0.99, type=float)
#     args = parser.parse_args()
#
#     if args.mode == 'train_reward':
#         train_reward(args)
#         tf.reset_default_graph()
#         eval_reward(args)
#     elif args.mode == 'eval_reward':
#         eval_reward(args)
#     elif args.mode =='train_rl':
#         train_rl(args)
#         tf.reset_default_graph()
#         eval_rl(args)
#     elif args.mode == 'eval_rl':
#         eval_rl(args)
#     else:
#         assert False
#
#         # Plot ctrl cost
#         # Try for a completely different friction out of intervention
#
# >>>>>>> master
