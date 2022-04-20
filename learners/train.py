from imitation.algorithms import adversarial, bc
from imitation.util import logger, util
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common import policies
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.rewards import discrim_nets
import numpy as np
import argparse
from utils import make_sa_dataloader, make_sads_dataloader, make_sa_dataset, linear_schedule, fetch_dataset_size, gradient_penalty
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envwrapper import BasicWrapper
from mimicmd import *
import os
from gym.spaces import Discrete
import gym
import pybullet_envs
from stable_baselines3.common.running_mean_std import RunningMeanStd
from optim import OAdam
from soft_q import SQLPolicy
from sqil import SQILReplayBuffer
from adril import AdRILWrapper, AdRILReplayBuffer
from advil import advil_training
from wgail import *

def train_bc(env, n=0):
    venv = util.make_vec_env(env, n_envs=8)
    if isinstance(venv.action_space, Discrete):
        w = 64
    else:
        w = 256

    for i in range(n):
        mean_rewards = []
        std_rewards = []
        for num_trajs in range(0, 26, 5):
            if num_trajs == 0:
                expert_data = make_sa_dataloader(env, normalize=False)
            else:
                expert_data = make_sa_dataloader(env, max_trajs=num_trajs, normalize=False)
            bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=expert_data,
                               policy_class=policies.ActorCriticPolicy,
                               ent_weight=0., l2_weight=0., policy_kwargs=dict(net_arch=[w, w]))
            if num_trajs > 0:
                bc_trainer.train(n_batches=int(5e5))

            def get_policy(*args, **kwargs):
                return bc_trainer.policy
            model = PPO(get_policy, env, verbose=1)
            model.save(os.path.join("learners", env,
                                    "bc_{0}_{1}".format(i, num_trajs)))
            mean_reward, std_reward = evaluate_policy(
                    model, model.get_env(), n_eval_episodes=10)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("{0} Trajs: {1}".format(num_trajs, mean_reward))
            np.savez(os.path.join("learners", env, "bc_rewards_{0}".format(
                i)), means=mean_rewards, stds=std_rewards)


def train_gail(env, n=0):
    venv = util.make_vec_env(env, n_envs=8)
    if isinstance(venv.action_space, Discrete):
        w = 64
    else:
        w = 256
    expert_data = make_sads_dataloader(env, max_trajs=None)
    logger.configure(os.path.join("learners", "GAIL"))

    for i in range(n):
        discrim_net = discrim_nets.ActObsMLP(
                action_space=venv.action_space,
                observation_space=venv.observation_space,
                hid_sizes=(w, w),
                )
        gail_trainer = adversarial.GAIL(venv, expert_data=expert_data, expert_batch_size=32,
                                        gen_algo=PPO("MlpPolicy", venv, verbose=1, n_steps=1024,
                                                     policy_kwargs=dict(net_arch=[w, w])),
                                                     discrim_kwargs={'discrim_net': discrim_net})
        mean_rewards = []
        std_rewards = []
        for train_steps in range(400):
            if train_steps > 0:
                if 'Bullet' in env:
                    gail_trainer.train(total_timesteps=25000)
                else:
                    gail_trainer.train(total_timesteps=16384)

            def get_policy(*args, **kwargs):
                return gail_trainer.gen_algo.policy
            model = PPO(get_policy, env, verbose=1)
            mean_reward, std_reward = evaluate_policy(
                model, model.env, n_eval_episodes=10)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("{0} Steps: {1}".format(train_steps, mean_reward))
            np.savez(os.path.join("learners", env, "gail_results", "gail_rewards_{0}".format(i)),
                     means=mean_rewards, stds=std_rewards)

def train_sqil(env, n=0):
    venv = gym.make(env)
    expert_data = make_sa_dataset(env, max_trajs=5)

    for i in range(n):
        if isinstance(venv.action_space, Discrete):
            model = DQN(SQLPolicy, venv, verbose=1, policy_kwargs=dict(net_arch=[64, 64]), learning_starts=1)
        else:
            model = SAC('MlpPolicy', venv, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), ent_coef='auto',
                        learning_rate=linear_schedule(7.3e-4), train_freq=64, gradient_steps=64, gamma=0.98, tau=0.02)

        model.replay_buffer = SQILReplayBuffer(model.buffer_size, model.observation_space,
                                               model.action_space, model.device, 1,
                                               model.optimize_memory_usage, expert_data=expert_data)
        mean_rewards = []
        std_rewards = []
        for train_steps in range(20):
            if train_steps > 0:
                if 'Bullet' in env:
                    model.learn(total_timesteps=25000, log_interval=1)
                else:
                    model.learn(total_timesteps=16384, log_interval=1)
            mean_reward, std_reward = evaluate_policy(
                model, model.env, n_eval_episodes=10)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("{0} Steps: {1}".format(train_steps, mean_reward))
            np.savez(os.path.join("learners", env, "sqil_rewards_{0}".format(i)),
                     means=mean_rewards, stds=std_rewards)


def train_adril(env, n=0, balanced=False):
    num_trajs = 20
    expert_data = make_sa_dataset(env, max_trajs=num_trajs)
    n_expert = len(expert_data["obs"])
    expert_sa = np.concatenate((expert_data["obs"], np.reshape(expert_data["acts"], (n_expert, -1))), axis=1)

    for i in range(0, n):
        venv = AdRILWrapper(gym.make(env))
        mean_rewards = []
        std_rewards = []
        # Create model
        if isinstance(venv.action_space, Discrete):
            model = DQN(SQLPolicy, venv, verbose=1, policy_kwargs=dict(net_arch=[64, 64]), learning_starts=1)
        else:
            model = SAC('MlpPolicy', venv, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), ent_coef='auto',
                        learning_rate=linear_schedule(7.3e-4), train_freq=64, gradient_steps=64, gamma=0.98, tau=0.02)
        model.replay_buffer = AdRILReplayBuffer(model.buffer_size, model.observation_space,
                                               model.action_space, model.device, 1,
                                               model.optimize_memory_usage, expert_data=expert_data, N_expert=num_trajs,
                                               balanced=balanced)
        if not balanced:
            for j in range(len(expert_sa)):
                obs = expert_data["obs"][j]
                act = expert_data["acts"][j]
                next_obs = expert_data["next_obs"][j]
                done = expert_data["dones"][j]
                model.replay_buffer.add(obs, next_obs, act, -1, done)
        for train_steps in range(400):
            # Train policy
            if train_steps > 0:
                if 'Bullet' in env:
                    model.learn(total_timesteps=1250, log_interval=1000)
                else:
                    model.learn(total_timesteps=25000, log_interval=1000)
                if train_steps % 1 == 0: # written to support more complex update schemes
                    model.replay_buffer.set_iter(train_steps)
                    model.replay_buffer.set_n_learner(venv.num_trajs)

            # Evaluate policy
            if train_steps % 20 == 0:
                model.set_env(gym.make(env))
                mean_reward, std_reward = evaluate_policy(
                    model, model.env, n_eval_episodes=10)
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
                print("{0} Steps: {1}".format(int(train_steps * 1250), mean_reward))
                np.savez(os.path.join("learners", env, "adril_rewards_{0}".format(i)),
                        means=mean_rewards, stds=std_rewards)
            # Update env
            if train_steps > 0:
                if train_steps % 1  == 0:
                    venv.set_iter(train_steps + 1)
            model.set_env(venv)


def train_advil(env, n=0):
    venv = gym.make(env)
    for i in range(n):
        mean_rewards = []
        std_rewards = []
        for num_trajs in range(0, 26, 5):
            if num_trajs == 0:
                expert_data = make_sa_dataloader(env, normalize=True)
                pi = advil_training(expert_data, venv, iters=0)
            else:
                expert_data = make_sa_dataloader(env, max_trajs=num_trajs, normalize=True, batch_size=1024)
                pi = advil_training(expert_data, venv)
            def get_policy(*args, **kwargs):
                return pi
            model = PPO(get_policy, env, verbose=1)
            mean_reward, std_reward = evaluate_policy(
                model, model.get_env(), n_eval_episodes=10)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("{0} Trajs: {1}".format(num_trajs, mean_reward))
            np.savez(os.path.join("learners", env, "advil_rewards_{0}".format(
                i)), means=mean_rewards, stds=std_rewards)

def train_wgail(env, n=0):
    for i in range(n):
        mean_rewards = []
        std_rewards = []

        expert_obs, expert_acts = make_sa_dataloader(env, normalize=False, raw=True, raw_traj=False)
        expert_obs, expert_acts = np.array(expert_obs), np.array(expert_acts)
        expert_sa_pairs = torch.cat((torch.tensor(expert_obs), torch.tensor(expert_acts)), axis=1)

        wgail_instance = WGAIL(env)
        wgail_instance.train(expert_sa_pairs, expert_obs, expert_acts)

def train_mimicmd(env, n=0, bc_steps=0):
    #The MIMIC-MD instance
    venv = util.make_vec_env(env, n_envs=8)

    if isinstance(venv.action_space, Discrete):
        w = 64
    else:
        w = 256

    #parameters
    #choose from ["ensemble", "rnd", "expert", "maxdist"]
    distance_metric = "ensemble"

    #beta is used for the sigmoid function, this needs to be tuned for each new set of expert dems and generated bc data
    beta = 0.1

    mimicmd_instance = MIMICMD(gym.make(env), venv, beta)
    total_expert_size = fetch_dataset_size(env)
    action_space_size = venv.action_space.shape[0]
    obs_space_size = venv.observation_space.shape[0]

    ratio_d1_d2 = 0.5
    expert_traj_numbers = 12
    traj_numbers = np.random.choice(total_expert_size, expert_traj_numbers, replace=False)
    d1_size = traj_numbers[:int(expert_traj_numbers*ratio_d1_d2)]
    d2_size = traj_numbers[int(expert_traj_numbers*ratio_d1_d2):]

    hidden = 256
    outer_steps = 250
    inner_steps = 2

    num_traj_sample = 4
    batch_size = 2
    learn_rate = 8e-5
    bc_steps = 10e5

    #Choose to use existing generated bc data or create [target_samps] new trajectories
    target_samps = 10
    load_d3 = False

    #Choose to use existing query nets or train new ones
    load_query = True
    save_rewards = True
    save_inner_model = False

    mean_rewards = []
    std_rewards = []

    #Step 2:
    mimicmd_instance.train_query_policy(venv, w, env, d1_size, bc_steps, load_query, distance_metric, num_trajs=d1_size)

    #Step 3 building the membership oracle includes steps 4-5 below
    #After building the oracle we can now just have the raw trajectories instead of raw obs and actions
    d1_obs, d1_acts = make_sa_dataloader(env, max_trajs=d1_size, normalize=False, raw=True, raw_traj=False)
    d2_obs, d2_acts = make_sa_dataloader(env, max_trajs=d2_size, normalize=False, raw=True, raw_traj=False)

    #Step 3 a building D3
    mimicmd_instance.generate_d3(env, target_samps, load=load_d3)
    mimicmd_instance.fetch_avg_dist(d1_obs, distance_metric, expert_acts=d1_acts)

    ######MIMICMD TRAINING####################################################################################
    #Step 4 We define the f function here
    pseudoreward = MIMICMDDiscriminator(gym.make(env))
    pseudoreward_optimizer = OAdam(pseudoreward.parameters(), lr=learn_rate)

    #wrapped environment with modified reward -> -f function is the reward
    wrapped_env = BasicWrapper(gym.make(env), pseudoreward)

    #initialize replay buffer
    mimic_replay_buffer = MIMICMDReplayBuffer(wrapped_env.observation_space.shape[0], wrapped_env.action_space.shape[0])
    mimicmd_instance.replay_buffer = mimic_replay_buffer

    #Step 4 learner policy to optimize
    model = SAC('MlpPolicy', wrapped_env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), ent_coef='auto',
                learning_rate=linear_schedule(7.3e-4), train_freq=64, gradient_steps=64, gamma=0.98, tau=0.02)

    #expert sa pairs for use in the gradient penalty
    expert_obs, expert_acts = make_sa_dataloader(env, max_trajs=traj_numbers, normalize=False, raw=True, raw_traj=False)
    expert_obs, expert_acts = np.array(expert_obs), np.array(expert_acts)
    expert_sa_pairs = torch.cat((torch.tensor(expert_obs), torch.tensor(expert_acts)), axis=1)

    #Step 4 training, outer maximizing the pseudo reward (f) function and inner steps are the minimization of policy over f
    for outer in range(outer_steps):
        learning_rate_used = learn_rate

        #f network
        pseudoreward_optimizer = OAdam(pseudoreward.parameters(), lr=learning_rate_used)
        model.learn(total_timesteps=inner_steps, log_interval=1)

        if save_inner_model:
            model.save("sac_mimicmd_training_model")

        #sample some more sa pairs using current model
        mimicmd_instance.sample_and_add(wrapped_env, model, num_traj_sample)

        #sample from replay buffer
        low = wrapped_env.action_space.low
        high = wrapped_env.action_space.high
        obs_samples, act_samples = mimic_replay_buffer.sample(batch_size)
        act_samples = (((act_samples - low) / (high - low)) * 2.0) - 1.0
        sa_samples = torch.cat((torch.tensor(obs_samples), torch.tensor(act_samples)), axis=1)

        #Do the outer step - min C(f) - E_model(f)
        pseudoreward_optimizer.zero_grad()

        prog = outer/outer_steps

        c = mimicmd_instance.cost(pseudoreward, d2_obs, d2_acts, d1_obs, d1_acts, distance_metric)
        learner_f_under_model = torch.mean(pseudoreward.forward(torch.tensor(sa_samples, dtype=torch.float)))

        random_sample = np.random.choice(len(expert_obs), len(obs_samples), replace=False)
        expert_sa_pairs = torch.cat((torch.tensor(expert_obs[random_sample]), torch.tensor(expert_acts[random_sample])), axis=1)
        gp = gradient_penalty(sa_samples, expert_sa_pairs, pseudoreward)

        #Maximize is same as minimize -(obj)
        obj = c - learner_f_under_model + 10 * gp
        obj.backward()
        pseudoreward_optimizer.step()

        #evaluate performance
        mean_reward, std_reward = evaluate_policy(
            model, gym.make(env), n_eval_episodes=10)
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        print("{0} Iteration: {1}".format(outer, mean_reward))
        if save_rewards:
            np.savez(os.path.join("learners", env, "mimicmd_rewards_{0}".format(
                outer)), means=mean_rewards, stds=std_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train expert policies.')
    parser.add_argument(
        '-a', '--algo', choices=['bc', 'gail', 'sqil', 'adril', 'advil', 'wgail', 'mimicmd'], required=True)
    parser.add_argument('-e', '--env', choices=['cartpole', 'lunarlander', 'acrobot', 'pendulum', 'halfcheetah', 'walker', 'hopper', 'ant'],
                        required=True)
    parser.add_argument('-n', '--num_runs', required=False)
    args = parser.parse_args()
    if args.env == "cartpole":
        envname = 'CartPole-v1'
    elif args.env == "lunarlander":
        envname = 'LunarLander-v2'
    elif args.env == "acrobot":
        envname = 'Acrobot-v1'
    elif args.env == "pendulum":
        envname = 'Pendulum-v0'
    elif args.env == "halfcheetah":
        envname = 'HalfCheetahBulletEnv-v0'
    elif args.env == "walker":
        envname = 'Walker2DBulletEnv-v0'
    elif args.env == "hopper":
        envname = 'HopperBulletEnv-v0'
    elif args.env == "ant":
        envname = 'AntBulletEnv-v0'
    else:
        print("ERROR: unsupported env.")
    if args.num_runs is not None and args.num_runs.isdigit():
        num_runs = int(args.num_runs)
    else:
        num_runs = 1
    if args.algo == 'bc':
        train_bc(envname, num_runs)
    elif args.algo == 'gail':
        train_gail(envname, num_runs)
    elif args.algo == 'wgail':
        train_wgail(envname, num_runs)
    elif args.algo == 'mimicmd':
        train_mimicmd(envname, num_runs)
    elif args.algo == 'sqil':
        train_sqil(envname, num_runs)
    elif args.algo == 'adril':
        train_adril(envname, num_runs)
    elif args.algo == 'advil':
        train_advil(envname, num_runs)
    else:
        print("ERROR: unsupported algorithm")
