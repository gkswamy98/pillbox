import gym
import argparse
import numpy as np
import os

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from utils import linear_schedule, make_vec_env
from torch import nn
import pybullet_envs
from stable_baselines3.common.vec_env import VecNormalize
from gym.spaces import Discrete

# All hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
# 64 hidden units for discrete actions, 256 for cont.

def train_cartpole_expert():
    env = make_vec_env('CartPole-v1', n_envs=8)
    model = PPO('MlpPolicy', env, verbose=1,
                n_steps=32, batch_size=256, gae_lambda=0.8, gamma=0.98,
                n_epochs=20, ent_coef=0.0, learning_rate=linear_schedule(0.001),
                clip_range=linear_schedule(0.2), policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=1e5)
    model.save("experts/CartPole-v1/cartpole_expert")
    gen_expert_demos('CartPole-v1', gym.make('CartPole-v1'), model, 25)


def train_lunarlander_expert():
    env = make_vec_env('LunarLander-v2', n_envs=16)
    # Used default hyperparams as tuned seemed to not work that well.
    model = PPO('MlpPolicy', env, verbose=1,
                policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=2e6)
    model.save("experts/LunarLander-v2/lunarlander_expert")
    gen_expert_demos('LunarLander-v2', gym.make('LunarLander-v2'), model, 25)

    
def train_pendulum_expert():
    env = make_vec_env('Pendulum-v0', n_envs=8)
    model = PPO('MlpPolicy', env, verbose=1,
                n_steps=2048, batch_size=64, gae_lambda=0.95, gamma=0.99,
                n_epochs=10, ent_coef=0.0, learning_rate=3e-4,
                clip_range=0.2, policy_kwargs=dict(net_arch=[256, 256]))
    model.learn(total_timesteps=2e6)
    model.save("experts/Pendulum-v0/pendulum_expert")
    gen_expert_demos('Pendulum-v0', gym.make('Pendulum-v0'), model, 25)


def train_halfcheetah_expert():
    # No env normalization.
    env = make_vec_env('HalfCheetahBulletEnv-v0', n_envs=1)
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=7.3e-4, 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("experts/HalfCheetahBulletEnv-v0/halfcheetah_expert")
    gen_expert_demos('HalfCheetahBulletEnv-v0', gym.make('HalfCheetahBulletEnv-v0'), model, 25)


def train_ant_expert():
    # No env normalization.
    env = make_vec_env('AntBulletEnv-v0', n_envs=1)
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=7.3e-4, 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("experts/AntBulletEnv-v0/ant_expert")
    gen_expert_demos('AntBulletEnv-v0', gym.make('AntBulletEnv-v0'), model, 25)


def train_walker_expert():
    # No env normalization.
    env = make_vec_env('Walker2DBulletEnv-v0', n_envs=1)
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=linear_schedule(7.3e-4), 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("experts/Walker2DBulletEnv-v0/walker_expert")
    gen_expert_demos('Walker2DBulletEnv-v0', gym.make('Walker2DBulletEnv-v0'), model, 25)


def train_hopper_expert():
    # No env normalization.
    env = make_vec_env('HopperBulletEnv-v0', n_envs=1)
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=linear_schedule(7.3e-4), 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("experts/HopperBulletEnv-v0/hopper_expert")
    gen_expert_demos('HopperBulletEnv-v0', gym.make('HopperBulletEnv-v0'), model, 25)


def gen_expert_demos(dirname, env, model, num_trajs):
    trajs = dict()
    rewards = []
    for traj in range(num_trajs):
        total_reward = 0
        obs = env.reset()
        done = False
        states = []
        actions = []
        while not done:
            states.append(obs)
            action, _state = model.predict(obs, deterministic=True)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        trajs[str(traj)] = {'states': np.array(
            states), 'actions': np.array(actions)}
        rewards.append(total_reward)
    print("Avg Reward:", np.mean(rewards))
    np.savez(os.path.join('experts', dirname, 'demos'), env=dirname,
             num_trajs=num_trajs,
             mean_reward=np.mean(rewards),
             std_reward=np.std(rewards),
             **trajs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train expert policies.')
  parser.add_argument('env', choices=['cartpole', 'lunarlander', 'acrobot', 'pendulum', 
                                      'lunarlandercontinuous', 'halfcheetah', 'walker',
                                      'hopper', 'ant', 'all',])
  args = parser.parse_args()
  if args.env == 'cartpole':
    train_cartpole_expert()
  elif args.env == 'lunarlander':
    train_lunarlander_expert()
  elif args.env == 'pendulum':
    train_pendulum_expert()
  elif args.env == 'halfcheetah':
    train_halfcheetah_expert()
  elif args.env == 'ant':
    train_ant_expert()
  elif args.env == 'walker':
    train_walker_expert()
  elif args.env == 'hopper':
    train_hopper_expert()
  elif args.env == 'all':
    train_cartpole_expert()
    train_lunarlander_expert()
    train_pendulum_expert()
    train_halfcheetah_expert()
    train_ant_expert()
    train_walker_expert()
    train_hopper_expert()
  else:
    print("ERROR: unsupported env.")
