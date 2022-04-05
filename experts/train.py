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
  parser.add_argument('env', choices=['halfcheetah', 'walker',
                                      'hopper',])
  args = parser.parse_args()

  if args.env == 'halfcheetah':
    train_halfcheetah_expert()
  elif args.env == 'walker':
    train_walker_expert()
  elif args.env == 'hopper':
    train_hopper_expert()
  else:
    print("ERROR: unsupported env.")
