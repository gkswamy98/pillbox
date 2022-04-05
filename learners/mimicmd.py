from imitation.algorithms import adversarial, bc
from imitation.util import logger, util
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common import policies
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.rewards import discrim_nets
import numpy as np
import argparse
from utils import make_sa_dataloader, make_sads_dataloader, make_sa_dataset, linear_schedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
from gym.spaces import Discrete
import gym
import pybullet_envs
from stable_baselines3.common.running_mean_std import RunningMeanStd
from torch.nn import functional as F
import torch
from torch import nn
from rlkit.policies.base import Policy
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import LayerNorm
from rlkit.torch.pytorch_util import activation_from_string
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer
from typing import List, Type

def create_mlp(
    input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU) -> List[nn.Module]:

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    return modules

def init_ortho(layer):
    if type(layer) == nn.Linear:
        nn.init.orthogonal_(layer.weight)

#This file should serve as storage for functions useful to MIMIC-MD algorithm
class MIMICMD():
    def __init__(self, base_env, vec_e, beta, num_vec=8, discrete_env=False):
        self.base_env = base_env
        self.query_policy = None
        self.member_oracle = None
        self.d3_trajs = []
        self.d2_prime_trajs = []
        self.mem_threshold = None
        self.action_threshold = None
        self.f_function = None
        self.replay_buffer = None
        self.discrete_env = discrete_env
        self.expert_states = None
        self.horizon = None
        self.avg_mem_dist = None
        self.beta = beta
        self.num_vec = num_vec
        self.first_alph_comp = True
        self.first_alpha_term = None
        self.second_alpha_term = None
        self.expert_pol = None

        #only for the cost function
        self.d3_obs = []
        self.d3_actions = []
        self.d2_prime_obs = []
        self.d2_prime_actions = []
        self.total_d3 = 0
        self.total_d3_traj = 0
        self.total_d3_states_included = 0
        self.vec_e = vec_e
        self.eval_env = None
        self.bc_nets = []
        self.rnd_net = None
        self.first_exp_term = None
        self.first_alph = None
        self.sec_alph = None

    def train_query_policy(self, venv, w, env, d1_size, bc_steps, load=False, distance="ensemble", num_trajs=None):
        #This function runs step 2 and trains BC on D_1 to learn a query policy
        mean_rewards = []
        std_rewards = []

        expert_data_d1 = make_sa_dataloader(env, max_trajs=d1_size, normalize=False)

        bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=expert_data_d1,
                           policy_class=policies.ActorCriticPolicy,
                           ent_weight=0., l2_weight=0., policy_kwargs=dict(net_arch=[w, w]))
        if not load:
            bc_trainer.train(n_batches=int(5e5), log_interval=5000)

        def get_policy(*args, **kwargs):
            return bc_trainer.policy
        model = PPO(get_policy, env, verbose=0)
        env_to_eval = model.get_env()
        self.eval_env = env_to_eval

        if not load:
            model.save(os.path.join("learners", env, "bc_query_1"))
        else:
            model = PPO.load(os.path.join("learners", env, "bc_query_1"))
            self.expert_pol = model
            print("loaded model for expert")

            mean_reward, std_reward = evaluate_policy(
                    model, env_to_eval, n_eval_episodes=10)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            print("BC query policy on D_1 Trajs: {0}".format(mean_reward))

        #save the learned query_policy
        self.query_policy = model

        if distance=="expert":
            #expert names are hard coded, so need to modify this
            self.expert_pol = SAC.load(os.path.join("experts", env, "hopper_expert"))

        #For now, assume models to already be saved
        elif distance=="ensemble" or distance=="maxdist":
            #we only use 5 networks for ensemble variance, including the original query policy
            for i in range(1, 5):
                if not load:
                    mean_rewards = []
                    std_rewards = []

                    expert_data = make_sa_dataloader(env, max_trajs=num_trajs, normalize=False)
                    bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=expert_data,
                                       policy_class=policies.ActorCriticPolicy,
                                       ent_weight=0., l2_weight=0., policy_kwargs=dict(net_arch=[w, w]))

                    bc_trainer.train(n_batches=int(5e5))

                    def get_policy(*args, **kwargs):
                        return bc_trainer.policy
                    model = PPO(get_policy, env, verbose=1)
                    model.save(os.path.join("learners", env,
                                            "bc_query_{0}".format(i+1)))
                    mean_reward, std_reward = evaluate_policy(
                            model, model.get_env(), n_eval_episodes=10)
                    mean_rewards.append(mean_reward)
                    std_rewards.append(std_reward)
                    print("BC Net Reward: {0}".format(mean_rewards))

                model = PPO.load(os.path.join("learners", env, "bc_query_{0}").format(i))
                print("loaded model")
                # mean_reward, std_reward = evaluate_policy(
                #         model, env_to_eval, n_eval_episodes=10)
                # mean_rewards.append(mean_reward)
                # std_rewards.append(std_reward)
                print("Ensemble query policy: {0}".format(mean_reward))
                self.bc_nets.append(model)

        elif distance=="rnd":
            if not load:
                #first generate data, stick with default 25 trajs train
                data_rnd = make_sa_dataloader(env, max_trajs=num_trajs, normalize=False, raw_traj=True)
                self.generate_d3(venv, target_samps=None, load=False, trajs=data_rnd, create_rnd_data=True, num_traj=len(num_trajs))

                mean_rewards = []
                std_rewards = []

                expert_data = make_sa_dataloader(env, max_trajs=list(range(len(num_trajs))), normalize=False, rnd=True)
                bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=expert_data,
                                   policy_class=policies.ActorCriticPolicy,
                                   ent_weight=0., l2_weight=0., policy_kwargs=dict(net_arch=[w, w]))

                bc_trainer.train(n_batches=int(5e5))

                def get_policy(*args, **kwargs):
                    return bc_trainer.policy
                model = PPO(get_policy, env, verbose=1)
                model.save(os.path.join("learners", env,
                                        "rnd_bc_trajs"))
                mean_reward, std_reward = evaluate_policy(
                        model, model.get_env(), n_eval_episodes=10)
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
                print("RND net reward: {0}".format(mean_reward))
                self.rnd_net = model
                model.save(os.path.join("learners", env, "rnd_trained_1"))
            else:
                model = PPO.load(os.path.join("learners", env, "rnd_trained_1"))
                self.rnd_net = model
        return

    #can manually set or automatically set the average used to compute sigmoid values
    def fetch_avg_dist(self, data, distance_metric="ensemble", expert_acts=None):
        if distance_metric=="ensemble":
            #self.bc_nets stores all bc_nets
            vals = []
            for mem in self.bc_nets:
                actions, _ = mem.predict(data)
                vals.append(actions)

            #sum instead of norm
            distance = np.sum(np.var(np.array(vals), axis=0), axis=1)

            #variance across each dimension
            var_dist = np.mean(distance)
            self.avg_mem_dist = 0.005
            print("avg mem dist", self.avg_mem_dist)

        elif distance_metric=="maxdist":
            first = True
            for idx_1, model_1 in enumerate(self.bc_nets):
                pred_1, _ = model_1.predict(data)
                for idx_2, model_2 in enumerate(self.bc_nets):
                    if idx_2 > idx_1:
                        pred_2, _ = model_2.predict(data)
                        disagreement = np.linalg.norm((pred_1-pred_2), axis=-1)
                        if first:
                            delta = disagreement
                            first = False
                        else:
                            delta = np.maximum(delta, disagreement)
            delta = max(delta)

            #variance across each dimension
            self.avg_mem_dist = 0.1
            print("avg mem dist", self.avg_mem_dist)
        elif distance_metric=="rnd":
            actions_orig, _ = self.query_policy.predict(self.d3_obs)
            actions_rnd, _ = self.rnd_net.predict(self.d3_obs)
            self.avg_mem_dist = 0.21
            print("avg mem dist", self.avg_mem_dist)
        elif distance_metric=="expert":
            self.avg_mem_dist = 0.35

    def generate_d3(self, env, target_samps, load, trajs=None, create_rnd_data=False, num_traj=None):
        if create_rnd_data:
            names = []
            values = []
            for i in range(num_traj):
                traj = trajs[i]
                names.append(str(i))
                t = {}
                traj_obs = traj["states"]
                t["states"] = traj["states"]
                actions = []
                for i in range(len(traj["states"])):
                    a, _ = self.query_policy.predict(traj["states"][i])
                    actions.append(a)
                t["actions"] = actions
                values.append(np.array(t))
            names.append("num_trajs")
            values.append(num_traj)
            np.savez('rnd_data.npz',**{name:value for name,value in zip(names,values)})
            return

        if load:
            data = np.load("d3_samps.npz", allow_pickle=True)
            self.d3_obs = data["obs"]
            self.d3_actions = data["acts"]
            return

        env_to_eval = self.eval_env
        model = self.query_policy

        n_eval_episodes = 10
        n_envs = 1
        episode_rewards = []
        episode_lengths = []
        episode_counts = 0
        episode_count_targets = target_samps
        current_rewards = 0
        current_lengths = 0
        observations = env_to_eval.reset()

        cur_obs, cur_acts = [], []
        while episode_counts < episode_count_targets:
            cur_obs.append(observations[0])
            actions, _ = model.predict(observations, state=None, deterministic=True)

            cur_acts.append(actions[0])
            observations, rewards, dones, infos = env_to_eval.step(actions)
            current_rewards += rewards
            current_lengths += 1
            for i in range(n_envs):
                if episode_counts < episode_count_targets:
                    reward = rewards
                    done = dones
                    info = infos

                    if dones:
                        print(episode_counts)
                        self.d3_obs.extend(cur_obs)
                        self.d3_actions.extend(cur_acts)
                        episode_rewards.append(current_rewards)
                        episode_lengths.append(current_lengths)

                        episode_counts += 1
                        current_rewards = 0
                        current_lengths = 0
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print("generated data reward: {0}".format(mean_reward))
        if not load:
            np.savez("d3_samps.npz", obs = self.d3_obs, acts = self.d3_actions)

    def alpha_func(self, states, distance_metric="ensemble"):
        if distance_metric=="ensemble":
            #self.bc_nets stores all bc_nets
            vals = []
            for mem in self.bc_nets:
                actions, _ = mem.predict(states)
                vals.append(actions)
            distance = np.linalg.norm(np.var(np.array(vals), axis=0), axis=1)

        elif distance_metric=="expert":
            actions_orig, _ = self.query_policy.predict(states)
            actions_exp, _ = self.expert_pol.predict(states)
            # use actual expert policy
            distance = np.linalg.norm(actions_orig - actions_exp, axis=1)

        elif distance_metric=="maxdist":
            vals = []
            for mem in self.bc_nets:
                actions, _ = mem.predict(states)
                vals.append(actions)
            distance = np.linalg.norm(np.ptp(np.array(vals), axis=0), axis=1)

        elif distance_metric=="rnd":
            vals = []
            actions_orig, _ = self.query_policy.predict(states)
            actions_rnd, _ = self.rnd_net.predict(states)
            vals.append(actions_orig)
            vals.append(actions_rnd)
            distance = np.linalg.norm(np.ptp(np.array(vals), axis=0), axis=1)

        #can choose to use prefixxing or not, uncomment following, assumes episode lengths of 1000
        # for j in range(0, len(distance), 1000):
        #     curr = distance[j]
        #     for i in range(1, 1000):
        #         curr+=distance[j+i]
        #         distance[j+i] = curr/(i+1)

        return torch.sigmoid(torch.FloatTensor((self.avg_mem_dist - distance)/self.beta))


    def cost(self, f_function, d2_obs, d2_acts, d1_obs=None, d1_acts=None, distance_metric="ensemble"):
        if not self.f_function:
            self.f_function = f_function

        sa_pairs_d3 = np.concatenate((self.d3_obs, self.d3_actions), axis=1)
        sa_pairs_d2 = np.concatenate((d2_obs, d2_acts), axis=1)

        self.first_exp_term = self.f_function.forward(torch.tensor(sa_pairs_d3, dtype=torch.float)).unsqueeze(dim = 1)
        self.sec_exp_term = self.f_function.forward(torch.tensor(sa_pairs_d2, dtype=torch.float)).unsqueeze(dim = 1)

        if self.first_alph_comp:
            self.first_alpha_term = self.alpha_func(self.d3_obs, distance_metric)
            self.sec_alpha_term = 1-self.alpha_func(d2_obs, distance_metric)

            combined = torch.mean(self.first_alpha_term) + torch.mean(self.sec_alpha_term)

            self.first_alpha_term = (1/combined) * self.first_alpha_term
            self.sec_alpha_term = (1/combined) * self.sec_alpha_term
            print("first term adjusted", torch.mean(self.first_alpha_term))
            print("second term adjusted", torch.mean(self.sec_alpha_term))

            self.first_alph_comp = False

        # second_alpha_term = self.second_alpha_term
        term_one = torch.mul(torch.unsqueeze(self.first_alpha_term, 1), self.first_exp_term)
        term_two = torch.mul(torch.unsqueeze(self.sec_alpha_term, 1), self.sec_exp_term)

        return torch.mean(term_one) + torch.mean(term_two)

    def sample_and_add(
        self,
        env,
        policy,
        trajs,
        max_path_length=np.inf,
    ):
        #rollout trajectories using a policy and add to replay buffer
        observations = []
        actions = []
        path_length = 0
        obs = env.reset()
        total_trajs = 0
        while total_trajs < trajs:
            while path_length < max_path_length:
                observations.append(obs)
                act = policy.predict(obs)[0]
                actions.append(act)
                obs, _, d, _ = env.step(act)

                path_length += 1

                if d:
                    total_trajs+=1
                    o = o = env.reset()
                    break
            if not d:
                total_trajs+=1
        self.replay_buffer.add(observations, actions)

        return

    def set_expert_states(self, expert_trajs):
        expert_states = []
        for traj in expert_trajs:
            states = traj["states"]
            for state in states:
                expert_states.append(state)
        self.expert_states = expert_states

    def set_horizon(self, horizon):
        self.horizon = horizon

#from stable_baselines3.common.buffers import ReplayBuffer
class MIMICMDReplayBuffer():
    def __init__(self, obs_space_size, action_space_size):
        self.obs_size = obs_space_size
        self.act_size = action_space_size
        self.size = 0
        self.obs = None
        self.actions = None
        self.first_addition = True

    def size():
        return self.size

    def add(self, obs, act):
        if not obs or not act:
            return

        if not len(obs[0]) == self.obs_size or  not len(act[0]) == self.act_size:
            raise Exception('incoming samples do not match the correct size')
        if self.first_addition:
            self.first_addition = False
            self.obs = np.array(obs)
            self.actions = np.array(act)
        else:
            self.obs = np.append(self.obs, np.array(obs), axis=0)
            self.actions = np.append(self.actions, np.array(act), axis=0)
        self.size += len(obs)
        return

    def sample(self, batch):
        indexes = np.random.choice(range(self.size), batch)
        return self.obs[indexes], self.actions[indexes]

class MIMICMDPolicy(nn.Module):
    def __init__(self, env, mean=None, std=None):
        super(MIMICMDPolicy, self).__init__()
        if isinstance(env.action_space, Discrete):
            self.net_arch = [64, 64]
            self.action_dim = env.action_space.n
            self.discrete = True
        else:
            self.net_arch = [256, 256]
            self.action_dim = int(np.prod(env.action_space.shape))
            self.low = torch.as_tensor(env.action_space.low)
            self.high = torch.as_tensor(env.action_space.high)
            self.discrete = False
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = env.observation_space
        net = create_mlp(self.obs_dim, self.action_dim, self.net_arch, nn.ReLU)
        if self.discrete:
            net.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*net)
        self.net.apply(init_ortho)
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
            self.is_normalized = True
        else:
            self.is_normalized = False
    def forward(self, obs):
        action = self.net(obs)
        return action
    def predict(self, obs, state, mask, deterministic):
        obs = obs.reshape((-1,) + (self.obs_dim,))
        if self.is_normalized:
            obs = (obs - self.mean) / self.std
        obs = torch.as_tensor(obs)
        with torch.no_grad():
            actions = self.forward(obs)
            if self.discrete:
                actions = actions.argmax(dim=1).reshape(-1)
            else:
                actions = self.low + ((actions + 1.0) / 2.0) * (self.high - self.low)
                actions = torch.max(torch.min(actions, self.high), self.low)
            actions = actions.cpu().numpy()
        return actions, state

class MIMICMDDiscriminator(nn.Module):
    def __init__(self, env):
        super(MIMICMDDiscriminator, self).__init__()
        if isinstance(env.action_space, Discrete):
            self.net_arch = [64, 64]
            self.action_dim = env.action_space.n
        else:
            self.net_arch = [256, 256]
            self.action_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        net = create_mlp(self.obs_dim + self.action_dim, 1, self.net_arch, nn.ReLU)
        self.net = nn.Sequential(*net)
        self.net.apply(init_ortho)

    def forward(self, inputs):
        output = self.net(inputs)
        return output.view(-1)
