import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import itertools
from utils import device
from model import ContinuousPolicy
import math


def batch_collection(env, policy, seed, *, total_trajectories=16, smoothing=False):
    """Collect `total_trajectories` episodes from a SyncVectorEnv."""
    trajectories = []
    num_envs = env.num_envs
    obs, infos = env.reset(seed=seed)
    current_trajectories = [[] for _ in range(num_envs)]

    while len(trajectories) < total_trajectories:
        with torch.no_grad():
            actions = policy.sample_action(obs,smooth=smoothing).detach().numpy()
            logps = policy.compute_log_likelihood(obs,actions).detach().numpy()
        next_obs, rewards, terminates, truncates, infos = env.step(actions)
        
        for i in range(env.num_envs):
            current_trajectories[i].append([obs[i], rewards[i],actions[i],logps[i]])
            if truncates[i]:
                obs_seq, rew_seq,act_seq,logp_seq= zip(*current_trajectories[i])
                obs_seq = np.array(obs_seq)
                rew_seq = np.array(rew_seq)
                act_seq = np.array(act_seq)
                logp_seq = np.array(logp_seq)
                traj = (obs_seq, rew_seq,act_seq,logp_seq)
                trajectories.append(traj)
                current_trajectories[i] = []
                #print(len(trajectories))
        if truncates[0]:
            obs,_ = env.reset()
        else:
            obs = next_obs
    return trajectories

def pair_trajectories(trajs, temp=1.0, seed=None):
    rng = np.random.default_rng(seed)
    returns = np.array([np.sum(t[1]) for t in trajs])

    idx_pairs = list(zip(rng.permutation(len(trajs)), rng.permutation(len(trajs))))
    idx_pairs = [pair for pair in idx_pairs if pair[0] != pair[1]]

    pair_data = []
    #TODO: implement pair_trajectories
    # probs = []
    # for pair in idx_pairs:
    #     delta_rew = returns[pair[0]]-returns[pair[1]]
    #     probs.append(1/(1+math.exp(-delta_rew)))
    delta_rew = np.array([returns[pair[0]]-returns[pair[1]] for pair in idx_pairs])
    probs = 1/(1+np.exp(-delta_rew))
    labels = np.where(np.random.rand(*probs.shape) < probs, 1, -1)
    pair_data = [{"traj1":trajs[idx_pairs[i][0]],"traj2":trajs[idx_pairs[i][1]],"label":labels[i]} for i in range(len(labels))]
    return pair_data

def collect_pair_data(policy, seed, total_trajectories=16, smoothing=False):
    env_fns = [lambda: gym.make("Swimmer-v5") for _ in range(16)]
    env = SyncVectorEnv(env_fns)

    example_env = gym.make("Swimmer-v5")
    obs_dim = example_env.observation_space.shape[0]
    act_dim = example_env.action_space.shape[0]
    example_env.close()

    trajectories = batch_collection(env, policy, seed, total_trajectories=total_trajectories, smoothing=smoothing)
    return pair_trajectories(trajectories, seed=seed)

def main():
    num_envs = 32 # set this based on your hardware
    seed = 42

    env_fns = [lambda: gym.make("Swimmer-v5") for _ in range(num_envs)]
    env = SyncVectorEnv(env_fns)

    example_env = gym.make("Swimmer-v5")
    obs_dim = example_env.observation_space.shape[0]
    act_dim = example_env.action_space.shape[0]
    example_env.close()

    policy = ContinuousPolicy(obs_dim, act_dim)
    policy.load_state_dict(torch.load("swimmer_checkpoint.pt", weights_only=False))

    trajectories = batch_collection(env, policy, seed, total_trajectories=1000)
    #trajectories = torch.load("trajs.pt", weights_only=False)
    mean_reward = np.mean([np.sum(traj[1]) for traj in trajectories])
    std_reward = np.std([np.sum(traj[1]) for traj in trajectories])
    print(f"Mean reward of trajectories: {mean_reward}, std: {std_reward}")
    torch.save(trajectories,"trajs.pt")
    pair_data = pair_trajectories(trajectories, seed=seed)
    torch.save(pair_data, "pair_data.pt")
    # pair_data = torch.load("pair_data.pt", weights_only=False)
    # print(type(pair_data),len(pair_data))

if __name__ == "__main__":
    main()
