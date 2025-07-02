import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from batch_collection import batch_collection
from utils import validate_model
from model import ContinuousPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_gae_returns(rewards, values, dones, gamma, gae_lambda):
    """
    Returns the advantages computed via GAE and the discounted returns. 

    Instead of using the Monte Carlo estimates for the returns,
    use the computed advantages and the value function
    to compute an estimate for the returns. 
    
    Hint: How can you easily do this if lambda = 1?

    :param rewards: The reward at each state-action pair
    :param values: The value estimate at the state
    :param dones: Whether the state is terminal/truncated
    :param gamma: Discount factor
    :param gae_lambda: lambda coef for GAE
    """       
    N = rewards.shape[0]
    M = rewards.shape[1]
    A = torch.zeros([N+1,M])
    #A2 = torch.zeros([N+1,M])
    for t in reversed(range(N)):
        rew = rewards[t]
        val = values[t]
        done = dones[t]
        delta_t = rew + gamma * values[t+1]*(1-done) - val
        A[t] = delta_t + gamma*gae_lambda*(1-done)*A[t+1]
        #A2[t] = delta_t + gamma*(1-done)*A2[t+1]
    returns = A[0:N] + values[0:N]
    return A[0:N],returns

def ppo_loss(agent, states, actions, advantages, logprobs, returns, clip_ratio=0.2, ent_coef=0.01, vf_coef=0.5) -> torch.Tensor:
    """
    Compute the PPO loss. You can combine the policy, value and entropy losses into a single value. 

    :param policy: The policy network
    :param states: States batch
    :param actions: Actions batch
    :param advantages: Advantages batch
    :param logprobs: Log probability of actions
    :param returns: Returns at each state-action pair
    :param clip_ratio: Clipping term for PG loss
    :param ent_coef: Entropy coef for entropy loss
    :param vf_coef: Value coef for value loss
    """  
    
    #policy loss
    _, new_logprobs, entropys,values = agent.action_value(states,actions)
    rt = torch.exp(new_logprobs-logprobs)
    rtAt = rt*advantages
    clip = torch.clamp(rt,min=1-clip_ratio, max = 1+clip_ratio)*advantages
    min_t = torch.minimum(rtAt,clip)
    L_clip = -torch.mean(min_t)
    lossV = nn.MSELoss()
    L_v = lossV(values,returns)
    L_s = torch.mean(entropys)
    return L_clip + vf_coef*L_v - ent_coef*L_s, L_clip,L_v,L_s


def train(
    env,
    eval_env,
    policy,
    optimizer,
    epochs=500,
    num_envs=4,
    gamma=0.99,
    gae_lambda=0.9,
    lr=3e-4,
    num_steps=128,
    minibatch_size=32,
    clip_ratio=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    update_epochs=3,
    seed=42,
    checkpoint=False,
    max_grad_norm=0.5,
):
    states = torch.zeros((num_steps, num_envs) + env.single_observation_space.shape)
    actions = torch.zeros((num_steps, num_envs) + env.single_action_space.shape)
    logprobs = torch.zeros((num_steps, num_envs))
    rewards = torch.zeros((num_steps, num_envs))
    dones = torch.zeros((num_steps, num_envs))
    values = torch.zeros((num_steps + 1, num_envs))
    max_eval_reward = 0
    for iteration in tqdm(range(1, epochs + 1)):
        policy = policy.to("cpu")
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float()
        for j in range(num_steps):
            with torch.no_grad():
                action,log_prob,_,value= policy.action_value(obs)
            #print(states.shape,actions.shape,obs.shape,action.shape,log_prob.shape)
            n_obs,rew,terminated,_,_ = env.step(action.numpy())
            states[j] = obs
            actions[j] = action
            logprobs[j] = log_prob
            rewards[j] = torch.from_numpy(rew).float()
            dones[j] = torch.from_numpy(terminated).float()
            values[j] = value
            obs = torch.from_numpy(n_obs).float()
        with torch.no_grad():
            values[j+1] = policy.value(torch.from_numpy(n_obs).float())
        advantages, returns = compute_gae_returns(rewards,values,dones,gamma,gae_lambda)
        # 创建 dataset
        dataset = TensorDataset(states, actions, logprobs, returns, advantages)

        # 创建 dataloader，shuffle 自动打乱，batch_size 控制 minibatch 大小
        dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)
        policy = policy.to(device)
        for _ in range(update_epochs):
            for batch in dataloader:
                states_batch, actions_batch, logprobs_batch, returns_batch, advantages_batch = [x.to(device) for x in batch]

                # 计算 PPO loss
                loss, L_c, L_v, L_s = ppo_loss(
                    policy, states_batch, actions_batch,
                    advantages_batch, logprobs_batch, returns_batch,
                    clip_ratio, ent_coef, vf_coef
                )

                # 优化器步骤
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Clip the gradient
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        # Uncomment for eval/checkpoint
        if iteration % 25 == 0:
            policy = policy.to("cpu")
            mean_rew,_ = validate_model(policy,eval_env,40)
            print(f"Eval Reward {iteration}:", mean_rew)
            if checkpoint and mean_rew > max_eval_reward:
                max_eval_reward = mean_rew
                torch.save(policy, f"learned_policies/model_{iteration}.pt")

def make_env(env_id, **kwargs):
    def env_fn():
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        return env
    return env_fn

def main():
    env_id = "Swimmer-v5"
    num_envs = 4
    env = gym.vector.SyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
    eval_env = make_env(env_id)()
    policy = ContinuousPolicy(env.single_observation_space.shape[0], env.single_action_space.shape[0]).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    train(env,eval_env,policy,optimizer,epochs=400,num_envs=num_envs,gamma=0.99,num_steps=512,gae_lambda=0.95,minibatch_size=64,ent_coef=0.03,update_epochs=6)

if __name__ == "__main__":
    main()