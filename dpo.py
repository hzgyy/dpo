import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import validate_model
from model import ContinuousPolicy
import gymnasium as gym
from tqdm import tqdm
from batch_collection import collect_pair_data
import argparse
import yaml
from imitation import ImitationTrainer,il_reorganize_data
from ppo import train,make_env
from scipy.stats import kstest, pearsonr
class PairedTrajectoryDataset(Dataset):
    def __init__(self, pair_data):
        self.data = pair_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p = self.data[idx]
        t1, t2 = p["traj1"], p["traj2"]
        return {
            "traj1_state": torch.from_numpy(t1[0]).float(),  # (T, obs_dim)
            "traj1_act": torch.from_numpy(t1[2]).float(),  # (T, act_dim)
            "traj1_logp": torch.from_numpy(t1[3]).float(),
            "traj2_state": torch.from_numpy(t2[0]).float(),
            "traj2_act": torch.from_numpy(t2[2]).float(),
            "traj2_logp": torch.from_numpy(t2[3]).float(),
            "label": torch.tensor(p["label"], dtype=torch.float32),
        }

    def collate_fn(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}


class DPOTrainer:
    def __init__(self, env, policy, optimizer, beta=1, batch_size=16, device="cpu"):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.beta = beta
        self.batch_size = batch_size
        self.device = device

    def _evaluate(self, seed, n_trajs=40):
        mean_rew, std_rew = validate_model(self.policy, self.env, n_trajs)
        return mean_rew, std_rew

    def train(self, pair_data, num_epochs_per_iter=6, num_iterations=10, seed=None):
        criterion = torch.nn.MSELoss()
        max_reward = 0

        for iteration in range(num_iterations):
            pair_data = collect_pair_data(self.policy,seed,1024)
            dataset = PairedTrajectoryDataset(pair_data)
            dataloader = DataLoader(
                dataset,
                batch_size=32, 
                shuffle=True,
                collate_fn=PairedTrajectoryDataset.collate_fn,
            )
            self.policy = self.policy.to(self.device)
            for epoch in tqdm(range(num_epochs_per_iter)):
                for batch in dataloader:
                    states_1 = batch["traj1_state"].to(self.device)      # shape: (B, T, obs_dim)
                    actions_1 = batch["traj1_act"].to(self.device)       # shape: (B, T, act_dim)
                    logps_1 = batch["traj1_logp"].to(self.device)        # shape: (B, T)

                    states_2 = batch["traj2_state"].to(self.device)
                    actions_2 = batch["traj2_act"].to(self.device)
                    logps_2 = batch["traj2_logp"].to(self.device)

                    labels = batch["label"].to(self.device)              # shape: (B,)
                    
                    logps_traj_1 = self.policy.compute_log_likelihood(states_1, actions_1)  # shape: (B, T)
                    logps_traj_2 = self.policy.compute_log_likelihood(states_2, actions_2)

                    # 注意 sum 是对 T 维度求和 (dim=1)，保留 batch 维度
                    y = self.beta * ((logps_traj_1.sum(dim=1) - logps_1.sum(dim=1)) -
                                    (logps_traj_2.sum(dim=1) - logps_2.sum(dim=1)))  # shape: (B,)
                    loss = criterion(y, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            mean_rew, std_rew = self._evaluate(seed)
            print(f"iteration {iteration}: mean reward: {mean_rew:.2f}, std: {std_rew:.2f}")
            if mean_rew > max_reward:
                max_reward = mean_rew
                torch.save(self.policy.state_dict(), "dpo_iterative.pt")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Swimmer-v5")
    num_envs = 4
    train_env = gym.vector.SyncVectorEnv([make_env("Swimmer-v5") for _ in range(num_envs)])
    policy = ContinuousPolicy(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    # load model
    #policy.load_state_dict(torch.load("il_policy.pt", weights_only=False,map_location=device))
    
    # pair_data = torch.load("pair_data.pt", weights_only=False)
    # pair_data = pair_data[0:1000]
    expert_data = torch.load("trajs.pt", weights_only=False)
    train_expert_data,validate_expert_data = il_reorganize_data(expert_data,True)
    train_expert_data_noshuffle,validate_expert_data_noshuffle = il_reorganize_data(expert_data,False)
    shuffle_data = [d[0] for d in train_expert_data]
    no_shuffle_data = [d[0] for d in train_expert_data_noshuffle]
    kst_iid = kstest(shuffle_data, 'norm')
    kst_non_iid = kstest(no_shuffle_data, 'norm')
    autocorr_iid = pearsonr(shuffle_data[:-1], shuffle_data[1:])[0]
    autocorr_non_iid = pearsonr(no_shuffle_data[:-1], no_shuffle_data[1:])[0]
    print("IID Data KS Test:", kst_iid.item().mean())
    print("Non-IID Data KS Test:", kst_non_iid.mean())
    print("IID Data Autocorrelation:", autocorr_iid.mean())
    print("Non-IID Data Autocorrelation:", autocorr_non_iid.mean())
    exit()
    # argparse
    # load hparams
    with open("hparam.yaml", "r") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(hparams["lr"]))
    dpo = DPOTrainer(env, policy, optimizer, float(hparams["beta"]), int(hparams["batch_size"]),device)
    il = ImitationTrainer(env,train_expert_data,validate_expert_data,policy,optimizer,20,"il_policy.pt")

    if hparams["iterative_dpo"]:
        iterations = 10
    else:
        iterations = 1

    il.learn()
    #dpo.train([], num_iterations=iterations, seed=42, num_epochs_per_iter=hparams["num_epochs_per_iter"])
    # for i in range(30):
    #     train(train_env,env,policy,optimizer,epochs=25,num_envs=num_envs,gae_lambda=0.95,num_steps=512,minibatch_size=64,ent_coef=0.03,update_epochs=6)
    #     il.learn_step()
    #     mean_rew,_ = validate_model(policy,env,40)
    #     print(f"iteration{i} reward:{mean_rew}")


if __name__ == "__main__":
    main()
