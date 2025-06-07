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
            pair_data = collect_pair_data(self.policy,seed,1000)
            dataset = PairedTrajectoryDataset(pair_data)
            dataloader = DataLoader(
                dataset,
                batch_size=32, 
                shuffle=True,
                collate_fn=PairedTrajectoryDataset.collate_fn,
            )
            for epoch in tqdm(range(num_epochs_per_iter)):
                for batch in dataloader:
                    states_1 = batch["traj1_state"]      # shape: (B, T, obs_dim)
                    actions_1 = batch["traj1_act"]       # shape: (B, T, act_dim)
                    logps_1 = batch["traj1_logp"]        # shape: (B, T)

                    states_2 = batch["traj2_state"]
                    actions_2 = batch["traj2_act"]
                    logps_2 = batch["traj2_logp"]

                    labels = batch["label"]              # shape: (B,)

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
                torch.save(self.policy, "dpo_iterative.pt")


def main():
    env = gym.make("Swimmer-v5")
    policy = ContinuousPolicy(env.observation_space.shape[0], env.action_space.shape[0])
    # load model
    policy.load_state_dict(torch.load("swimmer_checkpoint.pt", weights_only=False))
    pair_data = torch.load("pair_data.pt", weights_only=False)
    pair_data = pair_data[0:1000]
    # argparse
    # load hparams
    with open("hparam.yaml", "r") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(hparams["lr"]))
    dpo = DPOTrainer(env, policy, optimizer, float(hparams["beta"]), int(hparams["batch_size"]))

    if hparams["iterative_dpo"]:
        iterations = 10
    else:
        iterations = 1

    dpo.train(pair_data, num_iterations=iterations, seed=42, num_epochs_per_iter=hparams["num_epochs_per_iter"])


if __name__ == "__main__":
    main()
