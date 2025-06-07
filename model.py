import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist


class ContinuousPolicy(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        self.x_dim = obs_dim
        self.a_dim = act_dim

        self.linear_1 = nn.Linear(self.x_dim, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.linear_3 = nn.Linear(128, self.a_dim)

        self.log_stds = nn.Parameter(torch.full((self.a_dim,), -0.5))

    def _mean_std(self, x):
        h = F.leaky_relu(self.linear_1(x))
        h = F.leaky_relu(self.linear_2(h))
        means = self.linear_3(h)
        stds = torch.exp(self.log_stds).expand_as(means)
        return means, stds

    def _gaussian_logp(self, means, stds, acts):
        var = stds ** 2
        logp = -0.5 * (((acts - means) ** 2) / var + 2 * torch.log(stds) + torch.log(torch.tensor(2 * torch.pi)))
        return logp.sum(dim=-1)

    def forward(self, x, *, with_logp=False, smooth=False):
        means, stds = self._mean_std(x)
        acts = torch.normal(means, stds)

        if smooth:  # optional exploration noise
            acts += torch.normal(torch.zeros_like(acts), torch.ones_like(acts) * 2.0)

        if with_logp:
            logp = self._gaussian_logp(means, stds, acts)
            return acts, logp
        return acts

    def sample_action(self, x, *, smooth=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.forward(x,smooth=smooth)

    def compute_log_likelihood(self, x, acts):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(acts, torch.Tensor):
            acts = torch.tensor(acts, dtype=torch.float32)
        means,stds = self._mean_std(x)
        return self._gaussian_logp(means,stds,acts)
