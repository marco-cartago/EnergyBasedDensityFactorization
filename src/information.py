import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


class EntropyNetwork(nn.Module):
    """MI NEE implementation"""

    def __init__(self, input_dim, hidden_dim, out_dim):
        super(EntropyNetwork, self).__init__()

        self.activation = nn.GELU

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class TotalCorrelationEstimator:

    def __init__(
        self,
        d,
        data_shape,
        hidden_dim=16,
        lr=1e-3,
        max_t=15.0,
        buffer_size=1000,
        batch_size=32,
        device=torch.device("cpu"),
    ):

        # Network and optimizer
        self.marginal_networks = nn.ModuleList(
            [EntropyNetwork(1, hidden_dim, 1).to(device) for _ in range(d)]
        )

        self.max_t = max_t
        self.device = device
        self.d = d
        self.data_shape = data_shape
        self.joint_network = EntropyNetwork(d, hidden_dim, 1).to(device)
        self.parameters = list(self.marginal_networks.parameters()) + list(
            self.joint_network.parameters()
        )
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        self.lr = lr

        # Buffer steps
        self.buffer = torch.rand(size=(buffer_size, d), device=self.device)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def update_buffer(
        self,
        samples: torch.Tensor,
    ):
        """
        Store new entropy samples. It randomly replace the old samples.
        """
        samples = samples.detach().cpu()
        n_sam = samples.size(0)

        idxs = torch.randperm(self.buffer_size)[:n_sam]
        self.buffer[idxs] = samples.to(self.device)

    def marginal_entropy(self, x, network):
        t = network(x)
        t = torch.clamp(t, max=self.max_t)
        et = torch.exp(t)
        return torch.mean(t) - torch.log(torch.mean(et))

    def joint_entropy(self, x):
        t = self.joint_network(x)
        t = torch.clamp(t, max=self.max_t)
        et = torch.exp(t)
        return torch.mean(t) - torch.log(torch.mean(et) + 1e-8)

    def total_correlation(self, x):
        x = x.to(self.device)
        sum_marginal_h = 0.0
        for i in range(self.d):
            xi = x[:, i].unsqueeze(-1)
            h = self.marginal_entropy(xi, self.marginal_networks[i])
            sum_marginal_h = sum_marginal_h + h

        h_joint = self.joint_entropy(x)
        tc = sum_marginal_h - h_joint
        return tc

    def train_step(self, x):
        x = x.to(self.device)
        self.optimizer.zero_grad()
        tc = -self.total_correlation(x)
        tc.backward()
        self.optimizer.step()
        return tc.item()

    def train_steps(self, n_epochs: int | None = 50, tol: float | None = 1e-4):
        data = TensorDataset(self.buffer)
        dbuf = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        if n_epochs is not None:
            for _ in range(n_epochs):
                for btc in dbuf:
                    b = btc[0]
                    b.to(self.device)
                    curr_loss = self.train_step(b)
            return

        if tol is not None:
            prev_loss = float("inf")
            curr_loss = float("-inf")
            while abs(prev_loss - curr_loss) < tol:
                for btc in dbuf:
                    b = btc[0]
                    b.to(self.device)
                    curr_loss = self.train_step(b)
            return

    def to(self, device):
        self.joint_network.to(device)
        self.marginal_networks.to(device)
