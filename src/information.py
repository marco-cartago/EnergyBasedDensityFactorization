import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyNetwork(nn.Module):

    """MI NEE implementation"""

    def __init__(self, input_dim, hidden_dim=128):
        super(EntropyNetwork, self).__init__()

        self.activation = nn.GELU()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class TotalCorrelationEstimator:

    def __init__(self, d, hidden_dim=128, lr=1e-2, max_t=15.0):
        self.d = d
        self.marginal_networks = nn.ModuleList(
            [EntropyNetwork(1, hidden_dim) for _ in range(d)]
        )
        self.max_t = max_t
        self.lr = lr
        self.joint_network = EntropyNetwork(d, hidden_dim)
        self.parameters =  list(self.marginal_networks.parameters()) + list(self.joint_network.parameters())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr)

        self.buffer = None

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
        sum_marginal_h = 0.0
        for i in range(self.d):
            xi = x[:, i].unsqueeze(1)
            h = self.marginal_entropy(xi, self.marginal_networks[i])
            sum_marginal_h = sum_marginal_h + h

        h_joint = self.joint_entropy(x)
        tc = sum_marginal_h - h_joint
        return tc

    def train_step(self, x):
        self.optimizer.zero_grad()
        tc = -self.total_correlation(x)
        tc.backward()
        self.optimizer.step()
        return tc.item()

    def train_steps(self, x, N):
        for _ in range(N):
            self.train_step(x)

    def to(self, device):
        self.joint_network.to(device)
        self.marginal_networks.to(device)
