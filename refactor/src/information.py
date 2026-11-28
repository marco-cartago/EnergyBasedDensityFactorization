"""
Total-correlation estimator for the factorized EBM heads.

Refactor notes
--------------
The old estimator computed ``mean(t) - log(mean(exp(t)))`` using the *same*
samples in both terms.  That is a Jensen gap (always <= 0), not an entropy, so
the "TC" penalty was either zero or had the wrong sign.  Here we use the
Donsker-Varadhan bound with genuine negatives obtained by independently
permuting each column of the batch (product of marginals).  The implementation
is self-contained: it does not depend on ``src/MINE.py``.
"""

from contextlib import contextmanager

from math import log

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset


class EntropyNetwork(nn.Module):

    """Small critic T(x) used by the DV bound."""

    def __init__(self, input_dim, hidden_dim, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TC_Estimator(object):

    def __init__(self, *args, **kwargs):
        pass

    def dv_bound(self, x):
        pass

    def train_step(self, head_outputs):
        pass


class TCEstimator1(TC_Estimator):

    def __init__(self, d, hidden_dim=64, lr=1e-3, device="cpu"):
        self.d, self.device = d, device
        self.Net = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        self.opt = torch.optim.Adam(self.Net.parameters(), lr=lr)

    @staticmethod
    def product_of_marginals(x):
        n, d = x.shape
        # independent permutation of each column
        idx = torch.stack([torch.randperm(n, device=x.device) for _ in range(d)], dim=1)
        return torch.gather(x, 0, idx)

    def dv_bound(self, x):
        x = x.reshape(x.shape[0], -1).to(self.device)
        x_prod = self.product_of_marginals(x)
        t_joint = self.Net(x)
        t_prod = self.Net(x_prod)
        n = t_prod.shape[0]
        log_mean_exp = torch.logsumexp(t_prod, dim=0) - log(n)  # stable
        return t_joint.mean() - log_mean_exp.squeeze()

    def train_step(self, x):
        self.opt.zero_grad()
        loss = -self.dv_bound(x)
        loss.backward()
        self.opt.step()
        return -loss.item()   # current lower bound on TC (in nats)

    def to(self, device):
        self.Net.to(device)
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(d={self.d}, Net={self.Net})"

    def __call__(self, x):
        return self.Net(x).mean()




def _permute_independently(x: torch.Tensor) -> torch.Tensor:
    """Samples from prod_i p(x_i): shuffle every column independently."""
    b, d = x.shape
    idx = torch.stack(
        [torch.randperm(b, device=x.device) for _ in range(d)], dim=1
    )
    return torch.gather(x, 0, idx)


class TotalCorrelationEstimator(TC_Estimator):
    """Differentiable lower bound on TC(h_1, ..., h_d).

    ``total_correlation`` keeps gradients w.r.t. its input, so it can be used
    as a penalty for the EBM.  The critic's own parameters are frozen while
    computing that penalty (see ``frozen``); call ``train_step`` on detached
    head outputs afterwards to tighten the bound.
    """

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
        self.device = torch.device(device)
        self.d = int(d)
        self.data_shape = tuple(data_shape)
        self.max_t = float(max_t)
        self.lr = float(lr)
        self.batch_size = int(batch_size)

        self.critic = EntropyNetwork(self.d, hidden_dim, 1).to(self.device)
        self.parameters = list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)

        self.buffer_size = int(buffer_size)
        self.buffer = torch.rand(self.buffer_size, self.d, device=self.device)

    # ------------------------------------------------------------------ utils
    def to(self, device):
        self.device = torch.device(device)
        self.critic.to(self.device)
        return self

    def set_requires_grad(self, flag: bool):
        for p in self.parameters:
            p.requires_grad_(bool(flag))

    @contextmanager
    def frozen(self):
        """ critic frozen, gradients still flow to the input tensor."""
        flags = [p.requires_grad for p in self.parameters]
        self.set_requires_grad(False)
        try:
            yield
        finally:
            for p, flag in zip(self.parameters, flags):
                p.requires_grad_(flag)

    def update_buffer(self, samples: torch.Tensor):
        """Randomly replace entries in the replay buffer of head outputs."""
        samples = samples.detach().reshape(samples.shape[0], -1).to(self.device)
        if samples.shape[1] != self.d:
            raise ValueError(
                f"expected {self.d} head output(s), got {samples.shape[1]}"
            )

        n = samples.size(0)
        if n <= 0:
            return
        if n >= self.buffer_size:
            idx = torch.randint(0, self.buffer_size, (n,), device=self.device)
        else:
            idx = torch.randperm(self.buffer_size, device=self.device)[:n]
        self.buffer[idx] = samples

    # ------------------------------------------------------------------- TC
    def dv_bound(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1).to(self.device)
        if x.shape[1] != self.d:
            raise ValueError(f"expected {self.d} head output(s), got {x.shape[1]}")

        x_marginal = _permute_independently(x)
        t_joint = self.critic(x).reshape(-1)
        t_marginal = self.critic(x_marginal).reshape(-1).clamp(max=self.max_t)
        return t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)

    # -------------------------------------------------------------- training
    def train_step(self, x: torch.Tensor) -> float:
        """One gradient step that tightens the lower bound (critic only)."""
        x = x.detach().to(self.device)
        self.set_requires_grad(True)
        self.optimizer.zero_grad(set_to_none=True)
        loss = -self.dv_bound(x)
        loss.backward()
        self.optimizer.step()
        return -loss.item()  # positive when the bound is positive

    def train_steps(
        self,
        n_epochs: int | None = None,
        tol: float | None = 1e-4,
        max_iters: int = 1000,
    ):
        """Optional batched training over the replay buffer.

        With both arguments None this is a no-op (the previous implementation
        never entered its tolerance loop because it started from inf/-inf).
        """
        dbuf = DataLoader(
            TensorDataset(self.buffer), batch_size=self.batch_size, shuffle=True
        )

        if n_epochs is not None:
            for _ in range(n_epochs):
                for (b,) in dbuf:
                    self.train_step(b)
            return

        if tol is not None:
            prev = None
            for _ in range(max_iters):
                losses = [self.train_step(b) for (b,) in dbuf]
                curr = float(sum(losses) / max(len(losses), 1))
                if prev is not None and abs(prev - curr) < tol:
                    break
                prev = curr
