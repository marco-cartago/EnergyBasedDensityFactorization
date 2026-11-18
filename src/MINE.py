import torch
import torch.nn as nn


def _permute_independently(x: torch.Tensor) -> torch.Tensor:
    """
    Shuffle each column of x independently across the batch.
    Produces (approximate) samples from prod_i p(x_i), i.e. genuine
    negatives for the Donsker-Varadhan bound.
    x: (batch, d)
    """
    batch_size, d = x.shape
    idx = torch.stack(
        [torch.randperm(batch_size, device=x.device) for _ in range(d)], dim=1
    )  # (batch, d), a different permutation per column
    return torch.gather(x, 0, idx)


class MINE_TC_Estimator:
    """
    MINE-based estimator of the Total Correlation of a d-dim vector x:
        TC(x) = KL( p(x_1,...,x_d) || prod_i p(x_i) )
    via the Donsker-Varadhan bound with a single statistics network T:
        TC(x) >= E_{x~p(x)}[T(x)] - log E_{x'~prod_i p(x_i)}[exp(T(x'))]
    Negatives x' are obtained by independently permuting each column of
    the batch (see _permute_independently). Unlike the old marginal/joint
    split, here there is only ONE network, and joint/marginal samples are
    genuinely different distributions.
    """

    def __init__(self, d: int, hidden_dim=128, lr=1e-4, max_t=15.0):
        self.d = d
        self.max_t = max_t
        self.net = EntropyNetwork(d, hidden_dim)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)

    def to(self, device):
        self.net.to(device)
        return self

    def _set_requires_grad(self, flag: bool):
        for p in self.net.parameters():
            p.requires_grad_(flag)

    def estimate(self, x: torch.Tensor):
        """
        DV lower bound on TC(x). Differentiable w.r.t. both self.net's
        parameters and x, so it's safe to use as an upstream regularizer.
        """
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] != self.d:
            raise ValueError(f"expected {self.d} head output(s), got {x.shape[1]}")

        x_marginal = _permute_independently(x)
        t_joint = self.net(x)
        t_marginal = torch.clamp(self.net(x_marginal), max=self.max_t)
        tc = torch.mean(t_joint) - \
            torch.log(torch.mean(torch.exp(t_marginal)) + 1e-8)
        return tc

    def train_step(self, x: torch.Tensor):
        """One gradient step tightening the bound (updates only self.net)."""
        self.optimizer.zero_grad()
        loss = -self.estimate(x)
        loss.backward()
        self.optimizer.step()
        return -loss.item()  # positive TC estimate, for logging


def mine_tc_regularization(head_outputs: torch.Tensor, tc_estimator: MINE_TC_Estimator):
    """
    TC regularization term for the model's loss. Gradients flow into
    head_outputs (hence into the model), NOT into tc_estimator's own
    parameters, which are frozen here and updated separately via
    tc_estimator.train_step() on a detached copy.
    """
    tc_estimator._set_requires_grad(False)
    tc = tc_estimator.estimate(head_outputs)
    tc_estimator._set_requires_grad(True)
    return tc


class EntropyNetwork(nn.Module):
    """
    Critic T(x) usato da MINE/Donsker-Varadhan.
    Input:  (batch, d)
    Output: (batch,) oppure (batch, 1) ridotto poi a scalare per esempio.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
