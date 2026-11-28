"""Loss components used by the refactored training loop."""

import torch

from src.information import TC_Estimator


def cd_loss(
    e_real: torch.Tensor,
    e_fake: torch.Tensor,
    energy_regularization: float = 0.01,
    return_components: bool = False,
):
    """Contrastive divergence with a (weak) quadratic energy penalty.

    Note: with CD = E_real - E_fake alone the optimum pushes the two means
    apart indefinitely.  The L2 term bounds them, but for a coefficient a its
    pointwise optimum is E_real = -1/(2a), E_fake = +1/(2a), i.e. constant
    energies.  Keep ``energy_regularization`` small and rely on the bounded
    (spectral-normalized) heads to retain input dependence.
    """
    cd = e_real.mean() - e_fake.mean()
    reg = energy_regularization * (e_real.pow(2).mean() + e_fake.pow(2).mean())
    total = cd + reg

    if return_components:
        return total, cd, reg
    return total


def head_correlation_penalty(H: torch.Tensor):
    H = H.reshape(H.shape[0], -1)
    if H.shape[0] < 2 or H.shape[1] < 2:
        return torch.zeros((), device=H.device)

    H = H - H.mean(dim=0, keepdim=True)
    H = H / (H.std(dim=0, keepdim=True) + 1e-8)
    corr = H.T @ H / (H.shape[0] - 1)
    corr = corr - torch.eye(corr.size(0), device=corr.device)
    return corr.pow(2).mean()


def total_correlation_TC(
    head_outputs: torch.Tensor,
    tc_estimator: TC_Estimator,
):
    return tc_estimator.dv_bound(head_outputs)
