import torch
import torch.nn as nn
from src.information import TotalCorrelationEstimator

def cd_loss(e_real: torch.Tensor, e_fake: torch.Tensor, energy_regularization=0.05, return_components=False):

    cd   = e_real.mean() - e_fake.mean()
    reg  = energy_regularization * (e_real.pow(2).mean() + e_fake.pow(2).mean())

    total = cd + reg

    if return_components:
        return total, cd, reg
    return total

def head_correlation_penalty(H: torch.Tensor):
    if H.shape[0] < 2:
        return torch.tensor(0.0, device=H.device)

    H = H - H.mean(dim=0, keepdim=True)
    H = H / (H.std(dim=0, keepdim=True) + 1e-8)
    corr = H.T @ H / (H.shape[0] - 1)
    corr = corr - torch.eye(corr.size(0), device=corr.device)
    return corr.pow(2).mean()

def total_correlation_TC(
        head_optputs: torch.Tensor,
        tc_estimator: TotalCorrelationEstimator
    ):
    tc = tc_estimator.total_correlation(head_optputs)
    return tc
