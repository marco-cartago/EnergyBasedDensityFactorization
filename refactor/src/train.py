"""Single-epoch training functions for the refactored EBM pipeline."""

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from src.information import TotalCorrelationEstimator
from src.losses import cd_loss, head_correlation_penalty, total_correlation_TC
from src.sampler import ReplaySampler


def _collect_traininfo(
    running_loss, running_cd, running_reg, running_corr,
    running_e_real, running_e_fake,
    l_loss, l_cd, l_reg, l_corr, l_e_real, l_e_fake,
    n_batches,
    verbose,
):
    n = max(n_batches, 1)
    traininfo = {
        "e_loss": running_loss / n,
        "e_cd": running_cd / n,
        "e_reg": running_reg / n,
        "e_corr": running_corr / n,
        "e_e_real": running_e_real / n,
        "e_e_fake": running_e_fake / n,
        "l_loss": l_loss,
        "l_cd": l_cd,
        "l_reg": l_reg,
        "l_corr": l_corr,
        "l_e_real": l_e_real,
        "l_e_fake": l_e_fake,
    }
    if verbose:
        print(f"  CD:     {traininfo['e_cd']:.4f}")
        print(f"  Reg:    {traininfo['e_reg']:.4f}")
        print(f"  Corr:   {traininfo['e_corr']:.4f}")
        print(f"  E_real: {traininfo['e_e_real']:.4f}")
        print(f"  E_fake: {traininfo['e_e_fake']:.4f}")
        print(
            f"  Gap:    {traininfo['e_e_real'] - traininfo['e_e_fake']:.4f}"
        )
    return traininfo


def train_one_epoch(
    model: nn.Module,
    sampler: ReplaySampler,
    train_loader: DataLoader,
    optimizer,
    sample_steps: int,
    sample_step_size: float,
    sample_noise_std: float,
    normalize_grad: bool,
    energy_reg: float,
    corr_param: float,
    clip_gradient: bool = False,
    device="cpu",
    verbose: bool = False,
):
    """Legacy Pearson-correlation variant (kept for ablations)."""
    model.train()
    running_loss = running_cd = running_reg = running_corr = 0.0
    running_e_real = running_e_fake = 0.0

    for x_real, _ in tqdm.tqdm(train_loader):
        x_real = x_real.to(device)
        x_neg = sampler.sample(
            batch_size=x_real.size(0),
            steps=sample_steps,
            step_size=sample_step_size,
            noise_std=sample_noise_std,
            normalize_grad=normalize_grad,
        )

        e_fake, h_fake = model(x_neg)
        e_real, h_real = model(x_real)

        loss, cd, reg = cd_loss(
            e_fake=e_fake, e_real=e_real,
            energy_regularization=energy_reg, return_components=True,
        )
        corr = corr_param * (
            head_correlation_penalty(h_real) + head_correlation_penalty(h_fake)
        )
        loss += corr

        running_loss += loss.item()
        running_cd += cd.item()
        running_reg += reg.item()
        running_corr += corr.item()
        running_e_real += e_real.mean().item()
        running_e_fake += e_fake.mean().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_gradient:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    n = len(train_loader)
    return _collect_traininfo(
        running_loss, running_cd, running_reg, running_corr,
        running_e_real, running_e_fake, [], [], [], [], [], [], n, verbose,
    )["e_loss"]


def train_one_epoch_TC(
    model: nn.Module,
    sampler: ReplaySampler,
    train_loader: DataLoader,
    optimizer,
    sample_steps: int,
    sample_step_size: float,
    sample_noise_std: float,
    normalize_grad: bool,
    energy_reg: float,
    tc_regularizations: float,
    tc_estimator: TotalCorrelationEstimator,
    tc_updates_per_step: int = 5,
    tc_n_steps: int | None = None,
    tc_abs_tol: float | None = None,
    clip_gradient: bool = False,
    train_noise: float | None = None,
    verbose: bool = False,
    device: torch.device = torch.device("cpu"),
):
    """CD + energy regularization + differentiable TC penalty.

    The TC critic is frozen while its bound contributes to the model loss;
    it is then updated once per batch on detached head outputs.  ``tc_n_steps``
    and ``tc_abs_tol`` are kept for API compatibility with the old call sites.
    """
    model.train()

    running_loss = running_cd = running_reg = running_corr = 0.0
    running_e_real = running_e_fake = 0.0
    l_loss, l_cd, l_reg, l_corr = [], [], [], []
    l_e_real, l_e_fake = [], []

    for x_real, _ in tqdm.tqdm(train_loader):
        x_real = x_real.to(device)

        if train_noise is not None:
            x_real = (x_real + torch.randn_like(x_real) * train_noise).clamp_(-1, 1)

        x_neg = sampler.sample(
            batch_size=x_real.size(0),
            steps=sample_steps,
            step_size=sample_step_size,
            noise_std=sample_noise_std,
            normalize_grad=normalize_grad,
        )

        e_fake, h_fake = model(x_neg)
        e_real, h_real = model(x_real)
        samples = torch.cat((h_real, h_fake), dim=0)

        loss, cd, reg = cd_loss(
            e_fake=e_fake, e_real=e_real,
            energy_regularization=energy_reg, return_components=True,
        )

        with tc_estimator.frozen():
            corr = tc_regularizations * total_correlation_TC(samples, tc_estimator)
        loss += corr

        loss_v, cd_v, reg_v, corr_v = (
            loss.item(), cd.item(), reg.item(), corr.item()
        )
        e_real_v, e_fake_v = e_real.mean().item(), e_fake.mean().item()
        running_loss += loss_v
        running_cd += cd_v
        running_reg += reg_v
        running_corr += corr_v
        running_e_real += e_real_v
        running_e_fake += e_fake_v
        l_loss.append(loss_v)
        l_cd.append(cd_v)
        l_reg.append(reg_v)
        l_corr.append(corr_v)
        l_e_real.append(e_real_v)
        l_e_fake.append(e_fake_v)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_gradient:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # The critic must see the *current* model's heads, not the pre-update
        # tensors used in the penalty; otherwise it lags behind and the bound
        # collapses to zero while the heads are still dependent.
        with torch.no_grad():
            _, h_real_now = model(x_real)
            _, h_fake_now = model(x_neg)
            critic_samples = torch.cat((h_real_now, h_fake_now), dim=0)
        for _ in range(max(1, tc_updates_per_step)):
            tc_estimator.train_step(critic_samples)

    traininfo = _collect_traininfo(
        running_loss, running_cd, running_reg, running_corr,
        running_e_real, running_e_fake,
        l_loss, l_cd, l_reg, l_corr, l_e_real, l_e_fake,
        len(train_loader), verbose,
    )
    return traininfo["e_loss"], traininfo


def train_one_epoch_no_corr(
    model: nn.Module,
    sampler: ReplaySampler,
    train_loader: DataLoader,
    optimizer,
    sample_steps: int,
    sample_step_size: float,
    sample_noise_std: float,
    normalize_grad: bool,
    energy_reg: float,
    clip_gradient: bool = False,
    train_noise: float | None = None,
    verbose: bool = False,
    device: torch.device = torch.device("cpu"),
):
    """Same workflow as the TC variant, with the correlation term disabled."""
    model.train()

    running_loss = running_cd = running_reg = running_corr = 0.0
    running_e_real = running_e_fake = 0.0
    l_loss, l_cd, l_reg, l_corr = [], [], [], []
    l_e_real, l_e_fake = [], []

    for x_real, _ in tqdm.tqdm(train_loader):
        x_real = x_real.to(device)

        if train_noise is not None:
            x_real = (x_real + torch.randn_like(x_real) * train_noise).clamp_(-1, 1)

        x_neg = sampler.sample(
            batch_size=x_real.size(0),
            steps=sample_steps,
            step_size=sample_step_size,
            noise_std=sample_noise_std,
            normalize_grad=normalize_grad,
        )

        e_fake, _ = model(x_neg)
        e_real, _ = model(x_real)

        loss, cd, reg = cd_loss(
            e_fake=e_fake, e_real=e_real,
            energy_regularization=energy_reg, return_components=True,
        )

        loss_v, cd_v, reg_v = loss.item(), cd.item(), reg.item()
        e_real_v, e_fake_v = e_real.mean().item(), e_fake.mean().item()
        running_loss += loss_v
        running_cd += cd_v
        running_reg += reg_v
        running_e_real += e_real_v
        running_e_fake += e_fake_v
        l_loss.append(loss_v)
        l_cd.append(cd_v)
        l_reg.append(reg_v)
        l_corr.append(0.0)
        l_e_real.append(e_real_v)
        l_e_fake.append(e_fake_v)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_gradient:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    traininfo = _collect_traininfo(
        running_loss, running_cd, running_reg, running_corr,
        running_e_real, running_e_fake,
        l_loss, l_cd, l_reg, l_corr, l_e_real, l_e_fake,
        len(train_loader), verbose,
    )
    return traininfo["e_loss"], traininfo
