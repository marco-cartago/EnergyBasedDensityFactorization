import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from src.information import TotalCorrelationEstimator
from src.losses import cd_loss, head_correlation_penalty, total_correlation_TC
from src.sampler import ReplaySampler



def train_one_epoch(
    model: nn.Module,
    sampler: ReplaySampler,
    train_loader: DataLoader,
    optimizer,
    sample_steps: int,
    sample_step_size: float,
    sample_noise_std: float,
    energy_reg: float,
    corr_param: float,
    scheduler=None,
    clip_gradient=False,
    device="cpu",
    verbose=False,
):

    model.train()
    running_loss = 0.0
    running_cd = 0.0
    running_reg = 0.0
    running_corr = 0.0
    running_e_real = 0.0
    running_e_fake = 0.0

    for x_real, _ in tqdm.tqdm(train_loader):
        x_real = x_real.to(device)

        x_neg = sampler.sample(
            batch_size=x_real.size(0),
            steps=sample_steps,
            step_size=sample_step_size,
            noise_std=sample_noise_std,
        )

        e_fake, h_fake = model(x_neg)
        e_real, h_real = model(x_real)

        loss, cd, reg = cd_loss(
            e_fake=e_fake,
            e_real=e_real,
            energy_regularization=energy_reg,
            return_components=True,
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

        optimizer.zero_grad()
        loss.backward()
        if clip_gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

    n = len(train_loader)
    if verbose:
        print(f"  CD:     {running_cd / n:.4f}")
        print(f"  Reg:    {running_reg / n:.4f}")
        print(f"  Corr:   {running_corr / n:.4f}")
        print(f"  E_real: {running_e_real / n:.4f}")
        print(f"  E_fake: {running_e_fake / n:.4f}")
        print(f"  Gap:    {(running_e_real - running_e_fake) / n:.4f}")

    return running_loss / n


def train_one_epoch_TC(
    model: nn.Module,
    sampler: ReplaySampler,
    train_loader: DataLoader,
    optimizer: torch.optim.optimizer,
    
    sample_steps: int,
    sample_step_size: float,
    sample_noise_std: float,
    
    energy_reg: float,
    tc_regularizations,
    tc_estimator: TotalCorrelationEstimator,
    tc_n_steps: int,

    scheduler=None,
    
    clip_gradient: bool = False,
    train_noise: float | None = None,
    verbose: bool = False,
    device: torch.device = torch.device("cpu"),
):

    running_loss = 0.0
    running_cd = 0.0
    running_reg = 0.0
    running_corr = 0.0
    running_e_real = 0.0
    running_e_fake = 0.0

    l_running_loss = []
    l_running_cd = []
    l_running_reg = []
    l_running_corr = []
    l_running_e_real = []
    l_running_e_fake = []

    for x_real, _ in tqdm.tqdm(train_loader):
        x_real = x_real.to(device)
        if train_noise is not None:
            small_noise = torch.randn_like(x_real) * train_noise
            x_real.add_(small_noise).clamp_(min=-1.0, max=1.0)

        x_neg = sampler.sample(
            batch_size=x_real.size(0),
            steps=sample_steps,
            step_size=sample_step_size,
            noise_std=sample_noise_std,
        )

        e_fake, h_fake = model(x_neg)
        e_real, h_real = model(x_real)

        loss, cd, reg = cd_loss(
            e_fake=e_fake,
            e_real=e_real,
            energy_regularization=energy_reg,
            return_components=True,
        )
        corr = tc_regularizations * total_correlation_TC(
            head_optputs=h_real, tc_estimator=tc_estimator
        )
        loss += corr

        running_loss += loss.item()
        running_cd += cd.item()
        running_reg += reg.item()
        running_corr += corr.item()
        running_e_real += e_real.mean().item()
        running_e_fake += e_fake.mean().item()

        l_running_loss.append(loss.item())
        l_running_cd.append(cd.item())
        l_running_reg.append(reg.item())
        l_running_corr.append(corr.item())
        l_running_e_real.append(e_real.mean().item())
        l_running_e_fake.append(e_fake.mean().item())

        # Update the model
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update the TC estimator
        samples = torch.cat((h_real.detatch(), h_fake.detatch()))
        tc_estimator.train_steps(samples, tc_n_steps)

        # LR scheduler step
        if scheduler is not None:
            scheduler.step()

    n = len(train_loader)

    traininfo = {
        "e_loss": running_loss / n,
        "e_cd": running_cd / n,
        "e_reg": running_reg / n,
        "e_corr": running_corr / n,
        "e_e_real": running_e_real / n,
        "e_e_fake": running_e_fake / n,
        "l_loss": l_running_loss,
        "l_cd": l_running_cd,
        "l_reg": l_running_reg,
        "l_corr": l_running_corr,
        "l_e_real": l_running_e_real,
        "l_e_fake": l_running_e_fake,
    }

    if verbose:
        print(f"  CD:     {running_cd / n:.4f}")
        print(f"  Reg:    {running_reg / n:.4f}")
        print(f"  Corr:   {running_corr / n:.4f}")
        print(f"  E_real: {running_e_real / n:.4f}")
        print(f"  E_fake: {running_e_fake / n:.4f}")
        print(f"  Gap:    {(running_e_real - running_e_fake) / n:.4f}")

    return running_loss / n, traininfo


from src.MINE import MINE_TC_Estimator, mine_tc_regularization


def train_one_epoch_MINE(
    model: nn.Module,
    sampler: ReplaySampler,
    train_loader: DataLoader,
    optimizer: torch.optim.optimizer,
    sample_steps: int,
    sample_step_size: float,
    sample_noise_std: float,
    energy_reg: float,
    tc_regularizations: float,
    tc_estimator: MINE_TC_Estimator,
    tc_updates_per_step: int = 1,
    scheduler=None,
    clip_gradient: bool = False,
    verbose: bool = False,
    device: torch.device = torch.device("cpu"),
):
    running_loss = 0.0
    running_cd = 0.0
    running_reg = 0.0
    running_corr = 0.0
    running_e_real = 0.0
    running_e_fake = 0.0
    l_running_loss = []
    l_running_cd = []
    l_running_reg = []
    l_running_corr = []
    l_running_e_real = []
    l_running_e_fake = []

    for x_real, _ in tqdm.tqdm(train_loader):
        x_real = x_real.to(device)
        x_neg = sampler.sample(
            batch_size=x_real.size(0),
            steps=sample_steps,
            step_size=sample_step_size,
            noise_std=sample_noise_std,
        )
        e_fake, h_fake = model(x_neg)
        e_real, h_real = model(x_real)

        loss, cd, reg = cd_loss(
            e_fake=e_fake,
            e_real=e_real,
            energy_regularization=energy_reg,
            return_components=True,
        )

        corr = tc_regularizations * mine_tc_regularization(h_real, tc_estimator)
        loss += corr

        running_loss += loss.item()
        running_cd += cd.item()
        running_reg += reg.item()
        running_corr += corr.item()
        running_e_real += e_real.mean().item()
        running_e_fake += e_fake.mean().item()
        l_running_loss.append(loss.item())
        l_running_cd.append(cd.item())
        l_running_reg.append(reg.item())
        l_running_corr.append(corr.item())
        l_running_e_real.append(e_real.mean().item())
        l_running_e_fake.append(e_fake.mean().item())

        # Update the model
        optimizer.zero_grad()
        loss.backward()
        if clip_gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update the MINE statistics network (on detached heads)
        for _ in range(tc_updates_per_step):
            tc_estimator.train_step(h_real.detach())

        if scheduler is not None:
            scheduler.step()

    n = len(train_loader)
    traininfo = {
        "e_loss": running_loss / n,
        "e_cd": running_cd / n,
        "e_reg": running_reg / n,
        "e_corr": running_corr / n,
        "e_e_real": running_e_real / n,
        "e_e_fake": running_e_fake / n,
        "l_loss": l_running_loss,
        "l_cd": l_running_cd,
        "l_reg": l_running_reg,
        "l_corr": l_running_corr,
        "l_e_real": l_running_e_real,
        "l_e_fake": l_running_e_fake,
    }
    if verbose:
        print(f"  CD:     {running_cd / n:.4f}")
        print(f"  Reg:    {running_reg / n:.4f}")
        print(f"  Corr:   {running_corr / n:.4f}")
        print(f"  E_real: {running_e_real / n:.4f}")
        print(f"  E_fake: {running_e_fake / n:.4f}")
        print(f"  Gap:    {(running_e_real - running_e_fake) / n:.4f}")
    return running_loss / n, traininfo
