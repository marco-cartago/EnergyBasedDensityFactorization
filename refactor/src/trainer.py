"""Full-model training drivers (gaussian and image pipelines)."""

import os
import pickle
from time import time_ns

import torch
import torch.nn as nn
import torch.optim as O
import tqdm

from src.information import TotalCorrelationEstimator
from src.sampler import ReplaySampler
from src.train import train_one_epoch_no_corr, train_one_epoch_TC


def _make_optimizer_and_scheduler(model, lr, n_epochs, weight_decay=1e-5):
    # betas=(0, 0.999) in the old code removed first-moment averaging.
    optimizer = O.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
    )

    warmup = max(1, min(5, n_epochs))
    scheduler = O.lr_scheduler.ChainedScheduler(
        [
            O.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
            ),
            O.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, n_epochs), eta_min=lr * 1e-1
            ),
        ]
    )
    return optimizer, scheduler


def _make_sampler(model, data_shape, sample_buf_sz, noise_frac, device):
    return ReplaySampler(
        model,
        img_shape=data_shape,
        buffer_size=sample_buf_sz,
        noise_fraction=noise_frac,
        device=device,
    )


def _save_run(model, traininfos, output_root, model_folder, epoch, data_shape):
    model_file = os.path.join(
        output_root,
        model_folder,
        f"{model.__class__.__name__}_{epoch}_epochs_{data_shape}"
        f"_shape_{model.n_heads}_heads_{time_ns()}",
    )
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    with open(model_file + ".tinfo", "wb") as f:
        pickle.dump(traininfos, f)
    with open(model_file + ".pth", "wb") as f:
        torch.save(model.state_dict(), f)
    return model_file


def train_full_model_TC(
    model: nn.Module,
    data_shape: tuple,
    n_heads: int,
    model_folder: str,
    train_loader,
    n_epochs=5,
    ebm_learning_rate=1e-3,
    sample_steps=100,
    sample_step_size=0.002,
    sample_buf_sz=128,
    noise_frac=0.05,
    noise_std=0.063,
    normalize_grad=False,
    energy_reg=0.01,
    tc_regulariz=1e-2,
    tc_lr=1e-2,
    tc_hidden_dim=16,
    tc_buffer_size=128,
    tc_batch_size=64,
    tc_updates_per_step=5,
    tc_n_steps=None,
    tc_abs_tol=None,
    output_root="./models",
    device: torch.device = torch.device("cpu"),
    verbose=False,
):
    model.to(device)
    optimizer, scheduler = _make_optimizer_and_scheduler(
        model, ebm_learning_rate, n_epochs
    )
    sampler = _make_sampler(model, data_shape, sample_buf_sz, noise_frac, device)

    tc_estimator = TotalCorrelationEstimator(
        d=n_heads,
        data_shape=data_shape,
        hidden_dim=tc_hidden_dim,
        lr=tc_lr,
        buffer_size=tc_buffer_size,
        batch_size=tc_batch_size,
        device=device,
    )
    tc_estimator.to(device)

    traininfos = []
    for epoch in tqdm.tqdm(range(n_epochs + 1)):
        _, traininfo = train_one_epoch_TC(
            model,
            sampler,
            train_loader,
            optimizer,
            sample_steps=sample_steps,
            sample_step_size=sample_step_size,
            sample_noise_std=noise_std,
            normalize_grad=normalize_grad,
            energy_reg=energy_reg,
            tc_regularizations=tc_regulariz,
            tc_estimator=tc_estimator,
            tc_updates_per_step=tc_updates_per_step,
            tc_n_steps=tc_n_steps,
            tc_abs_tol=tc_abs_tol,
            device=device,
        )
        # One scheduler step per epoch (never per batch).
        scheduler.step()

        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs + 1}, loss: {traininfo['e_loss']:.4f},"
                f" TC: {traininfo['e_corr']:.4f}, lr: {scheduler.get_last_lr()[0]:.2e}"
            )
        traininfos.append(traininfo)

    _save_run(model, traininfos, output_root, model_folder, n_epochs, data_shape)
    return traininfos, sampler, tc_estimator


def train_full_model_no_TC(
    model: nn.Module,
    data_shape: tuple,
    n_heads: int,
    model_folder: str,
    train_loader,
    n_epochs=5,
    ebm_learning_rate=1e-3,
    sample_steps=100,
    sample_step_size=0.002,
    sample_buf_sz=128,
    noise_frac=0.05,
    noise_std=0.063,
    normalize_grad=False,
    energy_reg=0.01,
    output_root="./models",
    device: torch.device = torch.device("cpu"),
    verbose=False,
):
    model.to(device)
    optimizer, scheduler = _make_optimizer_and_scheduler(
        model, ebm_learning_rate, n_epochs
    )
    sampler = _make_sampler(model, data_shape, sample_buf_sz, noise_frac, device)

    traininfos = []
    for epoch in tqdm.tqdm(range(n_epochs + 1)):
        _, traininfo = train_one_epoch_no_corr(
            model,
            sampler,
            train_loader,
            optimizer,
            sample_steps=sample_steps,
            sample_step_size=sample_step_size,
            sample_noise_std=noise_std,
            normalize_grad=normalize_grad,
            energy_reg=energy_reg,
            device=device,
        )
        scheduler.step()

        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs + 1}, loss: {traininfo['e_loss']:.4f},"
                f" lr: {scheduler.get_last_lr()[0]:.2e}"
            )
        traininfos.append(traininfo)

    _save_run(model, traininfos, output_root, model_folder, n_epochs, data_shape)

    return traininfos, sampler
