import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.sampler import ReplaySampler


def diagnose(
    model: nn.Module,
    sampler: ReplaySampler,
    img_shape: tuple,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device="cpu",
    n_samples=16,
):
    """
    Given a model calculates and prints diagnostics on:
     1. The difference between energy on real and on fake data (replay buffer)
     2. Energy on pure noise
     3. Head means and correlation on a sample batch
     4. Size of the replay buffer
     5. Size of the replay buffer
     6. Visual samples
    """

    model.eval()

    print("=" * 50)
    print("EBM DIAGNOSTICS")
    print("=" * 50)

    with torch.no_grad():
        # 1. Energy on real data
        print("\n[1] Energy on real data")

        e_train_all = []
        for x, _ in train_loader:
            e, _ = model(x.to(device))
            e_train_all.append(e)
        e_train = torch.cat(e_train_all)

        e_test_all = []
        for x, _ in test_loader:
            e, _ = model(x.to(device))
            e_test_all.append(e)
        e_test = torch.cat(e_test_all)

        print(f"  Train  →  mean={e_train.mean():.4f},  std={e_train.std():.4f}")
        print(f"  Test   →  mean={e_test.mean():.4f},  std={e_test.std():.4f}")
        gap_train_test = (e_test.mean() - e_train.mean()).abs().item()
        print(f"  |E_test - E_train| = {gap_train_test:.4f}  ", end="")
        print("(ok)" if gap_train_test < 0.2 else "(WARNING: possible overfit)")

        # 2. Energy on fake data
        print("\n[2] Energy on fake data (Langevin samples)")

        x_neg = sampler.sample(
            batch_size=64, steps=100, step_size=10.0, noise_std=0.005
        )
        e_fake, _ = model(x_neg)

        print(f"  Fake   →  mean={e_fake.mean():.4f},  std={e_fake.std():.4f}")
        print(
            f"  Gap (E_real_train - E_fake) = {(e_train.mean() - e_fake.mean()):.4f}  ",
            end="",
        )
        print("(ok)" if e_train.mean() < e_fake.mean() else "(WARNING: gap inverted)")

        # 3. Energy on pure noise
        print("\n[3] Energy on pure noise")

        x_noise = torch.rand((64, *img_shape), device=device) * 2 - 1
        e_noise, _ = model(x_noise)

        print(f"  Noise  →  mean={e_noise.mean():.4f},  std={e_noise.std():.4f}")
        print(
            f"  Gap (E_real_train - E_noise) = {(e_train.mean() - e_noise.mean()):.4f}  ",
            end="",
        )
        print(
            "(ok)"
            if e_train.mean() < e_noise.mean()
            else "(WARNING: model assigns low energy to noise)"
        )

        # 4. Per-head analysis
        print("\n[4] Per-head energy (on train batch)")

        x_batch, _ = next(iter(train_loader))
        x_batch = x_batch.to(device)
        _, h = model(x_batch)  # h: (B, n_heads)

        for i in range(h.shape[1]):
            hi = h[:, i]
            print(f"  Head {i}  →  mean={hi.mean():.4f},  std={hi.std():.4f}")

        # Check head correlation
        H = h - h.mean(dim=0, keepdim=True)
        H = H / (H.std(dim=0, keepdim=True) + 1e-8)
        corr = (H.T @ H / (H.shape[0] - 1)).cpu()
        corr_off = corr - torch.eye(corr.size(0))
        print(f"  Max off-diagonal correlation: {corr_off.abs().max():.4f}  ", end="")
        print(
            "(ok)" if corr_off.abs().max() < 0.5 else "(WARNING: heads are correlated)"
        )

        # 5. Sampler buffer
        print("\n[5] Replay buffer")
        print(f"  Buffer size: {len(sampler.buffer)} / {sampler.buffer_size}")
        if len(sampler.buffer) > 0:
            buf = sampler.buffer[:64]
            print(
                f"  Buffer samples  →  mean={buf.mean():.4f},  std={buf.std():.4f},  min={buf.min():.4f},  max={buf.max():.4f}"
            )

    # 6. Visual samples
    print("\n[6] Visual samples (Langevin from noise)")
    x_vis = sampler.sample(
        batch_size=n_samples, steps=200, step_size=10.0, noise_std=0.005
    )
    x_vis = x_vis.cpu()

    grid_size = int(n_samples**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        img = x_vis[idx].squeeze()
        img = (img + 1.0) / 2.0
        img = img.clamp(0, 1).numpy()
        ax.imshow(img, cmap="gray")
        e_i = e_fake[idx].item() if idx < len(e_fake) else 0.0
        ax.set_title(f"E={e_i:.2f}", fontsize=7)
        ax.axis("off")
    plt.suptitle("Generated samples", fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


def mean_energy(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    n_batches=5,
):
    """
    Evaluates the mean difference between the
    energies of the train set and of the test set.
    """

    model.eval()
    energies_train = []
    energies_test = []

    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= n_batches:
                break
            e, _ = model(x.to(device))
            energies_train.append(e.mean().item())

        for i, (x, _) in enumerate(test_loader):
            if i >= n_batches:
                break
            e, _ = model(x.to(device))
            energies_test.append(e.mean().item())

        mean_e_train = np.mean(energies_train)
        mean_e_test = np.mean(energies_test)
        print(
            f"E_train={mean_e_train:.4g}, E_test={mean_e_test:.4g}, gap={mean_e_test - mean_e_train:.4g}"
        )

        return mean_e_train, mean_e_test


def nearest_neighbor_distance(
    generated, train_loader, n_samples=1000, device=None, normalize=True
):
    """
    Check the mean distance (normalized for the size of the image) between a set generated from the model and training points
    If the distance goes to zero, the model may be overfitting by generating samples really close to data points
    """
    device = device or generated.device

    train_data = []
    n_collected = 0
    for x, _ in train_loader:
        train_data.append(x)
        n_collected += x.size(0)
        if n_collected >= n_samples:
            break

    train_data = torch.cat(train_data, dim=0)[:n_samples].to(device)

    gen_flat = generated.view(generated.size(0), -1)
    train_flat = train_data.view(train_data.size(0), -1)

    dists = torch.cdist(gen_flat, train_flat)

    if normalize:
        dists = dists / (gen_flat.size(1) ** 0.5)  # RMS distance per pixel

    min_dists, min_idx = dists.min(dim=1)
    return min_dists, min_idx
