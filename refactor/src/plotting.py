"""Shared plotting utilities for the refactored experiments."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_traininfo(
    traininfos,
    smooth=5,
    plot_batch=False,
    figsize=(16, 10),
    save_path=None,
    dpi=200,
    show=False,
):
    """Plot epoch metrics; batch metrics are flattened across epochs."""
    if not traininfos:
        raise ValueError("traininfos is empty")

    epochs = np.arange(1, len(traininfos) + 1)

    def scalar_values(key):
        return np.array(
            [info.get(key, np.nan) for info in traininfos], dtype=float
        )

    def moving_average(values, window):
        values = np.asarray(values, dtype=float)
        if window <= 1 or len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode="valid")

    def plot_curve(ax, key, label, color=None):
        values = scalar_values(key)
        ax.plot(epochs, values, label=label, color=color,
                alpha=0.35 if smooth > 1 else 1.0)
        if smooth > 1 and len(values) >= smooth:
            ax.plot(
                epochs[smooth - 1:], moving_average(values, smooth),
                color=color, linewidth=2.2, label=f"{label} (MA{smooth})",
            )

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    plot_curve(ax1, "e_loss", "Total loss")
    ax1.set(title="Training convergence - Total loss", xlabel="Epoch", ylabel="Loss")

    for key, label in [
        ("e_cd", "Contrastive Divergence"),
        ("e_reg", "Energy regularization"),
        ("e_corr", "TC regularization"),
    ]:
        plot_curve(ax2, key, label)
    ax2.set(title="Loss components", xlabel="Epoch", ylabel="Contribution")

    plot_curve(ax3, "e_e_real", "Real energy")
    plot_curve(ax3, "e_e_fake", "Fake energy")
    ax3.set(title="Energy - Real vs Fake", xlabel="Epoch", ylabel="Energy")

    for key, label in [
        ("e_cd", "CD"), ("e_reg", "Energy reg."),
        ("e_corr", "TC reg."), ("e_loss", "Total loss"),
    ]:
        plot_curve(ax4, key, label)
    ax4.set(title="Loss contributions", xlabel="Epoch", ylabel="Loss")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()

    fig_batch = None
    if plot_batch:
        fig_batch, axes_batch = plt.subplots(2, 2, figsize=figsize)
        for ax, key, title in zip(
            axes_batch.flatten(),
            ["l_loss", "l_cd", "l_reg", "l_corr"],
            ["Batch loss", "Batch CD", "Batch energy reg.", "Batch TC reg."],
        ):
            series = []
            for info in traininfos:
                v = info.get(key, [])
                if isinstance(v, (list, tuple, np.ndarray)):
                    series.append(np.asarray(v, dtype=float))
            if series:
                flat = np.concatenate(series)
                ax.plot(flat, alpha=0.3, label="batch")
                if smooth > 1:
                    ax.plot(
                        np.arange(smooth - 1, len(flat)),
                        moving_average(flat, smooth),
                        linewidth=2, label=f"MA{smooth}",
                    )
            ax.set(title=title, xlabel="Batch", ylabel="Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig_batch.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if fig_batch is not None:
            base, ext = os.path.splitext(save_path)
            fig_batch.savefig(f"{base}_batch{ext or '.png'}", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    return fig, fig_batch


def save_image_grid(samples: torch.Tensor, path: str, n=16, ncols=4):
    """Save a grid of images assuming values live in [-1, 1]."""
    x = samples.detach().cpu()[:n]
    if x.ndim == 4:
        x = x[:, 0] if x.shape[1] == 1 else x.mean(dim=1)
    n = len(x)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(((x[i] + 1) / 2).clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
