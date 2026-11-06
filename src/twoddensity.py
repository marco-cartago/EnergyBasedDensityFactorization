import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def _sample_truncated(rng, mean, cov, n):
    """n samples from N(mean, cov), truncated to [0,1]^2 by rejection."""
    out = np.empty((0, 2))
    while len(out) < n:
        pts = rng.multivariate_normal(mean, cov, size=2 * n + 16)
        pts = pts[((pts >= 0) & (pts <= 1)).all(axis=1)]
        out = np.concatenate([out, pts])
    return out[:n]


def _random_covs(rng, n, std_range):
    """n random SPD 2x2 matrices: Sigma = R diag(s1^2, s2^2) R^T."""
    theta = rng.uniform(0, np.pi, size=n)
    s = rng.uniform(*std_range, size=(n, 2))  # axis std devs
    c, si = np.cos(theta), np.sin(theta)
    R = np.stack([np.stack([c, -si], -1), np.stack([si, c], -1)], -2)  # (n, 2, 2)
    return R @ (s[:, :, None] ** 2 * np.eye(2)) @ R.transpose(0, 2, 1)


def mog_dataset(
    n_samples=1000,
    n_classes=3,
    comps_per_class=1,
    std_range=(0.03, 0.08),
    covs=None,
    seed=None,
):
    """
    Randomly generates a classification problem from a mixture of gaussians
    with full (non-isotropic) covariances.

    Each class owns `comps_per_class` gaussian components. Centers are drawn
    uniformly in [0,1]^2 (with a margin of 2 * the largest axis std of the
    component). Covariances are Sigma = R(theta) diag(s1^2, s2^2) R(theta)^T
    with theta ~ U(0, pi) and s1, s2 ~ U(std_range), unless `covs` of shape
    (n_classes * comps_per_class, 2, 2) is given. Samples are truncated to
    [0,1]^2 by rejection (no clipping).

    Returns
    -------
    x       : (N, 2) float tensor in [0,1]^2
    y       : (N,)   long tensor, class labels in {0, ..., n_classes-1}
    centers : (n_comps, 2)    float tensor
    covs    : (n_comps, 2, 2) float tensor (covariances before truncation)
    """
    rng = np.random.default_rng(seed)
    n_comps = n_classes * comps_per_class

    if covs is None:
        covs = _random_covs(rng, n_comps, std_range)
    else:
        covs = np.asarray(covs, dtype=float)
        assert covs.shape == (n_comps, 2, 2)

    # per-component margin from the largest eigenvalue, so mass stays mostly inside
    max_std = np.sqrt(np.linalg.eigvalsh(covs)[:, -1])  # (n_comps,)
    m = np.minimum(2 * max_std, 0.5)[:, None]
    centers = rng.uniform(m, 1 - m, size=(n_comps, 2))

    comp = rng.integers(0, n_comps, size=n_samples)  # uniform mixture weights
    x = np.empty((n_samples, 2))
    for k in range(n_comps):
        idx = np.where(comp == k)[0]
        if len(idx):
            x[idx] = _sample_truncated(rng, centers[k], covs[k], len(idx))

    y = comp // comps_per_class  # component -> class
    perm = rng.permutation(len(x))
    x = x[perm]
    y = y[perm]

    return (
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
        torch.from_numpy(centers).float(),
        torch.from_numpy(covs).float(),
    )


def draw_cov_ellipses(ax, centers, covs, comps_per_class=1, n_std=2.0, lw=1.2):
    """Draws the n_std-sigma contour of each (untruncated) component, colored by class."""
    centers = np.asarray(centers)
    covs = np.asarray(covs)
    cmap = plt.get_cmap("tab10")
    for k, (mu, S) in enumerate(zip(centers, covs)):
        w, V = np.linalg.eigh(S)  # ascending eigenvalues
        angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))  # direction of largest axis
        width, height = 2 * n_std * np.sqrt(w[::-1])
        ax.add_patch(
            Ellipse(
                mu,
                width,
                height,
                angle=angle,
                fill=False,
                edgecolor=cmap((k // comps_per_class) % 10),
                linewidth=lw,
                linestyle="--",
            )
        )


@torch.no_grad()
def _eval_heads_on_grid(model, res=200, mode="log_density", device=None, chunk=8192):
    """Evaluates every head of `model` on a res x res grid over [0,1]^2.
    Returns X, Y (res, res) and Z (n_heads, res, res)."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

    lin = np.linspace(0, 1, res)
    X, Y = np.meshgrid(lin, lin, indexing="xy")  # X varies along columns
    grid = torch.from_numpy(np.stack([X, Y], -1).reshape(-1, 1, 2)).float().to(device)

    was_training = model.training
    model.eval()
    try:
        Z = []
        for head in model.heads:
            out = torch.cat(
                [
                    head(grid[i : i + chunk]).reshape(-1)
                    for i in range(0, len(grid), chunk)
                ]
            )
            Z.append(out.cpu().numpy().reshape(res, res))
    finally:
        model.train(was_training)
    Z = np.stack(Z)

    # Heads are generally unnormalized, so shift before exponentiating
    if mode == "log_density":
        Z = np.exp(Z - Z.max(axis=(1, 2), keepdims=True))
    elif mode == "energy":
        Z = np.exp(-(Z - Z.min(axis=(1, 2), keepdims=True)))
    elif mode != "raw":
        raise ValueError(f"unknown mode: {mode!r}")
    return X, Y, Z


def plot_model_density(
    model, axes=None, res=200, mode="log_density", device=None, cmap="magma", levels=30
):
    """
    Given a model with a `.heads` attribute (each head maps (B, 1, 2) -> scalar
    per point), plots one head per panel on [0,1]^2.

    mode: "log_density" -> exp(head(x)), "energy" -> exp(-head(x)),
          "raw" -> head(x) as is (e.g. a density or probability).
    Panels are individually normalized (shifted by max/min) for display.
    """
    X, Y, Z = _eval_heads_on_grid(model, res, mode, device)
    n = len(Z)
    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        axes = axes.ravel()
    for k, ax in enumerate(axes):
        ax.contourf(X, Y, Z[k], levels=levels, cmap=cmap)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"head {k}")
    return axes


def plot_gm_points(
    x,
    y,
    model=None,
    axes=None,
    point_size=8,
    alpha=0.8,
    centers=None,
    covs=None,
    comps_per_class=1,
    **density_kwargs,
):
    """
    Plots points generated from a MOG on the 2d plane, colored by class.
    If a model is provided, its per-head density is drawn underneath
    (one panel per head); extra kwargs go to plot_model_density.
    """
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    n_panels = len(model.heads) if model is not None else 1
    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)
        axes = axes.ravel()

    if model is not None:
        plot_model_density(model, axes=axes, **density_kwargs)

    cmap = plt.get_cmap("tab10")
    for ax in axes:
        for c in np.unique(y):
            m = y == c
            ax.scatter(
                x[m, 0],
                x[m, 1],
                s=point_size,
                alpha=alpha,
                color=cmap(int(c) % 10),
                edgecolors="white",
                linewidths=0.3,
                label=f"class {c}",
            )
        if covs is not None:
            draw_cov_ellipses(ax, centers, covs, comps_per_class)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_aspect("equal")

    axes[0].legend(loc="upper right", fontsize=7)
    return axes
