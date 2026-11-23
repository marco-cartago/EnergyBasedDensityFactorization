import os

import torch
from torch import nn
import torch.utils.data as D

from src.twoddensity import mog_circle_dataset, mog_dataset
from src.model import EBM2_Dense, EBM2
from src.twoddensity import *
from src.trainer import train_full_model_TC, train_full_model_no_TC
from src.sampler import ReplaySampler

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def plot_traininfo(
    traininfos,
    smooth=5,
    plot_batch=False,
    figsize=(16, 10),
    save_path=None,
    dpi=300,
    show=True,
):
    """
    Visualizza e/o salva le metriche di training.

    Parameters
    ----------
    traininfos : list[dict]
        Metriche per ogni epoca.

    smooth : int
        Moving average window. 1 = nessuno smoothing.

    plot_batch : bool
        Se True, plotta anche le metriche dell'ultimo batch.

    figsize : tuple
        Dimensione della figura.

    save_path : str or None
        Path in cui salvare il grafico principale.
        Se None, non salva.

    dpi : int
        Qualità del salvataggio.

    show : bool
        Se True, mostra i grafici.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura principale.

    fig_batch : matplotlib.figure.Figure or None
        Figura delle metriche batch, se richiesta.
    """

    if not traininfos:
        raise ValueError("traininfos è vuoto.")

    epochs = np.arange(1, len(traininfos) + 1)

    def get_values(key):
        return np.array(
            [info.get(key, np.nan) for info in traininfos],
            dtype=float,
        )

    def moving_average(values, window):
        if window <= 1 or len(values) < window:
            return values

        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    def plot_curve(ax, key, label, color=None, linestyle="-"):
        values = get_values(key)

        ax.plot(
            epochs,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            alpha=0.35 if smooth > 1 else 1.0,
        )

        if smooth > 1 and len(values) >= smooth:
            smoothed = moving_average(values, smooth)

            ax.plot(
                epochs[smooth - 1:],
                smoothed,
                color=color,
                linewidth=2.2,
                label=f"{label} (MA{smooth})",
            )

    # ==================================================
    # MAIN FIGURE
    # ==================================================

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax1, ax2, ax3, ax4 = axes.flatten()

    # 1. Total loss
    plot_curve(ax1, "e_loss", "Total loss")

    ax1.set_title("Training convergence - Total loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Loss components
    plot_curve(ax2, "e_cd", "Contrastive Divergence")
    plot_curve(ax2, "e_reg", "Energy regularization")
    plot_curve(ax2, "e_corr", "TC regularization")

    ax2.set_title("Loss components")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss contribution")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Real vs Fake energy
    plot_curve(ax3, "e_e_real", "Real energy")
    plot_curve(ax3, "e_e_fake", "Fake energy")

    ax3.set_title("Energy - Real vs Fake")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Energy")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Comparison of loss contributions
    plot_curve(ax4, "e_cd", "CD")
    plot_curve(ax4, "e_reg", "Energy regularization")
    plot_curve(ax4, "e_corr", "TC regularization")
    plot_curve(ax4, "e_loss", "Total loss")

    ax4.set_title("Loss contributions comparison")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.tight_layout()

    # ==================================================
    # SAVE MAIN FIGURE
    # ==================================================

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"Main plot saved to: {save_path}")

    # ==================================================
    # BATCH FIGURE
    # ==================================================

    fig_batch = None

    if plot_batch:
        fig_batch, axes_batch = plt.subplots(2, 2, figsize=figsize)

        batch_keys = [
            ("l_loss", "Batch total loss"),
            ("l_cd", "Batch CD"),
            ("l_reg", "Batch energy regularization"),
            ("l_corr", "Batch TC regularization"),
        ]

        for ax, (key, title) in zip(axes_batch.flatten(), batch_keys):
            values = get_values(key)

            ax.plot(
                epochs,
                values,
                alpha=0.4,
                label="Batch value",
            )

            if smooth > 1 and len(values) >= smooth:
                ax.plot(
                    epochs[smooth - 1:],
                    moving_average(values, smooth),
                    linewidth=2,
                    label=f"MA{smooth}",
                )

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig_batch.tight_layout()

        # Save batch plot separately
        if save_path is not None:
            base, ext = os.path.splitext(save_path)

            batch_path = f"{base}_batch{ext or '.png'}"

            fig_batch.savefig(
                batch_path,
                dpi=dpi,
                bbox_inches="tight",
            )

            print(f"Batch plot saved to: {batch_path}")

    # ==================================================
    # DISPLAY
    # ==================================================

    if show:
        plt.show()

    return fig, fig_batch

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Parameters
# -------------------------------------------------------------------------------------- #
os.makedirs("./ablations/gauss", exist_ok=True)
os.makedirs(f"./ablations/gauss/{time_stamp}", exist_ok=True)

plot_path = f"./ablations/gauss/{time_stamp}/plots"
os.makedirs(plot_path, exist_ok=True)

# The seed (duh)
SEED = 200

# Data
SZ_GAUS = (1, 2)
N_SAMPLES = 1000
N_CLASSES = 6
STD_RANGE = (0.05, 0.1)
# Training
BATCH_SIZE = 32
N_EPOCH = 1_000
TRAIN_PER = 0.8
SHUFFLE = False

# Model
N_HEAD = 2
HID_DIM_SIZE = 32
NETWORK_DEPTH = 2
TC_REG_STR = 1e-2
ENERGY_REGULARIZ = 0.5
SAMPLE_STEP_SIZE = 1
LR = 1e-3
SAMPLE_STEPS = 50

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"CURRENTLY USING: {str(DEVICE).capitalize()}"
    + (" :) " if torch.cuda.is_available() else "")
)

# Dump parameters and to file
with open(f"./ablations/gauss/{time_stamp}/params.txt", "w") as f:
    f.write(
        f"SEED: {SEED}\n"
        + f"SZ_GAUS: {SZ_GAUS}\n"
        + f"N_SAMPLES: {N_SAMPLES}\n"
        + f"N_CLASSES: {N_CLASSES}\n"
        + f"BATCH_SIZE: {BATCH_SIZE}\n"
        + f"N_EPOCH: {N_EPOCH}\n"
        + f"TRAIN_PER: {TRAIN_PER}\n"
        + f"SHUFFLE: {SHUFFLE}\n"
        + f"N_HEAD: {N_HEAD}\n"
        + f"HID_DIM_SIZE: {HID_DIM_SIZE}\n"
        + f"NETWORK_DEPTH: {NETWORK_DEPTH}\n"
        + f"TC_REG_STR: {TC_REG_STR}\n"
        + f"ENERGY_REGULARIZ: {ENERGY_REGULARIZ}\n"
        + f"SAMPLE_STEP_SIZE: {SAMPLE_STEP_SIZE}\n"
        + f"LR: {LR}\n"
        + f"SAMPLE_STEPS: {SAMPLE_STEPS}\n"
        + f"DEVICE: {str(DEVICE).capitalize()}\n"
    )


# Training
print("TRAINING MODEL")
print(f"GENERATED {N_SAMPLES} FROM {N_CLASSES} GAUSSIANS.", end="")
X, y, means, covs = mog_dataset(
    n_samples=N_SAMPLES, n_classes=N_CLASSES, std_range=STD_RANGE, seed=SEED
)
print(" [✓]")

sep = int(N_SAMPLES * TRAIN_PER)
Xtrain, Xtest, ytrain, ytest = X[:sep], X[sep:], y[:sep], y[sep:]


print(f"CREATING DATASET", end="")
data_train, data_test = D.TensorDataset(Xtrain, ytrain), D.TensorDataset(Xtest, ytest)
train_dataloader = D.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = D.DataLoader(data_test, batch_size=BATCH_SIZE)
print(" [✓]")

print("CREATING AND COMPILING MODEL", end="")
model_gauss = EBM2(n_head=N_HEAD, hid_dim=HID_DIM_SIZE, depth=NETWORK_DEPTH)
model_gauss.to(DEVICE)
print(" [✓]")

# Model is trained (and saved?) along the (long?) way
# -------------------------------------------------------------------------------------- #

# train_full_model_no_TC(
#     model=model_gauss,
#     sample_step_size=SAMPLE_STEP_SIZE,
#     ebm_learning_rate=LR,
#     energy_reg=ENERGY_REGULARIZ,
#     sample_steps=SAMPLE_STEPS,
#     data_shape=SZ_GAUS,
#     n_heads=N_HEAD,
#     model_folder="gauss",
#     train_loader=train_dataloader,
#     n_epochs=N_EPOCH,
#     device=DEVICE,
#     verbose=False,
# )

print("TRAINING MODEL")

tinfo = train_full_model_TC(
    model=model_gauss,
    tc_regulariz=TC_REG_STR,
    sample_step_size=SAMPLE_STEP_SIZE,
    ebm_learning_rate=LR,
    energy_reg=ENERGY_REGULARIZ,
    sample_steps=SAMPLE_STEPS,
    data_shape=SZ_GAUS,
    n_heads=N_HEAD,
    model_folder="gauss",
    train_loader=train_dataloader,
    n_epochs=N_EPOCH,
    device=DEVICE,
    verbose=False,
)

model_file = "./ablations/gauss/" + f"{time_stamp}/" + f"_shape_{model_gauss.n_heads}_heads_{time_stamp}.pth"
with open(model_file, "wb") as f:
    torch.save(model_gauss.state_dict(), f)

gen = plot_gm_points(Xtrain, ytrain, centers=means, covs=covs, title="Training Data", save_path=f"{plot_path}/train_data.png")

plot_traininfo(
    tinfo,
    smooth=5,
    save_path=f"{plot_path}/train_info.png",
    show=False,
)

print("EVALUATING MODEL")

n_samples = 1250

sampler = ReplaySampler(model=model_gauss, img_shape=SZ_GAUS, buffer_size=600, noise_fraction=0.05, device=DEVICE)

samples = sampler.generate(n_samples, steps=1050, step_size=SAMPLE_STEP_SIZE, noise_std=0.005)
samples = samples.squeeze(1)

data = plot_gm_points(samples, torch.ones(n_samples, dtype=torch.long), model=model_gauss, centers=means, covs=covs, mode="raw",  cmap="gray", save_path=f"{plot_path}/generated_samples.png")
gen = plot_gm_points(Xtrain, ytrain, model=model_gauss, centers=means, covs=covs, mode="raw", cmap="gray", title="Eval on Training Data", save_path=f"{plot_path}/eval_on_train_data.png")
plot_model_density(model_gauss, mode = "log_density", save_path=f"{plot_path}/model_density.png")


# Sampling from single heads
class HeadView(nn.Module):
    def __init__(self, model, i):
        super().__init__()
        self.model, self.i = model, i
    def forward(self, x):
        return self.model(x, head_idx=self.i)   # (h.sum(1) -> [B], h -> [B,1])
    

for i in range(N_HEAD):
    head_view = HeadView(model_gauss, i)
    sampler = ReplaySampler(model=head_view, img_shape=SZ_GAUS, buffer_size=600, noise_fraction=0.05, device=DEVICE)
    samples = sampler.generate(n_samples, steps=SAMPLE_STEPS, step_size=SAMPLE_STEP_SIZE, noise_std=0.005)
    samples = samples.squeeze(1)
    data = plot_gm_points(samples, torch.ones(n_samples, dtype=torch.long), centers=means, covs=covs, mode="raw",  cmap="gray", title=f"Generated Samples from Head {i}", save_path=f"{plot_path}/generated_samples_head_{i}.png")