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
N_SAMPLES = 500
N_CLASSES = 2
STD_RANGE = (0.03, 0.09)

# Training
BATCH_SIZE = 32
N_EPOCH = 1_000
TRAIN_PER = 0.8
SHUFFLE = False

# Model
N_HEAD = 2
HID_DIM_SIZE = 32
NETWORK_DEPTH = 2
TC_REG_STR = 0.0
ENERGY_REGULARIZ = 0.5
SAMPLE_STEP_SIZE = 1e-3
LR = 1e-3
SAMPLE_STEPS = 100

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
X, y, means, covs = mog_circle_dataset(
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

train_full_model_no_TC(
    model=model_gauss,
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

print("TRAINING MODEL")

# train_full_model_TC(
#     model=model_gauss,
#     tc_regulariz=TC_REG_STR,
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

model_file = "./ablations/gauss/" + f"{time_stamp}/" + f"_shape_{model_gauss.n_heads}_heads_{time_stamp}.pth"
with open(model_file, "wb") as f:
    torch.save(model_gauss.state_dict(), f)

gen = plot_gm_points(Xtrain, ytrain, centers=means, covs=covs, title="Training Data", save_path=f"{plot_path}/train_data.png")

print("EVALUATING MODEL")

n_samples = 1250

sampler = ReplaySampler(model=model_gauss, img_shape=SZ_GAUS, buffer_size=600, noise_fraction=0.05, device=DEVICE)

samples = sampler.generate(n_samples, steps=1050, step_size=0.01, noise_std=0.005)
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
    samples = sampler.generate(n_samples, steps=1050, step_size=0.01, noise_std=0.005)
    samples = samples.squeeze(1)
    data = plot_gm_points(samples, torch.ones(n_samples, dtype=torch.long), centers=means, covs=covs, mode="raw",  cmap="gray", title=f"Generated Samples from Head {i}", save_path=f"{plot_path}/generated_samples_head_{i}.png")