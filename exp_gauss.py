import os

import torch
import torch.utils.data as D

from src.twoddensity import mog_dataset
from src.model import EBM2
from src.trainer import train_full_model

from icecream import ic

# Parameters
# -------------------------------------------------------------------------------------- #

# The seed (duh)
SEED = 0

# Data
SZ_GAUS = (1, 2)
N_SAMPLES = 10000
N_CLASSES = 3

# Training
BATCH_SIZE = 32
SHUFFLE = False

# Model
N_HEAD = 3
N_EPOCH = 150
TRAIN_PER = 0.8

# Network stuff
# -------------------------------------------------------------------------------------- #
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"CURRENTLY USING: {str(DEVICE).capitalize()}"
    + (" :) " if torch.cuda.is_available() else "")
)

os.makedirs("./models/gauss", exist_ok=True)

print(f"GENERATED {N_SAMPLES} FROM {N_CLASSES} GAUSSIANS.", end="")
X, y, _, _ = mog_dataset(
    n_samples=N_SAMPLES, n_classes=N_CLASSES, std_range=(0.03, 0.09), seed=SEED
)
print(" [✓]")

sep = int(N_SAMPLES * TRAIN_PER)
Xtrain, Xtest, ytrain, ytest = X[:sep], X[sep:], y[:sep], y[sep:]

ic(X.shape, y.shape)

print(f"CREATING DATASET", end="")
data_train, data_test = D.TensorDataset(Xtrain, ytrain), D.TensorDataset(Xtest, ytest)
train_dataloader = D.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = D.DataLoader(data_test, batch_size=BATCH_SIZE)
print(" [✓]")

print("CREATING AND COMPILING MODEL", end="")
model_gauss = EBM2(n_head=N_HEAD, hid_dim=8, depth=2)
model_gauss.to(DEVICE)
model_gauss.compile()
print(" [✓]")

# Model is trained (and saved?) along the (long?) way
# -------------------------------------------------------------------------------------- #

train_full_model(
    model=model_gauss,
    tc_regulariz=1e-2,
    sample_step_size=1.0,
    data_shape=SZ_GAUS,
    n_heads=N_HEAD,
    model_folder="gauss",
    train_loader=train_dataloader,
    n_epochs=N_EPOCH,
    device=DEVICE,
)
