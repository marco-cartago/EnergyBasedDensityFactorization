import os

import torch
import torch.utils.data as D

from src.twoddensity import mog_dataset
from src.model import EBM2
from src.trainer import train_full_model

# Parameters
# -------------------------------------------------------------------------------------- #

# The seed (duh)
SEED = 0

# Data
SZ_GAUS = (1, 2)
N_SAMPLES = 1500
N_CLASSES = 5

# Training
BATCH_SIZE = 64
N_EPOCH = 10_000
TRAIN_PER = 0.8
SHUFFLE = False

# Model
N_HEAD = 4
HID_DIM_SIZE = 32
NETWORK_DEPTH = 2
TC_REG_STR = 1e-3
SAMPLE_STEP_SIZE = 0.1
LR = 1e-4
SAMPLE_STEPS = 5


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


print(f"CREATING DATASET", end="")
data_train, data_test = D.TensorDataset(Xtrain, ytrain), D.TensorDataset(Xtest, ytest)
train_dataloader = D.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = D.DataLoader(data_test, batch_size=BATCH_SIZE)
print(" [✓]")

print("CREATING AND COMPILING MODEL", end="")
model_gauss = EBM2(n_head=N_HEAD, hid_dim=HID_DIM_SIZE, depth=NETWORK_DEPTH)
model_gauss.to(DEVICE)
model_gauss.compile()
print(" [✓]")

# Model is trained (and saved?) along the (long?) way
# -------------------------------------------------------------------------------------- #

train_full_model(
    model=model_gauss,
    tc_regulariz=TC_REG_STR,
    sample_step_size=SAMPLE_STEP_SIZE,
    ebm_learning_rate=LR,
    sample_steps=SAMPLE_STEPS,
    data_shape=SZ_GAUS,
    n_heads=N_HEAD,
    model_folder="gauss",
    train_loader=train_dataloader,
    n_epochs=N_EPOCH,
    device=DEVICE,
    verbose=False,
)
