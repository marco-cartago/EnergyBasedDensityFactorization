import os

import torch

from src.data_import import *
from src.model import *
from src.trainer import train_full_model

from torchsummary import summary

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

IM_SZ_2D = (1, 1, 2)
IM_SZ_OLI = (1, 64, 64)
IM_SZ_DIG = (1, 8, 8)
IM_SZ_MNI = (1, 28, 28)

N_HEAD = 3
N_EPOCH = 150


print(f"CURRENTLY USING: {DEVICE}" + (" :) " if torch.cuda.is_available() else ""))

os.makedirs("./models/olivetti", exist_ok=True)
os.makedirs("./models/mnist", exist_ok=True)
os.makedirs("./models/fmnist", exist_ok=True)


train_loader, test_loader = load_olivetti(batch_size=BATCH_SIZE, shuffle=True)

model_oli = EBM(image_shape=IM_SZ_OLI, n_heads=N_HEAD, batch_size=BATCH_SIZE)

train_full_model(
    model=model_oli,
    tc_regulariz=1e-2,
    sample_step_size=1.0,
    data_shape=IM_SZ_OLI,
    n_heads=N_HEAD,
    model_folder="fmnist",
    train_loader=train_loader,
    n_epochs=N_EPOCH,
    device=DEVICE,
)
