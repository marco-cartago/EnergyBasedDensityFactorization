import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle
from time import time_ns

from src.data_import import *
from src.train import train_one_epoch_TC, train_one_epoch
from src.model import EBM
from src.sampler import ReplaySampler
from src.information import TotalCorrelationEstimator
from src.trainer import train_full_model

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

IM_SZ_FMN = (1, 28, 28)
IM_SZ_OLI = (1, 64, 64)
IM_SZ_LFW = (1, 128, 128)

N_HEAD = 3
N_EPOCH = 30

print(f"Currently using {DEVICE}")

oli_train_loader, oli_test_loader = load_olivetti(batch_size=BATCH_SIZE, shuffle=True)

model_oli = EBM(image_shape=IM_SZ_OLI, n_heads=N_HEAD).to(DEVICE)
train_full_model(
    model=model_oli,
    data_shape=IM_SZ_OLI,
    n_heads=N_HEAD,
    model_folder="olivetti",
    train_loader=oli_train_loader,
    n_epochs=N_EPOCH,
    img_shape=IM_SZ_OLI,
    device=DEVICE
)