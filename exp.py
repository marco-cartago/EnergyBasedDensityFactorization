import torch
from time import time_ns

from src.data_import import *
from src.model import *
from src.trainer import train_full_model

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

IM_SZ_2D = (1, 1, 2)

N_HEAD = 5
N_EPOCH = 30

print(f"Currently using {DEVICE}")

train_loader, test_loader = circle_g_peaks(
    n_peaks          = 4,
    points_per_gauss = 150,
    var              = 0.1,
    batch_size       = BATCH_SIZE,
    test_size        = 0.2,
    seed             = 0,
    shuffle          = True,
)

model_oli = SmallEBM2d(image_shape=IM_SZ_2D, n_heads=N_HEAD).to(DEVICE)

train_full_model(
    model        = model_oli,
    data_shape   = IM_SZ_2D,
    n_heads      = N_HEAD,
    model_folder ="2d",
    train_loader = train_loader,
    n_epochs     = N_EPOCH,
    img_shape    = IM_SZ_2D,
    device       = DEVICE
)