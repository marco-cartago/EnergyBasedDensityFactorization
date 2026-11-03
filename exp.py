import torch
from time import time_ns

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

N_HEAD = 3
N_EPOCH = 50

print(f"Currently using {DEVICE}")

# train_loader, test_loader = circle_g_peaks(
#     n_peaks          = 4,
#     points_per_gauss = 150,
#     var              = 0.1,
#     batch_size       = BATCH_SIZE,
#     test_size        = 0.2,
#     seed             = 0,
#     shuffle          = True,
# )

train_loader, test_loader = load_olivetti(
    batch_size       = BATCH_SIZE,
    shuffle          = True,
)


# model_2d = SmallEBM2d(image_shape=IM_SZ_2D, n_heads=N_HEAD).to(DEVICE)
model_oli = EBM(image_shape=IM_SZ_OLI, n_heads=3, batch_size=BATCH_SIZE)
summary(model_oli, IM_SZ_OLI)

train_full_model(
    model             = model_oli,
    tc_regulariz      = 0.0,
    data_shape        = IM_SZ_OLI,
    n_heads           = N_HEAD,
    model_folder      = "olivetti",
    train_loader      = train_loader,
    n_epochs          = N_EPOCH,
    device            = DEVICE
)