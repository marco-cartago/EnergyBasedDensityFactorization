import pickle
from time import time_ns

import torch
import torch.nn as nn

from src.sampler import ReplaySampler
from src.information import TotalCorrelationEstimator
from src.train import train_one_epoch_TC

def train_full_model(
        
        model: nn.Module, 
        n_heads: int,
        model_folder: str,
        img_shape: tuple,

        train_loader,

        n_epochs          = 30,
        ebm_learning_rate = 1e-3,

        sample_steps      = 50,
        sample_step_size  = 10.0,
        sample_buf_sz     = 8192,
        noise_frac        = 0.05,

        noise_std         = 0.005,
        energy_reg        = 1e-1,
        
        tc_regulariz      = 1e-2,
        tc_lr             = 1e-3,
        tc_hidden_dim     = 64,
        tc_buffer_size    = 8192,
        tc_batch_size     = 128,
        tc_n_steps        = 50,

        device: torch.device = torch.device("cpu"),
    ):

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=ebm_learning_rate, 
        # betas=(0.000, 0.999)
    )

    sampler = ReplaySampler(
        model, 
        img_shape      = img_shape, 
        buffer_size    = sample_buf_sz, 
        noise_fraction = noise_frac, 
        device         = device
    )
    
    total_correlation_estimator = TotalCorrelationEstimator(
        n_heads, 
        hidden_dim  = tc_hidden_dim, 
        lr          = tc_lr,
        buffer_size = tc_buffer_size,
        batch_size  = tc_batch_size,
    )
    total_correlation_estimator.to(device)

    traininfos = []

    for epoch in range(n_epochs+1):
        epoch_loss, traininfo = train_one_epoch_TC(
            model, 
            sampler, 
            train_loader, 
            optimizer,

            sample_steps       = sample_steps,
            sample_step_size   = sample_step_size,
            sample_noise_std   = noise_std,
            energy_reg         = energy_reg,
            tc_regularizations = tc_regulariz,
            tc_n_steps         = tc_n_steps,     
            
            tc_estimator       = total_correlation_estimator,
            device             = device
        )
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        traininfos.append(traininfo)

        if epoch % 5 == 0:
            model_file = (
                f"./models/{model_folder}/" + 
                f"EBM_{epoch}_epochs_{img_shape}_shape_{len(model.heads)}_heads_{time_ns()}"
            )
            traininf_f = model_file + ".tinfo"
            modelf = model_file + ".pth"

            with open(traininf_f, "b+w") as f:
                pickle.dump(traininfos, f)

            with open(modelf, "b+w") as f:
                torch.save(model.state_dict(), f)