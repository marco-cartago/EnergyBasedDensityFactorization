import random

import numpy as np
import torch
import torch.nn as nn


class ReplaySampler:
    """
    Persistent Langevin sampler for Energy-Based Models.
    """

    def __init__(
        self,
        model: nn.Module,
        img_shape: tuple[int],
        buffer_size=8192,
        noise_fraction=0.05,
        device=None,
    ):
        self.model = model
        self.img_shape = img_shape
        self.buffer_size = buffer_size
        self.noise_fraction = noise_fraction

        self.device = device if device is not None else next(model.parameters()).device

        # Replay buffer: tensors with shape [C,H,W]
        self.buffer = []

    def langevin_step(self, x, step_size=1.0, noise_std=0.005, grad_clip=0.03):
        x = x.detach()
        noise = torch.randn_like(x) * noise_std
        x = (x + noise).clamp(-1.0, 1.0)

        with torch.enable_grad():
            x.requires_grad_(True)
            energy = self.model(x)[0].sum()
            grad = torch.autograd.grad(energy, x, only_inputs=True)[0]

        grad = grad.detach().clamp(-grad_clip, grad_clip)
        x = (x.detach() - step_size * grad).clamp(-1.0, 1.0)
        return x

    def run_langevin(
        self,
        x: torch.Tensor,
        steps=60,
        step_size=10.0,
        noise_std=0.005,
    ):
        """
        Refine samples through Langevin Dynamics.
        """

        # Disable parameter gradients
        was_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Langevin dynamics
        for _ in range(steps):
            x = self.langevin_step(x, step_size=step_size, noise_std=noise_std)

        # Restore model state
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.model.train(was_training)
        return x

    def initialize_chains(
        self,
        batch_size: int,
    ):
        """
        Create starting points for Langevin.

        Most samples come from replay buffer,
        a few are pure noise.
        """
        n_noise = np.random.binomial(batch_size, self.noise_fraction)
        n_buffer = batch_size - n_noise
        samples = []

        # Replay buffer samples
        if len(self.buffer) > 0 and n_buffer > 0:
            idx = np.random.choice(len(self.buffer), size=n_buffer, replace=True)
            old_samples = torch.stack([self.buffer[i] for i in idx], dim=0)
            samples.append(old_samples)

        # Fresh noise
        if n_noise > 0:
            noise_samples = (torch.rand(n_noise, *self.img_shape,)* 2- 1)
            samples.append(noise_samples)

        # On the first iteration buffer is empty
        if len(samples) == 0:
            samples.append(torch.rand(batch_size, *self.img_shape,)* 2- 1)

        x = torch.cat(samples, dim=0)

        return x.to(self.device)

    def update_buffer(
        self,
        samples: torch.Tensor,
        only_recent=True
    ):
        """
        Store new MCMC samples.
        
        If `only_recent` is true only the most recent samples
        produced by the chain get added to the buffer. 

        Otherwise the newly generated randomly replace the old samples. 
        """
        # Keep only newest samples

        samples = samples.detach().cpu()
        self.buffer.extend(list(samples))

        if len(self.buffer) > self.buffer_size:
            if only_recent:
                self.buffer = self.buffer[-self.buffer_size :]

            else:
                random.shuffle(self.buffer)
                self.buffer = self.buffer[-self.buffer_size :]


    def sample(
        self,
        batch_size: int,
        steps=60,
        step_size=10.0,
        noise_std=0.005,
    ):
        """
        Generate negative samples.
        """

        # Choose initial states
        x = self.initialize_chains(
            batch_size,
        )

        # Refine through Langevin Dynamics and save
        x = self.run_langevin(x, steps=steps, step_size=step_size, noise_std=noise_std)
        self.update_buffer(x,)

        return x

    def generate(
        self,
        n_samples: int,
        steps=60,
        step_size=10.0,
        noise_std=0.005,
        batch_size=128,
    ):
        """
        Generate arbitrary number of samples.
        """
        generated = []
        while len(generated) * batch_size < n_samples:
            x = self.sample(
                batch_size=batch_size,
                steps=steps,
                step_size=step_size,
                noise_std=noise_std,
            )
            generated.append(x.detach().cpu())

        generated = torch.cat(generated, dim=0)
        return generated[:n_samples]

    def reset_buffer(self):
        self.buffer = []
