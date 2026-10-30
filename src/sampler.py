from contextlib import contextmanager

import torch
import torch.nn as nn


class ReplaySampler:

    """
    Persistent (replay-buffer) Langevin sampler for Energy-Based Models.

    The buffer is a preallocated tensor of shape [N, C, H, W] living on the
    sampling device. Each slot is one persistent chain: `sample` draws a set
    of slots, advances them with `steps` Langevin updates, and writes them
    back to the same slots (PCD-style). A fraction `noise_fraction` of the
    chains is re-initialised from uniform noise at every call.

    The model is assumed to return a tuple whose first element is the
    per-sample energy, shape [B] or [B, 1].

    Note: this is the short-run, clipped-gradient sampler used in
    Du & Mordatch (2019), not exact Langevin dynamics for exp(-E):
    the noise scale is decoupled from the step size. Use the same
    (steps, step_size, noise_std, grad_clip) at training and test time.
    """

    def __init__(
        self,
        model: nn.Module,
        img_shape: tuple[int, int, int],
        buffer_size: int = 8192,
        noise_fraction: float = 0.05,
        device=None,
        amp_dtype: torch.dtype | None = None,
    ):
        self.model = model
        self.img_shape = tuple(img_shape)
        self.buffer_size = buffer_size
        self.noise_fraction = noise_fraction
        self.amp_dtype = amp_dtype  # e.g. torch.bfloat16, or None to disable

        self.device = torch.device(
            device if device is not None else next(model.parameters()).device
        )

        # Shape that broadcasts a per-sample vector over [B, C, H, W]
        self._bshape = (-1,) + (1,) * len(self.img_shape)

        self.buffer = self._noise(buffer_size)

    # ------------------------------------------------------------------ utils
    def _noise(self, n: int) -> torch.Tensor:
        """Uniform noise in [-1, 1], shape [n, C, H, W], on the device."""
        return torch.rand(n, *self.img_shape, device=self.device) * 2 - 1

    @contextmanager
    def _frozen_model(self):
        """eval() mode and no parameter gradients; restored on exit."""
        was_training = self.model.training
        flags = [p.requires_grad for p in self.model.parameters()]
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, f in zip(self.model.parameters(), flags):
                p.requires_grad_(f)
            self.model.train(was_training)

    # --------------------------------------------------------------- Langevin
    def langevin_step(
        self,
        x: torch.Tensor,
        step_size: float | torch.Tensor = 10.0,
        noise_std: float = 0.005,
        grad_clip: float = 0.03,
    ) -> torch.Tensor:
        """
        One update: perturb, evaluate grad_x E at the perturbed point,
        clipped gradient step, clamp to the data range [-1, 1].

        `step_size` may be a float or a tensor broadcastable to x
        (e.g. per-chain step sizes of shape [B, 1, 1, 1]).
        """
        x = (x + noise_std * torch.randn_like(x)).clamp_(-1.0, 1.0)
        x.requires_grad_(True)

        with torch.enable_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                energy = self.model(x)[0]
                
            # Per-sample grads: valid because samples do not interact
            # (model in eval mode, no batch statistics).
            (grad,) = torch.autograd.grad(energy.float().sum(), x)

        grad = grad.clamp_(-grad_clip, grad_clip)
        return (x.detach() - step_size * grad).clamp_(-1.0, 1.0)

    def run_langevin(
        self,
        x: torch.Tensor,
        steps: int = 60,
        step_size: float = 10.0,
        noise_std: float = 0.005,
        grad_clip: float = 0.03,
        step_size_sigma: float = 0.0,
    ) -> torch.Tensor:
        """
        Refine `x` with `steps` Langevin updates.

        If `step_size_sigma` > 0, each chain gets its own step size
        eta_i = step_size * exp(sigma * z_i), z_i ~ N(0, 1)  (lognormal
        jitter, median = step_size), drawn once per call.
        """
        if step_size_sigma > 0.0:
            z = torch.randn(x.shape[0], device=x.device)
            eta = step_size * torch.exp(step_size_sigma * z).view(self._bshape)
        else:
            eta = step_size

        with self._frozen_model():
            for _ in range(steps):
                x = self.langevin_step(
                    x, step_size=eta, noise_std=noise_std, grad_clip=grad_clip
                )
        return x

    # ----------------------------------------------------------------- buffer
    def initialize_chains(self, batch_size: int):
        """
        Starting points: slots drawn from the buffer (without replacement
        when batch_size <= buffer_size), a fraction `noise_fraction` of
        which is replaced by fresh uniform noise.

        Returns (x, idx) where idx are the buffer slots that were drawn.
        """
        n = self.buffer.shape[0]
        if batch_size <= n:
            idx = torch.randperm(n, device=self.device)[:batch_size]
        else:
            idx = torch.randint(0, n, (batch_size,), device=self.device)

        x = self.buffer[idx]  # advanced indexing -> copy, no aliasing
        reset = torch.rand(batch_size, device=self.device) < self.noise_fraction
        x = torch.where(reset.view(self._bshape), self._noise(batch_size), x)
        return x, idx

    def reset_buffer(self):
        self.buffer = self._noise(self.buffer_size)

    def state_dict(self) -> dict:
        return {"buffer": self.buffer.detach().cpu()}

    def load_state_dict(self, state: dict):
        self.buffer = state["buffer"].to(self.device)
        self.buffer_size = self.buffer.shape[0]

    # --------------------------------------------------------------- sampling
    def sample(
        self,
        batch_size: int,
        steps: int = 60,
        step_size: float = 10.0,
        noise_std: float = 0.005,
        grad_clip: float = 0.03,
        step_size_sigma: float = 0.0,
        update_buffer: bool = True,
    ) -> torch.Tensor:
        """
        Generate negative samples (on the sampler's device).

        With `update_buffer=True` the refined chains are written back in
        place to the slots they came from.
        """
        x, idx = self.initialize_chains(batch_size)
        x = self.run_langevin(
            x,
            steps=steps,
            step_size=step_size,
            noise_std=noise_std,
            grad_clip=grad_clip,
            step_size_sigma=step_size_sigma,
        )
        if update_buffer:
            self.buffer[idx] = x
        return x

    def generate(
        self,
        n_samples: int,
        steps: int = 60,
        step_size: float = 10.0,
        noise_std: float = 0.005,
        grad_clip: float = 0.03,
        step_size_sigma: float = 0.0,
        batch_size: int = 128,
    ) -> torch.Tensor:
        """
        Generate an arbitrary number of samples (returned on CPU).
        Does not modify the training buffer.
        """
        out = []
        remaining = n_samples
        while remaining > 0:
            b = min(batch_size, remaining)
            x = self.sample(
                b,
                steps=steps,
                step_size=step_size,
                noise_std=noise_std,
                grad_clip=grad_clip,
                step_size_sigma=step_size_sigma,
                update_buffer=False,
            )
            out.append(x.cpu())
            remaining -= b

        return torch.cat(out, dim=0)