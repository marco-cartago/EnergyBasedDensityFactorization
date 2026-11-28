"""Persistent (replay-buffer) Langevin sampler for energy-based models."""

import math
import warnings
from contextlib import contextmanager

import torch
import torch.nn as nn


class ReplaySampler:
    """PCD-style sampler with persistent chains living on ``device``.

    Refactor notes
    --------------
    * ``autograd.grad`` is called with ``x`` as the only input, so model
      parameters never accumulate gradients during sampling.  We therefore
      avoid toggling ``requires_grad`` on every parameter for every batch.
    * ``eval()`` is entered once per sampling call, not once per step.
    * Langevin now uses an **unclipped** gradient with noise matched to the
      step size (``noise_std = sqrt(2 * step_size * T)``, T=1 by default).
      Clipping the gradient to a fixed magnitude changes the stationary
      distribution away from ``p(x) ~ exp(-E(x))``: with the old
      ``grad_clip=0.03`` and ``noise_std=0.2`` the noise dominated the drift by
      ~7x and chains simply random-walked, producing uniform-looking samples.
    """

    def __init__(
        self,
        model: nn.Module,
        img_shape: tuple,
        buffer_size: int = 8192,
        noise_fraction: float = 0.05,
        device=None,
        amp_dtype: torch.dtype | None = None,
        eval_during_sampling: bool = True,
    ):
        self.model = model
        self.img_shape = tuple(img_shape)
        self.buffer_size = int(buffer_size)
        self.noise_fraction = float(noise_fraction)
        self.amp_dtype = amp_dtype
        self.eval_during_sampling = eval_during_sampling

        self.device = torch.device(
            device if device is not None else next(model.parameters()).device
        )
        self._bshape = (-1,) + (1,) * len(self.img_shape)
        self.buffer = self._noise(self.buffer_size)

    # ------------------------------------------------------------------ utils
    def _noise(self, n: int) -> torch.Tensor:
        """Uniform noise in [-1, 1] with the configured sample shape."""
        return torch.rand(n, *self.img_shape, device=self.device) * 2 - 1

    @contextmanager
    def _sampling_mode(self):
        if not self.eval_during_sampling:
            yield
            return
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(was_training)

    @staticmethod
    def _check_noise_scale(step_size, noise_std, grad_clip, normalize_grad):
        if (
            grad_clip is not None
            or normalize_grad
            or isinstance(step_size, torch.Tensor)
        ):
            return
        try:
            eta = float(step_size)
        except (TypeError, ValueError):
            return
        if eta <= 0 or noise_std <= 0:
            return
        ideal = math.sqrt(2.0 * eta)  # T = 1
        ratio = noise_std / ideal
        if ratio < 0.25:
            warnings.warn(
                f"Langevin noise_std={noise_std:g} << sqrt(2*step_size)={ideal:.3g}: "
                "chains will mode-find rather than sample.",
                RuntimeWarning,
                stacklevel=3,
            )
        elif ratio > 4.0:
            warnings.warn(
                f"Langevin noise_std={noise_std:g} >> sqrt(2*step_size)={ideal:.3g}: "
                "chains will diffuse and samples will look uniform.",
                RuntimeWarning,
                stacklevel=3,
            )

    # --------------------------------------------------------------- Langevin
    def langevin_step(
        self,
        x: torch.Tensor,
        step_size: float | torch.Tensor = 0.002,
        noise_std: float = 0.063,
        grad_clip: float | None = None,
        normalize_grad: bool = False,
    ) -> torch.Tensor:
        x = (x + noise_std * torch.randn_like(x)).clamp_(-1.0, 1.0)
        x = x.detach().requires_grad_(True)

        with torch.enable_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                energy = self.model(x)[0]
            (grad,) = torch.autograd.grad(energy.float().sum(), x)

        if normalize_grad:
            # Keeps the drift scale independent of the model's gradient
            # magnitude (useful for conv heads whose per-pixel grads are ~1e-5).
            grad = grad / (grad.square().mean().sqrt().clamp_min(1e-12))
        elif grad_clip is not None:
            grad = grad.clamp_(-grad_clip, grad_clip)

        return (x.detach() - step_size * grad).clamp_(-1.0, 1.0)

    def run_langevin(
        self,
        x: torch.Tensor,
        steps: int = 100,
        step_size: float = 0.002,
        noise_std: float = 0.063,
        grad_clip: float | None = None,
        normalize_grad: bool = False,
        step_size_sigma: float = 0.0,
    ) -> torch.Tensor:
        if steps <= 0:
            return x.detach()

        self._check_noise_scale(step_size, noise_std, grad_clip, normalize_grad)

        if step_size_sigma > 0.0:
            z = torch.randn(x.shape[0], device=x.device)
            eta = step_size * torch.exp(step_size_sigma * z).view(self._bshape)
        else:
            eta = step_size

        with self._sampling_mode():
            for _ in range(steps):
                x = self.langevin_step(
                    x,
                    step_size=eta,
                    noise_std=noise_std,
                    grad_clip=grad_clip,
                    normalize_grad=normalize_grad,
                )
        return x.detach()

    # ----------------------------------------------------------------- buffer
    def initialize_chains(self, batch_size: int):
        n = self.buffer.shape[0]
        if batch_size <= n:
            idx = torch.randperm(n, device=self.device)[:batch_size]
        else:
            idx = torch.randint(0, n, (batch_size,), device=self.device)

        x = self.buffer[idx]
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
        steps: int = 100,
        step_size: float = 0.002,
        noise_std: float = 0.063,
        grad_clip: float | None = None,
        normalize_grad: bool = False,
        step_size_sigma: float = 0.0,
        update_buffer: bool = True,
    ) -> torch.Tensor:
        x, idx = self.initialize_chains(batch_size)
        x = self.run_langevin(
            x,
            steps=steps,
            step_size=step_size,
            noise_std=noise_std,
            grad_clip=grad_clip,
            normalize_grad=normalize_grad,
            step_size_sigma=step_size_sigma,
        )
        if update_buffer:
            self.buffer[idx] = x.detach()
        return x

    def generate(
        self,
        n_samples: int,
        steps: int = 100,
        step_size: float = 0.002,
        noise_std: float = 0.063,
        grad_clip: float | None = None,
        normalize_grad: bool = False,
        step_size_sigma: float = 0.0,
        batch_size: int = 128,
    ) -> torch.Tensor:
        """Generate samples on CPU without touching the training buffer."""
        out = []
        remaining = int(n_samples)
        while remaining > 0:
            b = min(int(batch_size), remaining)
            x = self.sample(
                b,
                steps=steps,
                step_size=step_size,
                noise_std=noise_std,
                grad_clip=grad_clip,
                normalize_grad=normalize_grad,
                step_size_sigma=step_size_sigma,
                update_buffer=False,
            )
            out.append(x.detach().cpu())
            remaining -= b
        return torch.cat(out, dim=0)
