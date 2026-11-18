import torch
import torch.nn as nn


def _resolve_img_shape(model, img_shape=None, img_dimension=None):
    if img_shape is not None:
        return tuple(img_shape)

    if img_dimension is not None:
        return (1, img_dimension, img_dimension)

    for attr in ("input_shape", "in_dim", "img_shape"):
        if hasattr(model, attr):
            return tuple(getattr(model, attr))

    return (1, 28, 28)


def _synthetize(
    model,
    energy_fn,
    n_images,
    steps,
    step_size,
    noise_std,
    init,
    clamp,
    device,
    img_shape,
):
    was_training = model.training
    model.eval()
    grad_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        if isinstance(init, torch.Tensor):
            x = init.clone().to(device)
        elif init == "uniform":
            x = torch.empty(n_images, *img_shape, device=device).uniform_(*clamp)
        elif init == "gaussian":
            x = torch.randn(n_images, *img_shape, device=device).clamp(*clamp)
        else:
            raise ValueError(f"init non valido: {init}")

        energies = []

        for _ in range(steps):
            x = x.detach().requires_grad_(True)

            energy = energy_fn(x).reshape(-1)
            energy.sum().backward()

            with torch.no_grad():
                grad = x.grad
                x = x - step_size * grad
                if noise_std > 0:
                    x = x + noise_std * torch.randn_like(x)
                x = x.clamp(*clamp)

            energies.append(energy.mean().item())
    finally:
        for p, flag in zip(model.parameters(), grad_flags):
            p.requires_grad_(flag)
        model.train(was_training)

    return x.detach(), energies


def synthetize_image(
    model: nn.Module,
    n_images=1,
    steps=500,
    step_size=10.0,
    noise_std=0.005,
    init="uniform",
    clamp=(-1.0, 1.0),
    device="cpu",
    img_dimension=None,
    img_shape=None,
    head_idx=None,
):
    """
    Finds images with low energy from a modeled distribution.

    If `head_idx` is given, only that head of `model` is optimized. Otherwise
    the full model energy is used.

    Return:
        images: tensor [n_images, *img_shape] with generated imgs
        energies: mean energies for every step
        init: either "uniform" or "gaussian" on the initial tensor
    """
    img_shape = _resolve_img_shape(model, img_shape, img_dimension)

    if head_idx is None:
        energy_fn = lambda x: model(x)[0]
    else:
        energy_fn = lambda x: model(x, head_idx=head_idx)[0]

    return _synthetize(
        model,
        energy_fn,
        n_images=n_images,
        steps=steps,
        step_size=step_size,
        noise_std=noise_std,
        init=init,
        clamp=clamp,
        device=device,
        img_shape=img_shape,
    )


def synthetize_image_from_head(
    model_head: nn.Module,
    n_images=1,
    steps=500,
    step_size=10.0,
    noise_std=0.005,
    init="uniform",
    clamp=(-1.0, 1.0),
    device="cpu",
    img_dimension=None,
    img_shape=None,
):
    """
    Finds images with low energy from a standalone head module.

    Kept for backwards compatibility; prefer `synthetize_image(model,
    head_idx=i)` so models without a `.heads` attribute are supported too.

    Return:
        images: tensor [n_images, *img_shape] with generated imgs
        energies: mean energies for every step
        init: either "uniform" or "gaussian" on the initial tensor
    """
    img_shape = _resolve_img_shape(model_head, img_shape, img_dimension)

    return _synthetize(
        model_head,
        lambda x: model_head(x),
        n_images=n_images,
        steps=steps,
        step_size=step_size,
        noise_std=noise_std,
        init=init,
        clamp=clamp,
        device=device,
        img_shape=img_shape,
    )
