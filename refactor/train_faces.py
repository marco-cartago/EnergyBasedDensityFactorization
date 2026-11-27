"""Train the factorized EBM on face images (LFW or Olivetti).

Examples (from the repository root):
    python refactor/train_faces.py --dataset lfw --epochs 30
    python refactor/train_faces.py --dataset olivetti --epochs 20 --no-tc

LFW is downloaded through scikit-learn on first use.  If no network is
available, use ``--dataset olivetti`` or pre-populate ``--data-home``.
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.data_import import load_lfw, load_olivetti  # noqa: E402
from src.model import EBM  # noqa: E402
from src.plotting import plot_traininfo, save_image_grid  # noqa: E402
from src.sampler import ReplaySampler  # noqa: E402
from src.trainer import train_full_model_TC, train_full_model_no_TC  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["lfw", "olivetti"], default="lfw")
    p.add_argument("--image-size", type=int, default=None,
                   help="default: 128 for LFW, 64 for Olivetti")
    p.add_argument("--data-home", default="./data")
    p.add_argument("--min-faces", type=int, default=20, help="LFW only")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap the number of images (useful for smoke tests)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-head", type=int, default=3)
    p.add_argument("--tc-reg", type=float, default=1e-2)
    p.add_argument("--tc-updates", type=int, default=5,
                   help="critic steps per model batch")
    p.add_argument("--energy-reg", type=float, default=1e-2)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--sample-step-size", type=float, default=0.01)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--normalize-grad", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="normalize the energy gradient so the drift scale is "
                        "independent of conv-head magnitude (needed for images)")
    p.add_argument("--buffer-size", type=int, default=256)
    p.add_argument("--gen-samples", type=int, default=16)
    p.add_argument("--gen-steps", type=int, default=200)
    p.add_argument("--no-tc", action="store_true")
    p.add_argument("--plot-batch", action="store_true",
                   help="also plot per-batch loss curves (as in ablation_gaussians)")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="./refactor/models/faces")
    return p.parse_args()


def load_data(args):
    if args.dataset == "lfw":
        return load_lfw(
            batch_size=args.batch_size,
            shuffle=True,
            image_size=args.image_size or 128,
            data_home=args.data_home,
            min_faces_per_person=args.min_faces,
            max_samples=args.max_samples,
            normalize=True,
        )
    train_loader, test_loader = load_olivetti(
        batch_size=args.batch_size,
        shuffle=True,
        data_home=args.data_home,
        normalize=True,
    )
    if args.image_size is not None:
        for loader in (train_loader, test_loader):
            ds = loader.dataset
            ds.tensors = (
                F.interpolate(ds.tensors[0], size=(args.image_size, args.image_size),
                              mode="bilinear", align_corners=False),
                ds.tensors[1],
            )
    return train_loader, test_loader


def main():
    args = parse_args()
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out, f"{args.dataset}_{time_stamp}")
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(0)
    print(f"CURRENTLY USING: {str(device).capitalize()}")

    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    try:
        train_loader, test_loader = load_data(args)
    except Exception as exc:  # dataset download can fail offline
        raise SystemExit(
            f"Could not load {args.dataset}: {exc}\n"
            "Try --dataset olivetti, or pre-download the dataset into "
            f"{args.data_home!r}."
        ) from exc

    sample = next(iter(train_loader))[0]
    img_shape = tuple(sample.shape[1:])
    print(
        f"Dataset={args.dataset}, shape={img_shape}, "
        f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    model = EBM(image_shape=img_shape, n_heads=args.n_head).to(device)

    common = dict(
        model=model,
        data_shape=img_shape,
        n_heads=args.n_head,
        model_folder=os.path.basename(run_dir),
        train_loader=train_loader,
        n_epochs=args.epochs,
        ebm_learning_rate=args.lr,
        energy_reg=args.energy_reg,
        sample_steps=args.sample_steps,
        sample_step_size=args.sample_step_size,
        sample_buf_sz=args.buffer_size,
        noise_std=args.noise_std,
        normalize_grad=args.normalize_grad,
        output_root=args.out,
        device=device,
        verbose=True,
    )
    if args.no_tc:
        tinfo, sampler = train_full_model_no_TC(**common)
    else:
        tinfo, sampler, _ = train_full_model_TC(
            **common, tc_regulariz=args.tc_reg, tc_buffer_size=args.buffer_size,
            tc_updates_per_step=args.tc_updates
        )

    plot_traininfo(
        tinfo, smooth=5, plot_batch=args.plot_batch,
        save_path=os.path.join(plot_dir, "train_info.png"), show=False,
    )
    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pth"))

    print("EVALUATING MODEL")
    save_image_grid(
        sample[: args.gen_samples], os.path.join(plot_dir, "real_faces.png"),
        n=args.gen_samples,
    )

    # A head-only view exposes a single factor's energy to the sampler:
    # E_i(x) = model(x, head_idx=i)[0].  Mirrors the gaussian script.
    class HeadView(nn.Module):
        def __init__(self, model, i):
            super().__init__()
            self.model, self.i = model, i

        def forward(self, x):
            return self.model(x, head_idx=self.i)

    gen_kwargs = dict(
        steps=args.gen_steps,
        step_size=args.sample_step_size,
        noise_std=args.noise_std,
        normalize_grad=args.normalize_grad,
        batch_size=min(32, args.gen_samples),
    )

    # Sample each head in isolation first, then the combined model.
    for i in range(args.n_head):
        head_sampler = ReplaySampler(
            HeadView(model, i), img_shape=img_shape,
            buffer_size=args.buffer_size, noise_fraction=0.05, device=device,
        )
        head_samples = head_sampler.generate(args.gen_samples, **gen_kwargs)
        save_image_grid(
            head_samples,
            os.path.join(plot_dir, f"generated_faces_head_{i}.png"),
            n=args.gen_samples,
        )

    generated = sampler.generate(args.gen_samples, **gen_kwargs)
    save_image_grid(
        generated, os.path.join(plot_dir, "generated_faces_combination.png"),
        n=args.gen_samples,
    )
    # Original filename kept for compatibility with existing outputs.
    save_image_grid(
        generated, os.path.join(plot_dir, "generated_faces.png"),
        n=args.gen_samples,
    )
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
