"""Refactored gaussian-mixture ablation pipeline.

Run from the repository root:
    python refactor/ablation_gaussians.py --epochs 20

All artifacts stay under ``--out`` (default ``./refactor/ablations/gauss``).
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.utils.data as D
from torch import nn

# Use the refactored src/ package, not the original one.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.model import EBM2  # noqa: E402
from src.plotting import plot_traininfo  # noqa: E402
from src.sampler import ReplaySampler  # noqa: E402
from src.trainer import train_full_model_TC, train_full_model_no_TC  # noqa: E402
from src.twoddensity import (  # noqa: E402
    mog_circle_dataset,
    mog_dataset,
    plot_gm_points,
    plot_model_density,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-classes", type=int, default=6)
    p.add_argument("--std-min", type=float, default=0.05)
    p.add_argument("--std-max", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--train-per", type=float, default=0.8)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--hid-dim", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--tc-reg", type=float, default=1e-2)
    p.add_argument("--tc-updates", type=int, default=5,
                   help="critic steps per model batch")
    p.add_argument("--energy-reg", type=float, default=1e-2)
    p.add_argument("--sample-step-size", type=float, default=0.002,
                   help="Langevin drift step; noise is matched as sqrt(2*step)")
    p.add_argument("--noise-std", type=float, default=0.063,
                   help="Langevin noise; sqrt(2*step_size) gives T=1")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--gen-steps", type=int, default=300)
    p.add_argument("--gen-samples", type=int, default=1250)
    p.add_argument("--no-tc", action="store_true")
    p.add_argument("--plot-batch", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="./refactor/ablations/gauss")
    return p.parse_args()


def main():
    args = parse_args()
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out, time_stamp)
    plot_path = os.path.join(run_dir, "plots")
    os.makedirs(plot_path, exist_ok=True)

    # The dataset is 2-D: keep a single consistent shape everywhere.
    sz_gaus = (2,)
    std_range = (args.std_min, args.std_max)

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"CURRENTLY USING: {str(device).capitalize()}")

    params = vars(args) | {"data_shape": sz_gaus, "std_range": std_range}
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

    print(f"GENERATED {args.n_samples} FROM {args.n_classes} GAUSSIANS.", end="")
    X, y, means, covs = mog_circle_dataset(
        n_samples=args.n_samples,
        n_classes=args.n_classes,
        std_range=std_range,
        seed=args.seed,
    )
    print(" [✓]")

    sep = int(args.n_samples * args.train_per)
    Xtrain, Xtest, ytrain, ytest = X[:sep], X[sep:], y[:sep], y[sep:]

    data_train = D.TensorDataset(Xtrain, ytrain)
    data_test = D.TensorDataset(Xtest, ytest)
    train_dataloader = D.DataLoader(
        data_train, batch_size=args.batch_size, shuffle=not args.no_shuffle
    )
    test_dataloader = D.DataLoader(data_test, batch_size=args.batch_size)
    print(f"train={len(data_train)}, test={len(data_test)}")

    model_gauss = EBM2(
        n_head=args.n_head,
        hid_dim=args.hid_dim,
        depth=args.depth,
        input_shape=sz_gaus,
    ).to(device)

    common = dict(
        model=model_gauss,
        data_shape=sz_gaus,
        n_heads=args.n_head,
        model_folder=time_stamp,
        train_loader=train_dataloader,
        n_epochs=args.epochs,
        ebm_learning_rate=args.lr,
        energy_reg=args.energy_reg,
        sample_steps=args.sample_steps,
        sample_step_size=args.sample_step_size,
        noise_std=args.noise_std,
        output_root=args.out,
        device=device,
        verbose=True,
    )
    if args.no_tc:
        tinfo, sampler, *_ = train_full_model_no_TC(**common)
    else:
        tinfo, sampler, _ = train_full_model_TC(
            **common, tc_regulariz=args.tc_reg, tc_buffer_size=256,
            tc_updates_per_step=args.tc_updates
        )

    model_file = os.path.join(
        run_dir, f"_shape_{model_gauss.n_heads}_heads_{time_stamp}.pth"
    )
    torch.save(model_gauss.state_dict(), model_file)

    plot_gm_points(
        Xtrain, ytrain, centers=means, covs=covs,
        title="Training Data", save_path=f"{plot_path}/train_data.png",
    )
    plot_traininfo(
        tinfo, smooth=5, plot_batch=args.plot_batch,
        save_path=f"{plot_path}/train_info.png", show=False,
    )

    print("EVALUATING MODEL")
    samples = sampler.generate(
        args.gen_samples, steps=args.gen_steps,
        step_size=args.sample_step_size, noise_std=args.noise_std,
    ).reshape(-1, 2)

    plot_gm_points(
        samples, torch.ones(len(samples), dtype=torch.long),
        model=model_gauss, centers=means, covs=covs, mode="raw",
        cmap="gray", save_path=f"{plot_path}/generated_samples.png",
    )
    plot_gm_points(
        Xtrain, ytrain, model=model_gauss, centers=means, covs=covs,
        mode="raw", cmap="gray", title="Eval on Training Data",
        save_path=f"{plot_path}/eval_on_train_data.png",
    )
    plot_model_density(
        model_gauss, mode="log_density",
        save_path=f"{plot_path}/model_density.png",
    )

    class HeadView(nn.Module):
        def __init__(self, model, i):
            super().__init__()
            self.model, self.i = model, i

        def forward(self, x):
            return self.model(x, head_idx=self.i)

    for i in range(args.n_head):
        head_view = HeadView(model_gauss, i)
        head_sampler = ReplaySampler(
            head_view, img_shape=sz_gaus, buffer_size=600,
            noise_fraction=0.05, device=device,
        )
        head_samples = head_sampler.generate(
            args.gen_samples, steps=args.gen_steps,
            step_size=args.sample_step_size, noise_std=args.noise_std,
        ).reshape(-1, 2)
        plot_gm_points(
            head_samples, torch.ones(len(head_samples), dtype=torch.long),
            centers=means, covs=covs, mode="raw", cmap="gray",
            title=f"Generated Samples from Head {i}",
            save_path=f"{plot_path}/generated_samples_head_{i}.png",
        )

    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
