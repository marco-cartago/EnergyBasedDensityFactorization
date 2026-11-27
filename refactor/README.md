# Refactored EBM factorization pipeline

Self-contained copy of the pipeline under `refactor/`. The original files in the
repository root and `src/` are untouched. Nothing is committed to git.

## Run

From the repository root:

```bash
# Gaussian mixtures (same workflow as the original ablation_gaussians.py)
python refactor/ablation_gaussians.py                 # full 1000 epochs
python refactor/ablation_gaussians.py --epochs 20 --sample-steps 5 --gen-steps 50

# Faces (LFW downloaded on first use by scikit-learn)
python refactor/train_faces.py --dataset lfw --epochs 30
python refactor/train_faces.py --dataset olivetti --epochs 20 --no-tc
```

`--out` controls where checkpoints, logs and plots are written (defaults are
`./refactor/ablations/gauss` and `./refactor/models/faces`).

## Shared loss / evaluation workflow

Both scripts use the same loss workflow: `plot_traininfo` writes
`train_info.png` with the epoch-level loss components (CD, energy regularization,
TC) and real-vs-fake energies, and `--plot-batch` additionally writes
`train_info_batch.png` with the per-batch curves. The same breakdown is printed
per epoch during training.

Face evaluation mirrors `ablation_gaussians.py`: it samples **each head in
isolation first** via a `HeadView` that exposes only `E_i(x) = model(x,
head_idx=i)` (`generated_faces_head_*.png`), **then the combined model**
(`generated_faces_combination.png`, also kept as `generated_faces.png` for
compatibility). All sampling uses the same Langevin settings as training, with
`--normalize-grad` enabled by default for conv heads.

## What changed relative to the original

| File | Fix |
| --- | --- |
| `src/trainer.py` | `LinearLR` uses relative factors (the old effective LR was ~1e-6 instead of 1e-3); Adam β₁ restored to 0.9; scheduler stepped once per epoch; output directories created automatically; trainers also return the sampler/critic. |
| `src/train.py` | Model set to `train()`; TC critic frozen while its bound contributes to the loss, then updated 5×/batch on freshly recomputed (detached) heads; scheduler double-step removed; MINE code dropped; batch metrics always populated. |
| `src/information.py` | The TC estimator now uses a proper Donsker–Varadhan bound with independently permuted columns as negatives instead of a same-sample Jensen gap; `train_steps` tolerance loop fixed; buffer indexing and `.to(device)` fixed; self-contained (no `MINE.py`). |
| `src/losses.py` | Weaker default energy L2 term (0.01 instead of 0.5, which drove energies to constant ±1). |
| `src/model.py` | MLP heads are spectral-normalized so energies stay bounded without a large energy penalty; gaussian input shape is consistently `(2,)`. |
| `src/sampler.py` | Langevin no longer toggles `requires_grad` on all parameters every batch (`autograd.grad` with `x` as sole input does not accumulate parameter gradients); `eval()` entered once per call; **gradient clipping removed and noise matched to the step size** (`step=0.002`, `noise=0.063`, T=1) so chains actually sample `exp(-E)`; warnings for mismatched noise. |
| `src/data_import.py` | LFW/Olivetti images normalized to `[-1, 1]` to match the sampler's clamp range; `data_home`, `max_samples` and stratified-split guards added. |
| `src/plotting.py` | Shared plotting; batch metrics no longer break on ragged per-epoch lists. |
| `train_faces.py` | Per-head sampling evaluation followed by combined-model sampling; `--plot-batch` support; same loss-component logging as the gaussian script. |
| scripts | CLI overrides for quick smoke runs; gaussian data shape `(2,)` end-to-end; artifacts kept under `refactor/`. |

The overall workflow (PCD/Langevin sampling, CD + energy reg + TC penalty,
epoch scheduling, per-head sampling) is unchanged.

## Why training collapsed and samples looked uniform (and the fix)

Both symptoms came from the Langevin sampler, not from TC or the energy L2 term.

**1. Flat/zero energy.** The old `noise_std=0.005` was ~40× smaller than the
clipped drift (`grad_clip=0.03`), so Langevin performed near-deterministic mode
finding instead of MCMC sampling. CD negatives already had *lower* energy than
the data from epoch 0 (`E_fake < E_real`), so CD flattened the energy and the L2
term pulled the flat offset to zero. With fixed negatives the same optimizer
creates a gap of **+8.1**; with the zero-temperature sampler it starts at
**−0.11**.

**2. Uniform-looking samples.** Raising `noise_std` to `0.2` fixed training but
introduced the opposite failure at generation time. Hard-clipping the gradient
to a fixed magnitude means the drift is **not** `∇E` of any potential, so the
chain's stationary distribution is not `p(x) ∝ exp(-E(x))`. With drift 0.03 and
noise 0.2 the noise won by ~7× and chains just diffused across the box: only
16% of generated 2-D points landed near a data mode (uniform noise: 11%).

**Final fix:** remove gradient clipping and match noise to the step size,
`noise_std = sqrt(2 * step_size * T)` with `T=1`. Defaults are
`step_size=0.002`, `noise_std=0.063`, 100 steps. This is proper Langevin, so
the same settings work for both training and generation. For image models,
whose conv heads produce per-pixel gradients of order 1e-5, use
`--normalize-grad` (enabled by default in `train_faces.py`) to keep the drift
scale independent of gradient magnitude.

Results after 250 epochs with the script defaults: learned energy gap between
uniform noise and data **+3.96**; generated samples **72%** within 0.15 of a
data mode (median distance 0.099, versus 0.073 for real data and 0.370 for
uniform noise). Previously this was 16% / 0.320.

`src/sampler.py` warns when `noise_std` is far from `sqrt(2*step_size)` so a
mismatched configuration cannot silently reintroduce either failure.

The workflow itself is unchanged (PCD/Langevin, CD + energy reg + TC, epoch
scheduling, per-head sampling). Monitor `e_e_real`, `e_e_fake`, `e_corr` and the
generated-sample plots in `plots/`.
