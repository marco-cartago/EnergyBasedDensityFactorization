import numpy as np

# Hard 2d toy distributions


# Thin axis aligned gaussians.
def make_thin_gaussians(
    k=12, a=0.8, sigma_thin=(0.008, 0.02), sigma_long=(0.08, 0.35), seed=0
):
    """Random layout of k anisotropic Gaussians with centres in [-a, a]^2."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-a, a, size=(k, 2))
    vertical = rng.random(k) < 0.5
    thin = rng.uniform(*sigma_thin, size=k)
    long = rng.uniform(*sigma_long, size=k)
    # per-axis std: thin across the line, long along it
    std = np.where(
        vertical[:, None],
        np.stack([thin, long], 1),  # vertical line: thin in x
        np.stack([long, thin], 1),
    )  # horizontal line: thin in y
    w = rng.dirichlet(np.ones(k) * 2.0)
    return dict(mu=mu, std=std, w=w)


def sample_thin_gaussians(n, params=None, seed=None, **kw):
    p = params or make_thin_gaussians(**kw)
    rng = np.random.default_rng(seed)
    comp = rng.choice(len(p["w"]), size=n, p=p["w"])
    return p["mu"][comp] + p["std"][comp] * rng.standard_normal((n, 2))


def log_prob_thin_gaussians(x, params):
    """Exact log density of axis aligned gaussians."""
    x = np.asarray(x)[:, None, :]  # (n,1,2)
    z = (x - params["mu"][None]) / params["std"][None]  # (n,k,2)
    lp = (
        -0.5 * (z**2).sum(-1)
        - np.log(params["std"]).sum(-1)[None]
        - np.log(2 * np.pi)
        + np.log(params["w"])[None]
    )
    m = lp.max(1, keepdims=True)
    return (m + np.log(np.exp(lp - m).sum(1, keepdims=True)))[:, 0]


# Checkerboard
def sample_checkerboard(n, cells=4, seed=None):
    """Uniform on alternating squares of a cells x cells board in [-1,1]^2."""
    rng = np.random.default_rng(seed)
    out = np.empty((0, 2))
    while len(out) < n:
        x = rng.uniform(-1, 1, size=(2 * n, 2))
        idx = np.floor((x + 1) / 2 * cells).astype(int).clip(0, cells - 1)
        out = np.concatenate([out, x[(idx.sum(1) % 2) == 0]])
    return out[:n]


# Two-arm spiral
def sample_spiral(n, turns=2.0, noise=0.01, seed=None):
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, 2, n)
    t = np.sqrt(rng.uniform(0.02, 1, n))  # ~uniform arc length
    theta = t * turns * 2 * np.pi + arm * np.pi
    r = 0.95 * t
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)
    return x + noise * rng.standard_normal((n, 2))


# Concentric thin rings
def sample_rings(n, radii=(0.3, 0.6, 0.9), noise=0.01, seed=None):
    rng = np.random.default_rng(seed)
    r = rng.choice(radii, n) + noise * rng.standard_normal(n)
    th = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)
