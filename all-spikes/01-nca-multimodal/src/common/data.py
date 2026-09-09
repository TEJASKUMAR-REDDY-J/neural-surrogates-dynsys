"""Eight datasets, five modalities, one shape.

The point of this spike is to see whether one architecture family behaves the same way on
data with very different *geometry*. So everything is forced into a single container:

    a lattice of cells, each holding a feature vector

    shape (n_samples, channels, *spatial)

    spatial = (64,)    a line of 64 cells        - signals, text, 1-D chaos
    spatial = (8, 8)   a grid                    - images
    spatial = (30,)    a line with no real order - tabular features

That last one matters. A tabular row is a "lattice" whose neighbours mean nothing. If the
local-only architecture does badly there and the global-context ones do fine, that is the
clearest possible evidence that the global branch is doing what it was added for.

Everything here is generated or bundled with numpy/scipy/sklearn. Nothing downloads except
scipy's `ascent`, which caches after the first call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]


@dataclass
class Dataset:
    name: str
    modality: str          # chaos | ca | image | text | tabular | signal
    X: np.ndarray          # (n, C, *spatial), already normalised to roughly [0, 1]
    spatial: tuple         # the lattice shape
    channels: int
    ordered: bool          # do neighbouring cells actually mean anything?
    temporal: bool         # is the lattice axis time, so masking the tail = forecasting?
    note: str = ""
    n_duplicates_removed: int = 0

    def __post_init__(self):
        # torch is float32 everywhere; a float64 array silently breaks the backward pass
        self.X = np.ascontiguousarray(self.X, dtype=np.float32)

    def __repr__(self) -> str:
        return (f"<{self.name} {self.modality} n={len(self.X)} "
                f"lattice={self.spatial} C={self.channels}>")


def _norm01(a: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] using robust percentiles, so one outlier cannot flatten everything."""
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max()) or 1.0
    return np.clip((a - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def dedup(X: np.ndarray) -> np.ndarray:
    """Drop exactly repeated examples, keeping the first occurrence and the original order.

    This exists because of a real failure. Two datasets were built by repeating a finite
    pool of rows to reach the requested count - the clinical features are tiled from 569
    rows to 900, and rule 110 settles into only 337 distinct states out of 900 steps. The
    train/test split then puts an exact copy of every test row in the training set, and the
    nearest-neighbour baseline scores exactly 0.000 because it retrieves the answer rather
    than predicting it.

    Deduplicating before the split is the fix. It makes the datasets smaller and the scores
    worse, which is the point: the earlier scores were measuring retrieval.
    """
    if X.ndim < 2:
        return X
    flat = X.reshape(len(X), -1)
    _, first = np.unique(flat, axis=0, return_index=True)
    return X[np.sort(first)]


def leakage(X: np.ndarray, train_frac: float = 0.75) -> float:
    """Fraction of test rows with an exact copy in train. Must be 0."""
    flat = X.reshape(len(X), -1)
    k = int(len(flat) * train_frac)
    seen = {r.tobytes() for r in flat[:k]}
    test = flat[k:]
    return sum(r.tobytes() in seen for r in test) / max(len(test), 1)


def gray_scott(n: int, size: int = 16, steps: int = 900, f: float = 0.037,
               k: float = 0.06, du: float = 0.16, dv: float = 0.08, seed: int = 0):
    """Reaction-diffusion on a torus. A genuinely local PDE - the honest case for an NCA.

    Two chemicals diffuse and react. Every rule is local by construction, so a local
    automaton should do well here and a global branch should have little to add. It is the
    positive control for locality, the mirror of the shuffled datasets below.
    """
    rng = np.random.default_rng(seed)
    u = np.ones((size, size))
    v = np.zeros((size, size))
    c = size // 2
    r = max(2, size // 6)
    u[c - r:c + r, c - r:c + r] = 0.5
    v[c - r:c + r, c - r:c + r] = 0.25
    u += 0.02 * rng.standard_normal((size, size))
    v += 0.02 * rng.standard_normal((size, size))

    def lap(a):                                   # periodic: no walls, same as the models
        return (np.roll(a, 1, 0) + np.roll(a, -1, 0)
                + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a)

    out, keep_every = [], max(1, steps // (n + 5))
    for t in range(steps):
        uvv = u * v * v
        u = np.clip(u + du * lap(u) - uvv + f * (1 - u), 0, 1)
        v = np.clip(v + dv * lap(v) + uvv - (f + k) * v, 0, 1)
        if t % keep_every == 0:
            out.append(v.copy())
    return np.stack(out[:n])


def game_of_life(n: int, size: int = 16, seed: int = 0, run_len: int = 6):
    """Conway's rule on a torus. Discrete, exactly local, exactly known - no numerics.

    Sampled from MANY short independent runs rather than one long one. A single 16x16 board
    settles into still lifes and blinkers within a few dozen steps, so one long run yields
    only ~46 distinct states out of 900 - almost everything is a duplicate, and after
    deduplication there is not enough left to train on. Short runs from fresh random starts
    keep the boards in the interesting transient where the rule is actually doing something.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        g = (rng.random((size, size)) < rng.uniform(0.25, 0.45)).astype(float)
        for _ in range(run_len):
            nb = sum(np.roll(np.roll(g, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
            g = ((nb == 3) | ((g == 1) & (nb == 2))).astype(float)
            out.append(g.copy())
    return np.stack(out[:n])


def shuffle_cells(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply ONE fixed permutation to the cells of every example.

    This is the control the first run was missing. The data, its difficulty and its
    statistics are all unchanged - only the claim that neighbouring cells are related is
    destroyed. So any gap between a local automaton and a global one on a shuffled twin is
    attributable to locality alone, rather than to the dataset differing in some other way.

    The first run had only two datasets where locality was meaningless, and one of them
    turned out to be leaking, which left the central prediction resting on a single clean
    dataset. These twins fix that.
    """
    rng = np.random.default_rng(seed)
    flat = X.reshape(len(X), X.shape[1], -1)
    perm = rng.permutation(flat.shape[-1])
    return flat[:, :, perm].reshape(X.shape)


def _windows(series: np.ndarray, width: int, n: int, seed: int) -> np.ndarray:
    """Cut n random windows of `width` consecutive rows out of a (T, d) trajectory."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(series) - width - 1, size=n)
    return np.stack([series[s:s + width] for s in starts])


# ---------------------------------------------------------------------------------------
# 1. coupled map lattice - spatiotemporal chaos with no integrator to destabilise
# ---------------------------------------------------------------------------------------

def coupled_map_lattice(
    n_sites: int = 64, n_steps: int = 6000, a: float = 3.9, eps: float = 0.3,
    burn_in: int = 1000, seed: int = 0,
) -> np.ndarray:
    """Kaneko's diffusively coupled logistic map.

        x_{n+1}(i) = (1-eps) f(x_n(i)) + (eps/2) [ f(x_n(i-1)) + f(x_n(i+1)) ]
        f(x) = a x (1 - x),  periodic boundaries

    Spatially extended, genuinely chaotic, and an *explicit map* - there is no integrator,
    so it cannot destabilise the way our Kuramoto-Sivashinsky attempt did. It also has an
    analytic tridiagonal Jacobian, so the full Lyapunov spectrum is exact rather than
    estimated. This is the substrate that unblocks the spatially-extended gap.
    """
    rng = np.random.default_rng(seed)
    x = rng.random(n_sites)
    out = np.empty((n_steps, n_sites))
    for t in range(burn_in + n_steps):
        fx = a * x * (1.0 - x)
        x = (1.0 - eps) * fx + 0.5 * eps * (np.roll(fx, 1) + np.roll(fx, -1))
        if t >= burn_in:
            out[t - burn_in] = x
    return out


def cml_lyapunov_spectrum(n_sites: int = 64, a: float = 3.9, eps: float = 0.3,
                          n_steps: int = 2000, seed: int = 0) -> np.ndarray:
    """Exact Lyapunov spectrum by QR, using the analytic Jacobian.

    Not needed by the smoke test, but it is the reason this substrate is worth having:
    ground truth we compute rather than look up.
    """
    rng = np.random.default_rng(seed)
    x = rng.random(n_sites)
    Q = np.linalg.qr(rng.standard_normal((n_sites, n_sites)))[0]
    total = np.zeros(n_sites)
    for t in range(n_steps):
        fp = a * (1.0 - 2.0 * x)                       # f'(x), per site
        J = (1.0 - eps) * np.diag(fp)
        idx = np.arange(n_sites)
        J[idx, (idx - 1) % n_sites] += 0.5 * eps * fp[(idx - 1) % n_sites]
        J[idx, (idx + 1) % n_sites] += 0.5 * eps * fp[(idx + 1) % n_sites]
        Q, R = np.linalg.qr(J @ Q)
        d = np.diag(R)
        s = np.sign(d); s[s == 0] = 1.0
        Q *= s; total += np.log(np.abs(d) + 1e-300)
        fx = a * x * (1.0 - x)
        x = (1.0 - eps) * fx + 0.5 * eps * (np.roll(fx, 1) + np.roll(fx, -1))
    return np.sort(total / n_steps)[::-1]


# ---------------------------------------------------------------------------------------
# 2. elementary cellular automaton - discrete, spatially extended, exactly known
# ---------------------------------------------------------------------------------------

def eca(rule: int = 110, n_sites: int = 64, n_steps: int = 4000, seed: int = 0) -> np.ndarray:
    table = np.array([(rule >> k) & 1 for k in range(8)], dtype=np.uint8)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n_sites).astype(np.uint8)
    out = np.empty((n_steps, n_sites), dtype=np.uint8)
    for t in range(n_steps):
        idx = (np.roll(x, 1) << 2) | (x << 1) | np.roll(x, -1)
        x = table[idx]
        out[t] = x
    return out.astype(float)


# ---------------------------------------------------------------------------------------
# the eight datasets
# ---------------------------------------------------------------------------------------

def build_all(n_per_set: int = 900, seed: int = 0) -> list[Dataset]:
    from sklearn.datasets import load_breast_cancer, load_digits, load_sample_images

    sets: list[Dataset] = []
    rng = np.random.default_rng(seed)

    # --- 1. spatiotemporal chaos, 1-D lattice ------------------------------------------
    cml = coupled_map_lattice(64, 4000, seed=seed)
    sets.append(Dataset(
        "cml_lattice", "chaos", cml[:n_per_set][:, None, :], (64,), 1,
        ordered=True, temporal=False,
        note="one time-slice of a 64-site coupled map lattice; neighbours are coupled"))

    # --- 2. the same system as a time window: masking the tail IS forecasting -----------
    win = _windows(cml[:, :1], 64, n_per_set, seed)          # (n, 64, 1) one site over time
    sets.append(Dataset(
        "cml_timewindow", "chaos", win.transpose(0, 2, 1), (64,), 1,
        ordered=True, temporal=True,
        note="64 consecutive steps of one CML site; tail-masking = forecasting"))

    # --- 3. low-dimensional chaos, no spatial structure at all --------------------------
    traj = None
    for p in (REPO / "all-spikes/00-replications/data/trajectories").glob("Lorenz__20000__*.npy"):
        traj = np.load(p); break
    if traj is None:
        traj = coupled_map_lattice(3, 20000, seed=seed + 1)
    lz = _norm01(traj[:, :3])
    sets.append(Dataset(
        "lorenz_state", "chaos", lz[:n_per_set][:, None, :], (3,), 1,
        ordered=False, temporal=False,
        note="a bare Lorenz state vector - a 'lattice' of 3 cells with no locality"))

    # --- 4. discrete CA -----------------------------------------------------------------
    ca = eca(110, 64, n_per_set + 10, seed=seed)
    sets.append(Dataset(
        "eca_rule110", "ca", ca[:n_per_set][:, None, :], (64,), 1,
        ordered=True, temporal=False,
        note="rule 110, binary, spatially extended, exactly known"))

    # --- 5. small images ----------------------------------------------------------------
    dg = load_digits().images.astype(float) / 16.0
    sets.append(Dataset(
        "digits_8x8", "image", dg[:n_per_set][:, None, :, :], (8, 8), 1,
        ordered=True, temporal=False, note="handwritten digits, 2-D lattice"))

    # --- 6. natural image patches --------------------------------------------------------
    try:
        from scipy.datasets import ascent
        img = _norm01(ascent().astype(float))
    except Exception:                                        # noqa: BLE001
        img = _norm01(load_sample_images().images[0].mean(-1))
    ps = 16
    ys = rng.integers(0, img.shape[0] - ps, n_per_set)
    xs = rng.integers(0, img.shape[1] - ps, n_per_set)
    patches = np.stack([img[y:y + ps, x:x + ps] for y, x in zip(ys, xs)])
    sets.append(Dataset(
        "natural_patches_16x16", "image", patches[:, None, :, :], (ps, ps), 1,
        ordered=True, temporal=False, note="natural-image patches - heavy-tailed statistics"))

    # --- 7. text, character level ---------------------------------------------------------
    chunks = []
    for md in sorted(REPO.glob("**/*.md")):
        if ".git" in str(md):
            continue
        try:
            chunks.append(md.read_text(encoding="utf-8", errors="ignore"))
        except Exception:                                    # noqa: BLE001
            pass
    text = "\n".join(chunks) or "the quick brown fox " * 5000
    codes = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8).astype(float)
    starts = rng.integers(0, max(1, len(codes) - 65), size=n_per_set)
    lines = np.stack([codes[s:s + 64] for s in starts]) / 255.0
    sets.append(Dataset(
        "text_chars", "text", lines[:, None, :], (64,), 1,
        ordered=True, temporal=True,
        note="characters as byte/255 on a 1-D lattice; a deliberately crude encoding, "
             "chosen so every modality goes through one identical pipeline"))

    # --- 8. tabular: the control, where locality is meaningless ---------------------------
    bc = load_breast_cancer().data
    bc = np.stack([_norm01(bc[:, j]) for j in range(bc.shape[1])], axis=1)
    reps = int(np.ceil(n_per_set / len(bc)))
    bc = np.tile(bc, (reps, 1))[:n_per_set]
    sets.append(Dataset(
        "tabular_breastcancer", "tabular", bc[:, None, :], (30,), 1,
        ordered=False, temporal=False,
        note="30 clinical features - adjacent 'cells' have no relationship. The control."))

    # --- 9. band-limited signal ------------------------------------------------------------
    t = np.linspace(0, 8 * np.pi, 64)
    freqs = rng.uniform(0.5, 4.0, (n_per_set, 3))
    phase = rng.uniform(0, 2 * np.pi, (n_per_set, 3))
    amp = rng.uniform(0.3, 1.0, (n_per_set, 3))
    sig = sum(amp[:, k:k + 1] * np.sin(freqs[:, k:k + 1] * t + phase[:, k:k + 1]) for k in range(3))
    sets.append(Dataset(
        "multitone_signal", "signal", _norm01(sig)[:, None, :], (64,), 1,
        ordered=True, temporal=True,
        note="sum of three sinusoids - smooth, predictable, an easy control"))

    # --- 10-12. genuinely local 2-D systems, and a null control -------------------------
    gs = _norm01(gray_scott(n_per_set, 16, seed=seed))
    sets.append(Dataset(
        "gray_scott_16x16", "pde", gs[:, None, :, :], (16, 16), 1,
        ordered=True, temporal=False,
        note="reaction-diffusion on a torus - every rule local, the positive control"))

    gol = game_of_life(n_per_set, 16, seed=seed)
    sets.append(Dataset(
        "game_of_life_16x16", "ca", gol[:, None, :, :], (16, 16), 1,
        ordered=True, temporal=False,
        note="Conway on a torus - discrete, exactly local, exactly known"))

    noise = rng.random((n_per_set, 64))
    sets.append(Dataset(
        "random_field", "control", noise[:, None, :], (64,), 1,
        ordered=False, temporal=False,
        note="pure noise. NOTHING should beat guessing the mean. If anything does, the "
             "pipeline is broken and every other row is suspect."))

    # --- 13-15. shuffled twins: the same data with locality destroyed --------------------
    # Each is an exact copy of an ordered dataset with one fixed cell permutation applied.
    # Same values, same difficulty, same statistics - only adjacency is gone.
    for src in ("cml_lattice", "digits_8x8", "multitone_signal"):
        base = next(d for d in sets if d.name == src)
        sets.append(Dataset(
            f"{src}__shuffled", base.modality, shuffle_cells(base.X, seed=7),
            base.spatial, base.channels, ordered=False, temporal=False,
            note=f"{src} with the cells permuted once - the locality control"))

    # Every dataset is deduplicated before it is ever split. See `dedup` for why - one
    # baseline was scoring a perfect 0.000 by retrieving an identical training row.
    for d in sets:
        before = len(d.X)
        d.X = dedup(d.X)
        d.n_duplicates_removed = before - len(d.X)

    return sets


def demo() -> None:
    """Self-check: shapes are consistent, values are in range, and the CML is chaotic."""
    # No test example may have an exact copy in training. This is asserted rather than
    # trusted: two datasets used to fail it outright (clinical features tiled from 569 rows
    # to 900, rule 110 settling into 337 distinct states), and the nearest-neighbour
    # baseline scored a perfect 0.000 by retrieving instead of predicting.
    for _d in build_all(900):
        _lk = leakage(_d.X)
        assert _lk == 0.0, f"{_d.name}: {_lk:.0%} of test rows have an exact copy in train"

    lam = cml_lyapunov_spectrum(16, n_steps=600)
    assert lam[0] > 0.1, f"CML should be chaotic, largest exponent was {lam[0]:.3f}"
    assert lam[0] > lam[-1], "spectrum must be sorted descending"

    ca = eca(110, 32, 200)
    assert set(np.unique(ca)) <= {0.0, 1.0}, "ECA must be binary"
    assert ca.std() > 0.1, "rule 110 should not settle to a constant"

    sets = build_all(n_per_set=40)
    assert len(sets) == 9, f"expected 9 datasets, built {len(sets)}"
    for d in sets:
        assert d.X.ndim == 2 + len(d.spatial), f"{d.name}: bad rank {d.X.shape}"
        assert d.X.shape[1] == d.channels, f"{d.name}: channel mismatch"
        assert d.X.shape[2:] == d.spatial, f"{d.name}: spatial mismatch"
        assert np.isfinite(d.X).all(), f"{d.name}: non-finite values"
        assert -0.01 <= d.X.min() and d.X.max() <= 1.01, f"{d.name}: out of [0,1]"
        assert d.X.dtype == np.float32, f"{d.name}: dtype {d.X.dtype}, torch needs float32"
    print(f"self-check passed - {len(sets)} datasets, CML lambda_max={lam[0]:.3f}")
    for d in sets:
        print(f"   {d!r}  ordered={d.ordered} temporal={d.temporal}")


if __name__ == "__main__":
    demo()
