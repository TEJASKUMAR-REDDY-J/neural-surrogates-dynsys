"""Point this at any dataset and it tells you three things before you train anything.

    1. Will a lookup table match whatever you build?
    2. How much of this data is irreducible noise?
    3. If it is a time series: is the difficulty noise, or is it genuine dynamics?

All three come out of the experiments in this project, and all three are cheap - no
training, no gradients, seconds on a laptop. Works on any float array: image patches,
tabular rows, sensor traces, embeddings.

Why these three and not something cleverer: we tried the clever thing. Predicting a
surrogate's accuracy from cheap statistics of the system reached about a third of the
variance on noiseless simulation and collapsed to nothing at one percent measurement noise.
What survived was coarser and more robust - not "how accurate will my model be" but "is
there anything here for a model to do that copying cannot".

    python -m diagnose            # runs the self-checks and the worked examples
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------
# 1. COVERAGE - will a lookup table match a trained model?
# ---------------------------------------------------------------------------------------

def coverage(X_train: np.ndarray, X_query: np.ndarray, n_query: int = 500,
             seed: int = 0) -> dict:
    """How close is a typical new point to its nearest neighbour in the training set?

    Reported relative to the spread of the data, so it is comparable across datasets with
    different units and dimensions. Small means the training set already contains a near
    twin of everything you will ever be asked about - and then retrieval works as well as
    learning, because there is nothing to interpolate.

    This is the single most transferable number in the project. It is what explains why a
    trained network so often only ties with copying on benchmark data: the benchmark was
    sampled densely enough that copying is already optimal.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(X_train, float).reshape(len(X_train), -1)
    B = np.asarray(X_query, float).reshape(len(X_query), -1)
    q = B[rng.choice(len(B), size=min(n_query, len(B)), replace=False)]

    # chunked so a large training set does not blow up memory
    nn = np.empty(len(q))
    for i in range(0, len(q), 64):
        d2 = ((q[i:i + 64, None, :] - A[None, :, :]) ** 2).sum(-1)
        nn[i:i + 64] = np.sqrt(d2.min(1))

    a = A[rng.choice(len(A), size=min(400, len(A)))]
    b = A[rng.choice(len(A), size=min(400, len(A)))]
    spread = float(np.sqrt(((a - b) ** 2).sum(-1)).mean())
    c = float(np.median(nn) / max(spread, 1e-12))
    return {"coverage": c, "nn_distance": float(np.median(nn)), "spread": spread}


# ---------------------------------------------------------------------------------------
# 2. NOISE FLOOR - how much of the target is unpredictable in principle?
# ---------------------------------------------------------------------------------------

def noise_floor(X: np.ndarray, Y: np.ndarray, k: int = 5, seed: int = 0) -> dict:
    """Estimate the share of target variance no model can ever explain.

    The idea is old and simple. Find pairs of inputs that are nearly identical. Whatever
    their targets disagree about cannot be a function of the input, so it must be noise.
    Averaging that disagreement over many near-identical pairs, and halving it because two
    independent noise draws contribute twice, estimates the irreducible variance.

    This is the number that tells you the ceiling. If it says 40%, then a model reporting
    R^2 of 0.6 has already finished and buying a bigger one is wasted money.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(X, float).reshape(len(X), -1)
    Yv = np.asarray(Y, float).reshape(len(Y), -1)
    n = min(len(A), 2000)
    idx = rng.choice(len(A), size=n, replace=False)
    A, Yv = A[idx], Yv[idx]

    diffs = []
    for i in range(0, n, 64):
        d2 = ((A[i:i + 64, None, :] - A[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2[:, i:i + 64], np.inf)          # never match yourself
        nb = np.argpartition(d2, k, axis=1)[:, :k]
        for r, row in enumerate(nb):
            diffs.append(((Yv[i + r] - Yv[row]) ** 2).mean(0))
    # half the mean squared disagreement between near-twins = one noise variance
    noise_var = float(np.mean(diffs) / 2)
    total_var = float(Yv.var(0).mean())
    frac = min(1.0, noise_var / max(total_var, 1e-12))
    return {"noise_fraction": frac, "noise_var": noise_var, "total_var": total_var,
            "best_possible_r2": max(0.0, 1.0 - frac)}


# ---------------------------------------------------------------------------------------
# 3. HORIZON SHAPE - for time series, is the difficulty noise or dynamics?
# ---------------------------------------------------------------------------------------

def horizon_shape(series: np.ndarray, horizons=(1, 2, 4, 8, 16, 32, 64),
                  n_query: int = 300, k: int = 4, seed: int = 0) -> dict:
    """How does prediction error grow as you forecast further ahead?

    Measured with a lookup forecaster, so it needs no training and describes the DATA
    rather than any particular model.

    The shape is the diagnostic, and it separates two things that look identical at a
    single horizon:

      flat          error is already at its ceiling at one step and does not grow.
                    The unpredictability is OBSERVATION NOISE. No model and no horizon
                    will help, because there is nothing systematic to get wrong.

      rising        error is small at one step and climbs steeply. The system is
                    DETERMINISTIC BUT DIVERGENT - chaotic. Short forecasts are worth
                    making, long ones are not, and the climb rate tells you where the
                    cutoff is.

      low and flat  error is small everywhere. The system is periodic, slow, or strongly
                    driven. A direct long-range jump will work as well as stepping.
    """
    rng = np.random.default_rng(seed)
    S = np.asarray(series, float)
    if S.ndim == 1:
        S = S[:, None]
    H = max(horizons)
    split = len(S) // 2
    train, test = S[:split], S[split:]
    sd = train.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    train, test = train / sd, test / sd

    usable = len(test) - H - 1
    starts = rng.choice(usable, size=min(n_query, usable), replace=False)
    anchors = np.arange(0, len(train) - H - 1)
    ctx = train[anchors]

    errs = {}
    for i, s in enumerate(starts):
        d = np.sqrt(((ctx - test[s]) ** 2).sum(-1))
        nb = anchors[np.argpartition(d, k)[:k]]
        for h in horizons:
            pred = train[nb + h].mean(0)
            e = float(((pred - test[s + h]) ** 2).mean())
            errs.setdefault(h, []).append(e)

    curve = {h: float(np.sqrt(np.mean(v))) for h, v in errs.items()}
    base = float(np.sqrt(((test - test.mean(0)) ** 2).mean()))   # predicting the mean
    rel = {h: v / max(base, 1e-12) for h, v in curve.items()}

    e1, eH = rel[min(horizons)], rel[max(horizons)]
    growth = eH / max(e1, 1e-9)
    if e1 > 0.75:
        verdict = "noise-dominated: already unpredictable at one step"
    elif growth > 3.0:
        verdict = "deterministic but divergent (chaos-like): short forecasts only"
    elif eH < 0.5:
        verdict = "predictable far ahead: periodic, slow, or strongly driven"
    else:
        verdict = "mixed: partly predictable, degrading with horizon"
    return {"relative_error_by_horizon": rel, "growth_ratio": growth,
            "one_step_relative_error": e1, "verdict": verdict}


# ---------------------------------------------------------------------------------------

def report(X, Y=None, series=None, name: str = "dataset") -> dict:
    """Everything at once, with the verdicts spelled out."""
    out = {"name": name}
    print(f"\n=== {name} ===")

    if X is not None:
        n = len(X)
        split = int(n * 0.75)
        cov = coverage(X[:split], X[split:])
        out.update(cov)
        v = ("a lookup table will likely MATCH a trained model - the data already contains "
             "a near twin of everything") if cov["coverage"] < 0.02 else (
            "training should pay - new points land far from anything seen")
        print(f"  coverage           {cov['coverage']:.4f}   -> {v}")

    if Y is not None:
        nf = noise_floor(X, Y)
        out.update(nf)
        print(f"  irreducible noise  {nf['noise_fraction']:.1%}      -> best achievable "
              f"R^2 is about {nf['best_possible_r2']:.2f}")

    if series is not None:
        hs = horizon_shape(series)
        out.update(hs)
        pretty = "  ".join(f"h={h}:{e:.2f}" for h, e in hs["relative_error_by_horizon"].items())
        print(f"  error vs horizon   {pretty}")
        print(f"  verdict            {hs['verdict']}")
    return out


def demo() -> None:
    """Self-checks on data whose answers we know in advance, then worked examples."""
    rng = np.random.default_rng(0)

    # --- self-check 1: pure noise must be called noise-dominated -----------------------
    pure = rng.standard_normal((4000, 3))
    h = horizon_shape(pure)
    assert h["one_step_relative_error"] > 0.75, h
    assert "noise" in h["verdict"], h["verdict"]

    # --- self-check 2: a clean sine must be predictable far ahead ----------------------
    t = np.linspace(0, 200 * np.pi, 8000)
    sine = np.c_[np.sin(t), np.cos(t)]
    h2 = horizon_shape(sine)
    assert h2["relative_error_by_horizon"][64] < 0.5, h2
    assert "predictable far ahead" in h2["verdict"], h2["verdict"]

    # --- self-check 3: the noise-floor estimator must recover a known noise level ------
    Xc = rng.standard_normal((3000, 4))
    clean = (Xc[:, :1] * 2 + Xc[:, 1:2]) ** 1
    for true_frac in (0.0, 0.25):
        sd = np.sqrt(true_frac / max(1 - true_frac, 1e-9)) * clean.std()
        Yn = clean + sd * rng.standard_normal(clean.shape)
        est = noise_floor(Xc, Yn)["noise_fraction"]
        assert abs(est - true_frac) < 0.18, (
            f"noise estimator off: true {true_frac:.2f}, estimated {est:.2f}")

    # --- self-check 4: more data must mean denser coverage -----------------------------
    big = rng.standard_normal((6000, 3))
    c_small = coverage(big[:300], big[5000:])["coverage"]
    c_big = coverage(big[:4000], big[5000:])["coverage"]
    assert c_big < c_small, (c_small, c_big)

    print("self-checks passed: noise called noise, sine called predictable, "
          "noise level recovered, coverage falls with data")

    # ------------------------------------------------------------------ worked examples
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits

    d = load_digits()
    Xd = d.images.reshape(len(d.images), -1) / 16.0
    report(Xd, name="sklearn digits (images, 8x8)")

    bc = load_breast_cancer()
    Xb = (bc.data - bc.data.mean(0)) / bc.data.std(0)
    report(Xb, Y=bc.target[:, None].astype(float), name="breast cancer (tabular, classify)")

    db = load_diabetes()
    report(db.data, Y=db.target[:, None] / db.target.std(),
           name="diabetes (tabular, regression)")

    # a chaotic trace, a noisy version of it, and pure noise - same pipeline
    x = np.zeros((6000, 3))
    x[0] = [1.0, 1.0, 1.0]
    dt = 0.01
    for i in range(1, 6000):                                   # Lorenz, explicit Euler
        a, b, c = x[i - 1]
        x[i] = x[i - 1] + dt * np.array([10 * (b - a), a * (28 - c) - b, a * b - 8 / 3 * c])
    report(None, series=x, name="Lorenz (chaotic, clean)")
    report(None, series=x + 0.30 * x.std(0) * rng.standard_normal(x.shape),
           name="Lorenz + 30% measurement noise")
    report(None, series=np.c_[np.sin(t), np.cos(t)][:6000], name="a clean sine wave")
    report(None, series=rng.standard_normal((6000, 3)), name="pure noise")


if __name__ == "__main__":
    demo()
