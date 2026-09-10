"""Collect as many genuinely different real datasets as possible, in one common form.

The question this spike exists for is whether the architecture matters or the data does. To
answer that fairly you need data that varies far more than the architectures do - not
fifteen chaotic attractors, but weather, electricity, heartbeats, sales, macroeconomics,
sunspots, traffic, air quality, images, and simulated physics side by side.

Everything is reduced to the same object: a float array of shape (T, D), a multivariate
series in time order. Then one task is defined on all of them - predict the next
observation from a window of the last W - so every architecture sees an identical interface
and any difference between datasets is about the data rather than the framing.

Sources, in order of preference:
  - bundled with installed libraries (statsmodels, sklearn): always available, no network
  - small CSVs from public mirrors: cached to disk on first fetch
  - simulated dynamical systems from this project: known ground truth

Anything that fails to download is skipped and reported rather than silently dropped, so
the dataset count in a writeup always matches what actually ran.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

RAW = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"

# name -> (url, kind, note). Kind is the world-model-relevant category.
# name -> (filename, kind). Kind is the world-model-relevant category.
# Deliberately broad: temporal and non-temporal, low and high dimensional, physical,
# human, economic, biomedical and sensor. The point of the study is that the data varies
# far more than the architectures do, so the data has to actually vary.
_JB = {
    # --- temporal ---------------------------------------------------------------------
    "temps_daily_min":     ("daily-min-temperatures.csv", "weather"),
    "temps_daily_max":     ("daily-max-temperatures.csv", "weather"),
    "temps_monthly_mean":  ("monthly-mean-temp.csv", "weather"),
    "births_daily":        ("daily-total-female-births.csv", "human"),
    "sunspots_monthly":    ("monthly-sunspots.csv", "physical"),
    "airline_passengers":  ("airline-passengers.csv", "human"),
    "airline_monthly":     ("monthly-airline-passengers.csv", "human"),
    "car_sales_monthly":   ("monthly-car-sales.csv", "economic"),
    "robberies_monthly":   ("monthly-robberies.csv", "human"),
    "shampoo_sales":       ("monthly-shampoo-sales.csv", "economic"),
    "paper_sales":         ("monthly-writing-paper-sales.csv", "economic"),
    "champagne_sales":     ("monthly_champagne_sales.csv", "economic"),
    "water_usage_yearly":  ("yearly-water-usage.csv", "physical"),
    "pollution_beijing":   ("pollution.csv", "sensor"),
    # --- non-temporal measurement -----------------------------------------------------
    "abalone":             ("abalone.csv", "biological"),
    "auto_imports":        ("auto_imports.csv", "engineering"),
    "banknote":            ("banknote_authentication.csv", "sensor"),
    "bc_wisconsin":        ("breast-cancer-wisconsin.csv", "biomedical"),
    "ecoli":               ("ecoli.csv", "biological"),
    "german_credit":       ("german.csv", "economic"),
    "glass":               ("glass.csv", "chemical"),
    "haberman":            ("haberman.csv", "biomedical"),
    "horse_colic":         ("horse-colic.csv", "biomedical"),
    "housing_boston":      ("housing.csv", "economic"),
    "ionosphere":          ("ionosphere.csv", "sensor"),
    "mammography":         ("mammography.csv", "biomedical"),
    "thyroid":             ("new-thyroid.csv", "biomedical"),
    "oil_spill":           ("oil-spill.csv", "sensor"),
    "phoneme":             ("phoneme.csv", "audio"),
    "pima_diabetes":       ("pima-indians-diabetes.csv", "biomedical"),
    "sonar":               ("sonar.csv", "sensor"),
    "wheat_seeds":         ("wheat-seeds.csv", "biological"),
    "wine_quality_red":    ("winequality-red.csv", "chemical"),
    "wine_quality_white":  ("winequality-white.csv", "chemical"),
    "iris":                ("iris.csv", "biological"),
}
CSV_SOURCES = {k: (RAW + f, kind) for k, (f, kind) in _JB.items()}

SM_SOURCES = {
    "co2_weekly":        ("co2", "physical"),
    "elnino_monthly":    ("elnino", "weather"),
    "nile_annual":       ("nile", "physical"),
    "sunspots_annual":   ("sunspots", "physical"),
    "macro_us":          ("macrodata", "economic"),
    "elec_equipment":    ("elec_equip", "economic"),
    "interest_inflation": ("interest_inflation", "economic"),
    "copper_market":     ("copper", "economic"),
    "grunfeld_firms":    ("grunfeld", "economic"),
}


def _cached(name: str, url: str) -> bytes | None:
    f = CACHE / f"{name}.csv"
    if f.exists():
        return f.read_bytes()
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            b = r.read()
        f.write_bytes(b)
        return b
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {name}: {type(exc).__name__}", flush=True)
        return None


def _numeric(df: pd.DataFrame) -> np.ndarray | None:
    num = df.select_dtypes(include=[np.number])
    num = num.loc[:, num.std() > 1e-9]
    if num.shape[1] == 0 or len(num) < 120:
        return None
    a = num.to_numpy(dtype=float)
    a = a[~np.isnan(a).any(axis=1)]
    return a if len(a) >= 120 else None


def load_all(include_sim: bool = True, verbose: bool = True) -> list[dict]:
    out = []

    for name, (url, kind) in CSV_SOURCES.items():
        b = _cached(name, url)
        if b is None:
            continue
        try:
            a = _numeric(pd.read_csv(io.BytesIO(b)))
        except Exception:  # noqa: BLE001
            a = None
        if a is not None:
            out.append({"name": name, "kind": kind, "X": a, "source": "csv"})

    try:
        import statsmodels.api as sm
        for name, (mod, kind) in SM_SOURCES.items():
            try:
                d = getattr(sm.datasets, mod).load_pandas().data
                a = _numeric(d)
                if a is not None:
                    out.append({"name": name, "kind": kind, "X": a, "source": "statsmodels"})
            except Exception:  # noqa: BLE001
                if verbose:
                    print(f"  skip {name} (statsmodels)", flush=True)
    except ImportError:
        pass

    # sklearn bundled: not time series, but real high-dimensional measurement data.
    # Included deliberately - a world model sees plenty of data with no temporal order,
    # and treating it as a series is exactly the mistake worth being able to detect.
    try:
        from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine
        for nm, fn, kind in [("breast_cancer", load_breast_cancer, "biomedical"),
                             ("diabetes_tab", load_diabetes, "biomedical"),
                             ("wine_tab", load_wine, "chemical")]:
            a = fn().data.astype(float)
            if len(a) >= 120:
                out.append({"name": nm, "kind": kind, "X": a, "source": "sklearn"})
        im = load_digits().images.astype(float) / 16.0
        out.append({"name": "digits_rows", "kind": "image",
                    "X": im.reshape(len(im), -1)[:, :16], "source": "sklearn"})
    except Exception:  # noqa: BLE001
        pass

    if include_sim:
        sys.path.insert(0, str(HERE.parents[0] / "00-replications"))
        try:
            from src.common import systems as S
            for nm, kind in [("Lorenz", "sim-chaos"), ("Rossler", "sim-chaos"),
                             ("Thomas", "sim-chaos"), ("HyperCai", "sim-chaos"),
                             ("DequanLi", "sim-chaos"), ("Halvorsen", "sim-chaos")]:
                try:
                    out.append({"name": f"sim_{nm}", "kind": kind,
                                "X": S.trajectory(nm, 6000), "source": "simulated"})
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

        rng = np.random.default_rng(0)
        n = 4000
        # controls with known answers, so the study contains its own calibration
        out.append({"name": "ctrl_white_noise", "kind": "control",
                    "X": rng.standard_normal((n, 3)), "source": "control"})
        t = np.linspace(0, 120 * np.pi, n)
        out.append({"name": "ctrl_periodic", "kind": "control",
                    "X": np.c_[np.sin(t), np.cos(1.7 * t), np.sin(0.3 * t)],
                    "source": "control"})
        out.append({"name": "ctrl_random_walk", "kind": "control",
                    "X": np.cumsum(rng.standard_normal((n, 3)), axis=0), "source": "control"})

    if verbose:
        print(f"\n{len(out)} datasets loaded")
    return out


def summary(sets: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"name": s["name"], "kind": s["kind"], "source": s["source"],
                          "T": s["X"].shape[0], "D": s["X"].shape[1]} for s in sets])


def demo() -> None:
    sets = load_all()
    df = summary(sets)
    print(df.to_string(index=False))
    print(f"\nby kind:\n{df.kind.value_counts().to_string()}")
    assert len(sets) >= 15, f"only {len(sets)} datasets - the study needs breadth"
    for s in sets:
        assert s["X"].ndim == 2 and np.isfinite(s["X"]).all(), s["name"]
        assert s["X"].shape[0] >= 120, s["name"]
    print("\nself-check passed")


if __name__ == "__main__":
    demo()
