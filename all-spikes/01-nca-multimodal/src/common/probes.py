"""Four measurements that ask what an automaton is actually doing, not how well it scores.

The point of this spike is not to fit data. It is to read a signal off an architecture. So
alongside the error numbers we measure the mechanism directly:

    propagation      how fast does information physically travel across the lattice?
    knockout         if the global branch is silenced, how much worse does it get?
    contribution     what fraction of each update does the global branch actually supply?
    memory           how much does the automaton rely on what it carried from earlier steps?

`propagation` is the one that motivates the whole design. A purely local rule moves
information one cell per step, so on a 100x100 grid a corner cannot influence the opposite
corner in fewer than ~100 steps, no matter how many parameters it has. That is a hard
structural limit, not a training problem, and it is measurable in a few seconds.

Two details that make these honest:

  - the update is normally stochastic (half the cells fire per step). Every probe here sets
    the fire rate to 1.0 and restores it afterwards, because otherwise two rollouts that
    should differ only in the thing being probed also differ in which cells fired.
  - `global_gain` scales whatever the global branch contributes. Setting it to 0 silences
    that branch without touching a single weight, so knockout compares one model against
    itself rather than against a separately trained one.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from src.common.task import score


@contextlib.contextmanager
def _deterministic(model):
    """Fire every cell, every step, so two rollouts differ only in what we are probing."""
    saved = model.fire_rate
    model.fire_rate = 1.0
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.fire_rate = saved
        model.train(was_training)


@contextlib.contextmanager
def _gain(model, value: float):
    saved = model.global_gain
    model.global_gain = value
    try:
        yield
    finally:
        model.global_gain = saved


# ---------------------------------------------------------------------------------------
# 1. how fast does information travel?
# ---------------------------------------------------------------------------------------

def propagation(model, spatial: tuple, steps: int = 16, seed: int = 0,
                tol: float = 1e-6) -> dict:
    """Poke one cell. Watch how far the ripple has reached after each step.

    Returns the fraction of the lattice affected at each step, and the radius in cells.
    A local rule gives radius ~= step. Anything with a working global branch reaches the
    whole lattice on step 1, and the difference is unmistakable.
    """
    torch.manual_seed(seed)
    nd = len(spatial)
    obs = torch.rand(1, model.c_data, *spatial)
    mask = torch.ones(1, 1, *spatial)

    centre = tuple(d // 2 for d in spatial)
    idx = (0, slice(None)) + centre

    with _deterministic(model), torch.no_grad():
        a = model.seed(obs, mask)
        b = a.clone()
        b[idx] = b[idx] + 1.0                       # the poke
        n_cells = int(np.prod(spatial))
        coords = np.stack(np.meshgrid(*[np.arange(d) for d in spatial], indexing="ij"), -1)
        # distance on a torus, because the lattice wraps - which is the whole reason the
        # boundary never blows up here
        delta = np.abs(coords - np.array(centre))
        delta = np.minimum(delta, np.array(spatial) - delta)
        dist = delta.max(-1) if nd > 1 else delta[..., 0]

        frac, radius = [], []
        for _ in range(steps):
            a, b = model.step(a), model.step(b)
            diff = (a - b).abs().sum(1)[0].numpy()
            hit = diff > max(tol, tol * float(np.abs(diff).max()))
            frac.append(float(hit.sum()) / n_cells)
            radius.append(float(dist[hit].max()) if hit.any() else 0.0)

    return {"frac_affected": frac, "radius": radius, "n_cells": n_cells,
            "max_radius": float(dist.max())}


def propagation_summary(prop: dict) -> dict:
    """Two numbers: how far after one step, and how many steps to cover the lattice."""
    frac = prop["frac_affected"]
    reach = next((i + 1 for i, f in enumerate(frac) if f > 0.99), None)
    return {
        "frac_affected_step1": frac[0] if frac else float("nan"),
        "radius_step1": prop["radius"][0] if prop["radius"] else float("nan"),
        "steps_to_cover": reach if reach is not None else float("nan"),
        "frac_affected_final": frac[-1] if frac else float("nan"),
    }


# ---------------------------------------------------------------------------------------
# 2. silence the global branch and see what breaks
# ---------------------------------------------------------------------------------------

def knockout(model, obs, mask, true, steps: int, binary: bool = False,
             seed: int = 0) -> dict:
    """Error with the global branch on, and with it silenced. Same weights both times."""
    out = {}
    for label, g in (("on", 1.0), ("off", 0.0)):
        torch.manual_seed(seed)                     # identical firing pattern both runs
        with _deterministic(model), _gain(model, g), torch.no_grad():
            pred = model(torch.tensor(obs), torch.tensor(mask), steps=steps).numpy()
        out[f"nrmse_global_{label}"] = score(pred, true, mask, binary)["nrmse_hidden"]
    on, off = out["nrmse_global_on"], out["nrmse_global_off"]
    # > 1 means silencing the global branch made things worse, i.e. it was doing something
    out["knockout_ratio"] = float(off / on) if on > 1e-12 else float("nan")
    return out


# ---------------------------------------------------------------------------------------
# 3. what share of each update does the global branch supply?
# ---------------------------------------------------------------------------------------

def contribution(model, obs, mask, steps: int = 8, seed: int = 0) -> dict:
    """At every step, compare the update the model makes with and without its global branch.

    Reported as the share of the update's magnitude that the global branch is responsible
    for. Zero means the branch is present but idle - which is a real outcome and worth
    catching, because a branch that is never used still costs parameters.
    """
    if not model.has_global():
        return {"global_share_mean": 0.0, "global_share_final": 0.0, "global_share": []}

    torch.manual_seed(seed)
    with _deterministic(model), torch.no_grad():
        s = model.seed(torch.tensor(obs), torch.tensor(mask))
        shares = []
        for _ in range(steps):
            with _gain(model, 1.0):
                full = model.rule(s)
            with _gain(model, 0.0):
                local = model.rule(s)
            num = (full - local).norm().item()
            den = full.norm().item()
            shares.append(num / den if den > 1e-12 else 0.0)
            s = s + full
    return {"global_share_mean": float(np.mean(shares)),
            "global_share_final": float(shares[-1]),
            "global_share": shares}


# ---------------------------------------------------------------------------------------
# 4. how much does it rely on what it remembered?
# ---------------------------------------------------------------------------------------

def memory(model, obs, mask, true, steps: int = 16, wipe_at: int = 8,
           binary: bool = False, seed: int = 0) -> dict:
    """Halfway through, erase the scratch channels and let it carry on.

    The visible data and the mask are left alone, so the automaton still knows what it was
    asked. Only what it worked out along the way is destroyed. If the answer barely changes,
    the rule is essentially memoryless and each step is recomputing from the data; if it
    collapses, the automaton has been accumulating something across steps - which is what
    "depends on history" means for a system with no explicit history buffer.
    """
    keep = model.c_data + 1                          # data channels + the mask channel
    if model.c_state <= keep:
        return {"memory_ratio": float("nan"), "nrmse_wiped": float("nan")}

    res = {}
    for label, wipe in (("intact", False), ("wiped", True)):
        torch.manual_seed(seed)
        with _deterministic(model), torch.no_grad():
            s = model.seed(torch.tensor(obs), torch.tensor(mask))
            for k in range(steps):
                s = model.step(s)
                if wipe and k + 1 == wipe_at:
                    s = s.clone()
                    s[:, keep:] = 0.0
            pred = s[:, : model.c_data].numpy()
        res[label] = score(pred, true, mask, binary)["nrmse_hidden"]
    intact = res["intact"]
    return {"nrmse_intact": intact, "nrmse_wiped": res["wiped"],
            "memory_ratio": float(res["wiped"] / intact) if intact > 1e-12 else float("nan")}


# ---------------------------------------------------------------------------------------

def demo() -> None:
    """Self-check: the probes must reproduce facts we already know to be true."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.common import models as M

    torch.manual_seed(0)
    spatial = (32,)

    # A trained model is not needed: propagation is a property of the wiring, and the
    # zero-initialised head would make every update exactly zero, so we perturb the weights
    # to make the rule non-trivial before measuring how far it reaches.
    def wake(m):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.05 * torch.randn_like(p))
        return m

    local = wake(M.build("A1_local", spatial, 1, scale=0.5))
    p_local = propagation(local, spatial, steps=6)
    r = p_local["radius"]
    assert r[0] <= 1.5, f"a local rule must reach ~1 cell in one step, got {r[0]}"
    assert r[-1] <= 6.5, f"a local rule cannot outrun one cell per step, got {r}"
    assert r[-1] > r[0], f"a local rule must still spread over time, got {r}"

    for name in ("A2_pooled", "A3_attentive", "A4_interleaved"):
        g = wake(M.build(name, spatial, 1, scale=0.5))
        pg = propagation(g, spatial, steps=3)
        assert pg["frac_affected"][0] > 0.5, (
            f"{name}: a global branch must reach most of the lattice in one step, "
            f"got {pg['frac_affected'][0]:.2f}")

    # contribution: a global branch must supply a non-zero share; A1 must report exactly 0
    obs = np.random.default_rng(0).random((4, 1, *spatial)).astype("float32")
    msk = (np.random.default_rng(1).random((4, 1, *spatial)) > 0.3).astype("float32")
    a2 = wake(M.build("A2_pooled", spatial, 1, scale=0.5))
    c2 = contribution(a2, obs, msk, steps=4)
    assert c2["global_share_mean"] > 0, "A2's global branch reported zero contribution"
    assert contribution(local, obs, msk, steps=4)["global_share_mean"] == 0.0

    # Instrument calibration. On a randomly perturbed model the measured share is near zero,
    # which is correct - an untrained branch is not used - but it means "> 0" is a weak test.
    # So drive the global path harder and check the reading follows. If it does not, the
    # probe is not measuring what it claims and every number it produces later is worthless.
    shares = []
    for boost in (1.0, 10.0, 100.0):
        m = M.build("A2_pooled", spatial, 1, scale=0.5)
        with torch.no_grad():
            for p_ in m.parameters():
                p_.add_(0.05 * torch.randn_like(p_))
            for p_ in m.global_net.parameters():
                p_.mul_(boost)
        shares.append(contribution(m, obs, msk, steps=4, seed=3)["global_share_mean"])
    assert shares[0] < shares[1] < shares[2], (
        f"global share must rise as the global branch is amplified, got {shares}")
    assert shares[2] > 0.05, f"a 100x-amplified global branch should dominate, got {shares[2]}"

    # knockout must be exactly inert for a model with no global branch
    true = np.random.default_rng(2).random(obs.shape).astype("float32")
    k1 = knockout(local, obs, msk, true, steps=4)
    assert abs(k1["knockout_ratio"] - 1.0) < 1e-6, (
        f"silencing a branch A1 does not have changed its answer: {k1}")
    k2 = knockout(a2, obs, msk, true, steps=4)
    assert np.isfinite(k2["knockout_ratio"])

    m = memory(a2, obs, msk, true, steps=8, wipe_at=4)
    assert np.isfinite(m["memory_ratio"])

    print(f"   A1 radius by step: {[round(v) for v in p_local['radius']]}  (one cell per step)")
    print(f"   A2 lattice covered on step 1: {propagation(a2, spatial, steps=1)['frac_affected'][0]:.0%}")
    print(f"   A2 global share of update: {c2['global_share_mean']:.2e} (untrained: near zero, as it should be)")
    print(f"   share as the branch is amplified 1x/10x/100x: "
          f"{shares[0]:.2e} -> {shares[1]:.2e} -> {shares[2]:.2e}")
    print("self-check passed")


if __name__ == "__main__":
    demo()
