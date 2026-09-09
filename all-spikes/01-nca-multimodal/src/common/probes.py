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
                tol: float = 1e-6, poke: float = 1.0) -> dict:
    """Poke one cell. Measure how the influence spreads - without picking a threshold.

    THE FIRST VERSION OF THIS WAS WRONG AND THE CORRECTION IS THE POINT.

    It reported "fraction of the lattice affected", counting a cell as reached when the
    difference exceeded an absolute 1e-6. Every architecture with a global branch scored
    exactly 1.000 on every dataset, which is what prompted a closer look. A tolerance sweep
    settled it: at 1e-3 the global architectures reach 0.047 of the lattice - exactly what
    the purely local one reaches - and at 1e-6 they reach 1.000. The number was a property
    of the threshold, not of the architecture. That is the same failure as R2b in
    00-replications, where the complexity statistics turned out to be reading the sampling
    rate.

    A null control cleared the other suspicion: with a poke of size 0 the measured
    difference is exactly 0, so the probe was never reading floating-point noise.

    What is reported now needs no threshold:

      lightcone_leak   the share of total influence MASS sitting outside the local light
                       cone (distance > step). For a rule that only talks to its immediate
                       neighbours this is EXACTLY zero, by construction, at any tolerance -
                       which is what makes it a trustworthy measure rather than a tuned one.
      decay_per_cell   how many orders of magnitude the influence loses per cell of
                       distance. A local rule loses about 2.5, so its reach is short for
                       reasons no threshold choice can change.
      reach_curve      the influence profile itself, so the raw shape is inspectable.

    The old threshold-based fields are still returned, marked as such, because the earlier
    run used them and the difference between the two is itself the finding.
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
        b[idx] = b[idx] + poke                      # the poke (0.0 gives the null control)
        n_cells = int(np.prod(spatial))
        coords = np.stack(np.meshgrid(*[np.arange(d) for d in spatial], indexing="ij"), -1)
        # distance on a torus, because the lattice wraps - which is the whole reason the
        # boundary never blows up here
        delta = np.abs(coords - np.array(centre))
        delta = np.minimum(delta, np.array(spatial) - delta)
        dist = delta.max(-1) if nd > 1 else delta[..., 0]

        frac, radius, leak, profiles = [], [], [], []
        for k in range(steps):
            a, b = model.step(a), model.step(b)
            diff = (a - b).abs().sum(1)[0].double().numpy()

            # threshold-free: how much of the influence is outside the local light cone
            total = float(diff.sum())
            outside = float(diff[dist > (k + 1)].sum())
            leak.append(outside / total if total > 0 else 0.0)

            # the profile, so the raw shape can be inspected rather than trusted
            profiles.append([float(diff[dist == r].mean()) if (dist == r).any() else 0.0
                             for r in range(int(dist.max()) + 1)])

            # kept only so the earlier, threshold-dependent numbers stay comparable
            hit = diff > max(tol, tol * float(np.abs(diff).max()))
            frac.append(float(hit.sum()) / n_cells)
            radius.append(float(dist[hit].max()) if hit.any() else 0.0)

    prof = np.array(profiles[-1])
    prof = prof / prof[0] if prof[0] > 0 else prof
    live = np.where(prof > 0)[0]
    if len(live) > 2:
        r = live[1:min(len(live), 6)]
        decay = float(-np.polyfit(r, np.log10(prof[r]), 1)[0])   # decades lost per cell
    else:
        decay = float("nan")

    return {"frac_affected": frac, "radius": radius, "n_cells": n_cells,
            "max_radius": float(dist.max()),
            "lightcone_leak": leak, "decay_per_cell": decay,
            "reach_curve": profiles[-1]}


def propagation_summary(prop: dict) -> dict:
    """The headline numbers. The threshold-free ones come first because they are the real ones."""
    frac = prop["frac_affected"]
    leak = prop.get("lightcone_leak", [])
    reach = next((i + 1 for i, f in enumerate(frac) if f > 0.99), None)
    return {
        # threshold-free - trust these
        "lightcone_leak_step1": leak[0] if leak else float("nan"),
        "lightcone_leak_final": leak[-1] if leak else float("nan"),
        "decay_per_cell": prop.get("decay_per_cell", float("nan")),
        # threshold-dependent - kept for comparability with the first run, NOT to be
        # reported as a property of the architecture. See the docstring above.
        "frac_affected_step1__thresholded": frac[0] if frac else float("nan"),
        "radius_step1__thresholded": prop["radius"][0] if prop["radius"] else float("nan"),
        "steps_to_cover__thresholded": reach if reach is not None else float("nan"),
        "frac_affected_final__thresholded": frac[-1] if frac else float("nan"),
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

    # NULL CONTROL: poke nothing, and nothing may light up. This is what rules out the
    # probe reading floating-point noise, and it must run before anything else is believed.
    for name in M.ARCHITECTURES:
        mnull = wake(M.build(name, spatial, 1, scale=0.5))
        z = propagation(mnull, spatial, steps=3, poke=0.0)
        assert max(z["frac_affected"]) == 0.0, (
            f"{name}: a poke of zero moved {max(z['frac_affected']):.3f} of the lattice - "
            f"the probe is reading numerical noise")

    local = wake(M.build("A1_local", spatial, 1, scale=0.5))
    p_local = propagation(local, spatial, steps=6)

    # A purely local rule must leak EXACTLY zero influence outside its light cone, at any
    # step. This is the measure the reported numbers rest on, so it is asserted, not assumed.
    assert max(p_local["lightcone_leak"]) == 0.0, (
        f"a local rule leaked outside its light cone: {p_local['lightcone_leak']}")

    # TOLERANCE SWEEP. The first version of this self-check asserted that a local rule's
    # thresholded reach is invariant to the tolerance. It is not, and the assertion failed
    # on a differently-seeded model (2 cells at 1e-3 against 6 at 1e-6). That is worth
    # keeping as a recorded fact rather than a fixed test: the thresholded reach moves with
    # the tolerance for EVERY architecture, local ones included, because it depends on how
    # fast that particular model's influence decays. It is not a property of the wiring.
    #
    # What IS a property of the wiring is the light cone, and it holds at any tolerance:
    # a rule that only talks to its neighbours cannot move influence further than one cell
    # per step, however loosely you choose to look.
    for tol in (1e-2, 1e-3, 1e-6, 1e-10):
        pr = propagation(local, spatial, steps=6, tol=tol)
        for k, rad in enumerate(pr["radius"]):
            assert rad <= k + 1, (
                f"a local rule reached distance {rad} in {k + 1} steps at tol={tol} - "
                f"that is outside its light cone and impossible")
        assert max(pr["lightcone_leak"]) == 0.0, f"local leak at tol={tol}"

    r = p_local["radius"]
    assert r[0] <= 1.5, f"a local rule must reach ~1 cell in one step, got {r[0]}"
    assert r[-1] <= 6.5, f"a local rule cannot outrun one cell per step, got {r}"
    assert r[-1] > r[0], f"a local rule must still spread over time, got {r}"

    for name in ("A2_pooled", "A3_attentive", "A4_interleaved"):
        g = wake(M.build(name, spatial, 1, scale=0.5))
        pg = propagation(g, spatial, steps=3)
        # The honest claim: a global branch puts SOME influence outside the light cone
        # immediately. Not "most of the lattice" - that was the thresholded number, and it
        # was an artefact. The real share is well under a percent.
        assert pg["lightcone_leak"][0] > 0, (
            f"{name}: a global branch must place some influence outside the light cone, "
            f"got {pg['lightcone_leak'][0]}")

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
    print(f"   A1 influence lost per cell of distance: {p_local['decay_per_cell']:.2f} decades")
    print(f"   A1 influence outside its light cone: {max(p_local['lightcone_leak']):.1e}  (exactly 0)")
    print(f"   A2 influence outside its light cone on step 1: "
          f"{propagation(a2, spatial, steps=1)['lightcone_leak'][0]:.2e}")
    print(f"   A2 global share of update: {c2['global_share_mean']:.2e} (untrained: near zero, as it should be)")
    print(f"   share as the branch is amplified 1x/10x/100x: "
          f"{shares[0]:.2e} -> {shares[1]:.2e} -> {shares[2]:.2e}")
    print("self-check passed")


if __name__ == "__main__":
    demo()
