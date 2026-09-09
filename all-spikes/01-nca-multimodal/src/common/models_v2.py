"""Four more automata, each built from a different layer family.

The first four differed in how much of the lattice a cell can see. These four differ in
*what kind of machinery* does the seeing, so the comparison moves from "how much context"
to "which inductive bias".

    A5  TransformerNCA   every cell is a token, full self-attention over all of them
    A6  GraphNCA         message passing on the lattice graph, messages learned per edge
    A7  RecurrentNCA     a gated recurrent cell at every site, so memory is explicit
    A8  SpectralNCA      mixing done in Fourier space, FNO-style

They plug into the same base class as A1-A4, so every probe, metric and baseline applies
unchanged, and each is sized to the same parameter budget so differences are about shape
rather than capacity.

Why these four in particular:

  A5 is the honest version of A3. A3 pools the lattice to at most 64 tokens before
     attending, which caps its cost but also blurs it. A5 attends at full cell resolution,
     so if A3 underperforms we can tell whether pooling was the reason.
  A6 replaces the fixed perception filters with learned messages computed per edge from
     (cell, neighbour, difference). Same locality as A1 - one hop per step - so it isolates
     "learned messages" from "more reach".
  A7 gives each cell a GRU. Every other architecture here carries memory only implicitly,
     in whatever the hidden channels happen to hold. This one has gates designed for it, so
     it is the direct test of the memory question - and the memory probe should show it.
  A8 mixes in the frequency domain, which is global in one step like A2-A4 but with a
     strong smoothness prior instead of a summary statistic. It is also the layer in the
     reference diagrams for this spike.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.models import (
    MLP_WIDTHS, _BaseNCA, _conv, _neighbourhood, _perception_mult, widths,
)


def _hidden(scale: float, base: int) -> int:
    return max(4, int(round(base * scale)))


# ---------------------------------------------------------------------------------------
# A5 - a transformer layer as the update rule
# ---------------------------------------------------------------------------------------

class TransformerNCA(_BaseNCA):
    """Every cell is a token; all tokens attend to all tokens, then a per-cell feed-forward.

    This is A3 without the pooling shortcut. A3 squeezes the lattice to at most 64 tokens
    so its cost never grows, which is a sensible engineering choice but it means a cell
    reads a blurred summary of its region. Here attention runs at full resolution, so the
    two together separate "attention helps" from "pooling hurt".

    Cost is quadratic in the number of cells, which is the honest price and is recorded in
    the timings rather than hidden.
    """

    arch = "A5_transformer"

    def __init__(self, spatial, c_data, c_state: int = 16, n_heads: int = 4,
                 scale: float = 1.0, **kw):
        super().__init__(spatial, c_data, c_state, scale=scale, **kw)
        self._has_global = True
        d = _hidden(scale, 40)
        d = max(n_heads, (d // n_heads) * n_heads)        # attention needs divisibility
        self.d = d
        c_in = self.c_state * _perception_mult(self.nd)
        self.tok = _conv(self.nd, c_in, d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        f = _hidden(scale, 64)
        self.ff = nn.Sequential(nn.Linear(d, f), nn.GELU(), nn.Linear(f, d))
        head = _conv(self.nd, d, self.c_state)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.head = head

    def rule(self, s):
        h = self.tok(_neighbourhood(s, self.nd))          # (B, d, *spatial)
        B, d = h.shape[:2]
        seq = h.reshape(B, d, -1).transpose(1, 2)         # (B, N, d)
        a, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm1(seq + self.global_gain * a)      # gain silences ONLY the mixing
        seq = self.norm2(seq + self.ff(seq))
        h = seq.transpose(1, 2).reshape(B, d, *s.shape[2:])
        return self.head(h)


# ---------------------------------------------------------------------------------------
# A6 - message passing on the lattice graph
# ---------------------------------------------------------------------------------------

class GraphNCA(_BaseNCA):
    """Learned messages along lattice edges, aggregated, then a per-node update.

    A1 perceives through fixed filters - identity, gradient, Laplacian - and learns only
    what to do with the result. Here the perception itself is learned: for each neighbour
    the network sees (this cell, that cell, their difference) and computes a message. The
    messages are summed and a second network turns the sum into an update.

    Crucially the locality is identical to A1 - one hop per step, on the same torus - so
    this isolates "learned edge functions" from "greater reach". If A6 beats A1, fixed
    filters were the limitation; if it does not, they were not.
    """

    arch = "A6_graph"

    def __init__(self, spatial, c_data, c_state: int = 16, scale: float = 1.0, **kw):
        super().__init__(spatial, c_data, c_state, scale=scale, **kw)
        self._has_global = False
        c = self.c_state
        m = _hidden(scale, 40)
        h = _hidden(scale, 56)
        self.msg = nn.Sequential(_conv(self.nd, 3 * c, h), nn.GELU(), _conv(self.nd, h, m))
        self.upd = nn.Sequential(_conv(self.nd, c + m, h), nn.GELU(), _conv(self.nd, h, h),
                                 nn.GELU())
        out = _conv(self.nd, h, c)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.out = out
        # unit shifts on the torus: 2 neighbours on a line, 4 on a grid
        self.shifts = ([(1,), (-1,)] if self.nd == 1
                       else [(1, 0), (-1, 0), (0, 1), (0, -1)])

    def rule(self, s):
        agg = None
        for sh in self.shifts:
            dims = tuple(range(2, s.ndim))
            nb = torch.roll(s, shifts=sh, dims=dims)      # roll = periodic, no walls
            m = self.msg(torch.cat([s, nb, nb - s], dim=1))
            agg = m if agg is None else agg + m
        return self.out(self.upd(torch.cat([s, agg], dim=1)))


# ---------------------------------------------------------------------------------------
# A7 - a gated recurrent cell at every site
# ---------------------------------------------------------------------------------------

class RecurrentNCA(_BaseNCA):
    """A GRU per cell, with the cell's own state as the recurrent hidden state.

    Every other automaton here carries memory only implicitly - whatever the hidden
    channels happen to hold from one step to the next. This one has gates built for it: a
    reset gate that decides what to forget and an update gate that decides how much of the
    new candidate to admit.

    That makes it the direct test of the history question. The memory probe wipes the
    scratch channels halfway through a rollout; if explicit gating matters, this is the
    architecture whose score should suffer most when that happens.

    Implemented as 1x1 convolutions rather than nn.GRUCell so the same weights apply at
    every site, which is what makes it a cellular automaton rather than a sequence model.
    """

    arch = "A7_recurrent"

    def __init__(self, spatial, c_data, c_state: int = 16, scale: float = 1.0, **kw):
        super().__init__(spatial, c_data, c_state, scale=scale, **kw)
        self._has_global = False
        c = self.c_state
        c_in = c * _perception_mult(self.nd)
        h = _hidden(scale, 48)
        self.enc = nn.Sequential(_conv(self.nd, c_in, h), nn.GELU())
        self.gates = _conv(self.nd, h + c, 2 * c)         # reset and update, together
        self.cand = _conv(self.nd, h + c, c)
        nn.init.zeros_(self.cand.weight)
        nn.init.zeros_(self.cand.bias)

    def rule(self, s):
        x = self.enc(_neighbourhood(s, self.nd))
        rz = torch.sigmoid(self.gates(torch.cat([x, s], dim=1)))
        r, z = rz.chunk(2, dim=1)
        # The candidate is a residual proposal AROUND the current state: n = s + tanh(...).
        # Written the textbook way, with n = tanh(...) alone, a zero-initialised candidate
        # proposes zero, so the update gate pulls every cell toward zero and the untrained
        # automaton erases its own input - it moved by 0.94 in the self-check. As a residual
        # the zero init means n = s exactly, the delta is exactly zero, and the automaton
        # starts as the identity like every other one here.
        n = torch.tanh(self.cand(torch.cat([x, r * s], dim=1)))
        return (1 - z) * n


# ---------------------------------------------------------------------------------------
# A8 - mixing in the frequency domain
# ---------------------------------------------------------------------------------------

class _SpectralMix(nn.Module):
    """One Fourier layer: transform, keep the low modes, multiply, transform back.

    Global in a single step like A2-A4, but the prior is completely different. A pooled
    summary says "here is one number about everything"; this says "here is how each spatial
    frequency should be rescaled", which is smoothness rather than aggregation.
    """

    def __init__(self, nd: int, c: int, modes: tuple):
        super().__init__()
        self.nd, self.modes = nd, modes
        shape = (c, c) + tuple(modes)
        self.w = nn.Parameter(torch.zeros(2, *shape))     # zero-init: starts as a no-op
        self.scale_ = 1.0 / c

    def forward(self, h):
        w = torch.complex(self.w[0], self.w[1]) * self.scale_
        if self.nd == 1:
            n = h.shape[-1]
            f = torch.fft.rfft(h, dim=-1)
            m = min(self.modes[0], f.shape[-1])
            out = torch.zeros_like(f)
            out[..., :m] = torch.einsum("bcx,cdx->bdx", f[..., :m], w[..., :m])
            return torch.fft.irfft(out, n=n, dim=-1)
        n1, n2 = h.shape[-2:]
        f = torch.fft.rfft2(h, dim=(-2, -1))
        m1 = min(self.modes[0], f.shape[-2])
        m2 = min(self.modes[1], f.shape[-1])
        out = torch.zeros_like(f)
        out[..., :m1, :m2] = torch.einsum(
            "bcxy,cdxy->bdxy", f[..., :m1, :m2], w[..., :m1, :m2])
        return torch.fft.irfft2(out, s=(n1, n2), dim=(-2, -1))


class SpectralNCA(_BaseNCA):
    """FNO-style: a local pointwise path and a spectral path, added together.

    This is the layer in the reference diagrams for this spike. The spectral path is
    zero-initialised, so the automaton begins as purely local and has to earn its global
    component - the same discipline used for the global branches in A2-A4.
    """

    arch = "A8_spectral"

    def __init__(self, spatial, c_data, c_state: int = 16, scale: float = 1.0, **kw):
        super().__init__(spatial, c_data, c_state, scale=scale, **kw)
        self._has_global = True
        w = _hidden(scale, 28)
        c_in = self.c_state * _perception_mult(self.nd)
        if self.nd == 1:
            modes = (max(2, min(spatial[0] // 2, _hidden(scale, 12))),)
        else:
            modes = (max(2, min(spatial[0] // 2, _hidden(scale, 6))),
                     max(2, min(spatial[1] // 2, _hidden(scale, 6))))
        self.modes = modes
        self.lift = _conv(self.nd, c_in, w)
        self.spec = _SpectralMix(self.nd, w, modes)
        self.local = _conv(self.nd, w, w)
        f = _hidden(scale, 48)
        self.ff = nn.Sequential(_conv(self.nd, w, f), nn.GELU())
        head = _conv(self.nd, f, self.c_state)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.head = head

    def rule(self, s):
        h = self.lift(_neighbourhood(s, self.nd))
        h = F.gelu(self.local(h) + self.global_gain * self.spec(h))
        return self.head(self.ff(h))


NEW_ARCHITECTURES = {
    "A5_transformer": TransformerNCA,
    "A6_graph": GraphNCA,
    "A7_recurrent": RecurrentNCA,
    "A8_spectral": SpectralNCA,
}


def demo() -> None:
    """Self-check: every new automaton on every lattice shape, identity at init, gradients."""
    from src.common.models import ARCHITECTURES, build, scale_for_budget

    torch.manual_seed(0)
    for spatial in [(64,), (3,), (8, 8), (16, 16), (30,)]:
        for name in NEW_ARCHITECTURES:
            m = build(name, spatial, 1, scale=0.6)
            obs = torch.rand(3, 1, *spatial)
            mask = (torch.rand(3, 1, *spatial) > 0.3).float()
            m.eval()
            with torch.no_grad():
                y = m(obs, mask, steps=6)
            assert y.shape == obs.shape, f"{name}{spatial}: {y.shape}"
            assert torch.isfinite(y).all(), f"{name}{spatial}: non-finite"
            assert torch.allclose(y, obs, atol=1e-5), (
                f"{name}{spatial}: untrained rule moved by "
                f"{(y - obs).abs().max():.2e} - it must start as the identity")
            m.train()
            m(obs, mask, steps=2).square().mean().backward()
            g = [p for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
            assert g, f"{name}{spatial}: no gradient reached any parameter"

    # A6 must be exactly as local as A1: one hop per step, nothing outside the light cone
    from src.common.probes import propagation
    m6 = build("A6_graph", (64,), 1, scale=0.6)
    with torch.no_grad():
        for p in m6.parameters():
            p.add_(0.05 * torch.randn_like(p))
    pr = propagation(m6, (64,), steps=6)
    assert max(pr["lightcone_leak"]) == 0.0, (
        f"A6 is message passing on the lattice graph and must be strictly local, "
        f"but it leaked {max(pr['lightcone_leak'])} outside its light cone")

    print("   budget-matched to 13k on a 64-cell line:")
    for name in list(ARCHITECTURES) + list(NEW_ARCHITECTURES):
        sc = scale_for_budget(name, (64,), 1, 13_000)
        m = build(name, (64,), 1, scale=sc)
        flag = "" if 11_000 <= m.n_params <= 15_000 else "   <-- OUTSIDE BUDGET"
        print(f"      {name:16s} scale {sc:5.3f} -> {m.n_params:>7,} params{flag}")
    print("self-check passed")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    demo()
