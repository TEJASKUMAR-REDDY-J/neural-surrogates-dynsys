"""Three neural cellular automata, differing only in how much of the lattice a cell can see.

All three share the same body: a per-cell MLP of widths 32-64-128-64-32 that reads a
description of the cell's situation and proposes a small change to that cell's state. The
whole lattice is updated at once, and then again, and again. Nothing is ever "run" in the
usual sense - the answer is what the lattice settles into.

    A1  LocalNCA        a cell sees its immediate neighbours. Nothing else.
    A2  PooledGlobal    a cell sees its neighbours, plus a summary of the entire lattice.
    A3  AttentiveGlobal a cell sees its neighbours, plus a summary it queries for itself.

A2 and A3 are the "parallel branch that looks at all the data". A2's branch is cheap and
blunt: mean, max and spread over every cell, pushed through a small network and handed to
everybody. A3's is selective: the lattice is squeezed to at most 64 tokens, those tokens
attend to each other, and each cell reads back the token nearest to it. A2 gives every cell
the same global sentence; A3 lets different cells hear different parts of it.

Everything is shape-agnostic: 1-D lattices and 2-D grids run through identical code, and a
lattice whose neighbours mean nothing (a row of tabular features) is a legitimate input. It
should do badly on A1. That is the point of including it.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MLP_WIDTHS = (32, 64, 128, 64, 32)      # as specified
MAX_TOKENS = 64                          # cap on A3's attention, so cost never blows up

# The specified body at full width costs ~23k parameters and the budget asked for is
# 10-15k, so the shape (1:2:4:2:1) is kept and the scale is a knob. scale 0.7 with a
# 12-channel state lands at ~13k; scale 1.0 is the literal specification.
def widths(scale: float = 1.0) -> tuple:
    return tuple(max(4, int(round(w * scale))) for w in MLP_WIDTHS)


# ---------------------------------------------------------------------------------------
# shape-agnostic helpers: one code path for lines and grids
# ---------------------------------------------------------------------------------------

def _conv(nd: int, cin: int, cout: int, k: int = 1, **kw) -> nn.Module:
    return (nn.Conv1d if nd == 1 else nn.Conv2d)(cin, cout, k, **kw)


def _neighbourhood(x: torch.Tensor, nd: int) -> torch.Tensor:
    """What a cell can see locally: itself, its gradients, and its Laplacian.

    Fixed filters, not learned - this is the NCA convention, and it keeps the learned
    parameters entirely inside the per-cell MLP so the three architectures differ only in
    the global branch.
    """
    C = x.shape[1]
    if nd == 1:
        k = x.new_tensor([[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]])          # d/dx, d2/dx2
        w = k[:, None, :].repeat(C, 1, 1)                                # (2C,1,3)
        y = F.conv1d(F.pad(x, (1, 1), mode="circular"), w, groups=C)
    else:
        gx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        lap = x.new_tensor([[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        k = torch.stack([gx, gx.T, lap])                                 # (3,3,3)
        w = k[:, None, :, :].repeat(C, 1, 1, 1)                          # (3C,1,3,3)
        y = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), w, groups=C)
    return torch.cat([x, y], dim=1)


def _perception_mult(nd: int) -> int:
    return 3 if nd == 1 else 4          # identity + (2 or 3) fixed filters


# ---------------------------------------------------------------------------------------
# the shared body
# ---------------------------------------------------------------------------------------

class CellMLP(nn.Module):
    """32-64-128-64-32, applied identically at every cell. The rule.

    The last layer is zero-initialised so an untrained automaton does nothing at all. That
    is what makes it safe to iterate: the model starts as the identity and has to earn every
    change it makes.
    """

    def __init__(self, nd: int, c_in: int, c_out: int, scale: float = 1.0):
        super().__init__()
        layers, prev = [], c_in
        for w in widths(scale):
            layers += [_conv(nd, prev, w), nn.GELU()]
            prev = w
        head = _conv(nd, prev, c_out)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _BaseNCA(nn.Module):
    """Shared machinery: lift to a hidden state, iterate the rule, read the answer back."""

    def __init__(self, spatial: tuple, c_data: int, c_state: int = 16, fire_rate: float = 0.5,
                 scale: float = 1.0):
        super().__init__()
        self.nd = len(spatial)
        self.spatial = spatial
        self.c_data = c_data
        self.c_state = max(c_state, c_data + 2)
        self.fire_rate = fire_rate
        self.scale = scale
        # Multiplies whatever the global branch contributes. Setting it to 0 at evaluation
        # time silences that branch without touching a weight, which is how the local and
        # global contributions are separated in probes.py. A1 has no global branch, so for
        # it this is inert - and that is the control.
        self.global_gain = 1.0

    def has_global(self) -> bool:
        return getattr(self, "_has_global", False)

    def seed(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Initial lattice: what we can see, what we know is missing, and blank scratch space."""
        B = obs.shape[0]
        pad = self.c_state - self.c_data - 1
        blank = obs.new_zeros((B, pad, *obs.shape[2:]))
        return torch.cat([obs, mask, blank], dim=1)

    def rule(self, s: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def step(self, s: torch.Tensor) -> torch.Tensor:
        ds = self.rule(s)
        if self.fire_rate < 1.0:
            keep = (torch.rand_like(s[:, :1]) < self.fire_rate).to(s.dtype)
            ds = ds * keep                       # asynchronous update, as in the NCA literature
        return s + ds

    def forward(self, obs, mask, steps: int, keep_all: bool = False):
        s = self.seed(obs, mask)
        outs = []
        for _ in range(steps):
            s = self.step(s)
            if keep_all:
                outs.append(s[:, : self.c_data])
        return (torch.stack(outs, 1) if keep_all else s[:, : self.c_data])

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------------------
# A1 - local only
# ---------------------------------------------------------------------------------------

class LocalNCA(_BaseNCA):
    """A cell sees its immediate neighbours and nothing else. The classic NCA."""

    arch = "A1_local"

    def __init__(self, spatial, c_data, c_state: int = 16, scale: float = 1.0, **kw):
        super().__init__(spatial, c_data, c_state, scale=scale, **kw)
        self._has_global = False
        self.mlp = CellMLP(self.nd, self.c_state * _perception_mult(self.nd),
                           self.c_state, scale)

    def rule(self, s):
        return self.mlp(_neighbourhood(s, self.nd))


# ---------------------------------------------------------------------------------------
# A2 - local + one global summary, broadcast to everyone
# ---------------------------------------------------------------------------------------

class PooledGlobalNCA(_BaseNCA):
    """Adds a parallel branch that reads the whole lattice and tells every cell the same thing.

    Mean, max and spread over all cells -> a small network -> a context vector handed
    identically to every cell. Cheap, and enough for anything that depends on a global
    property (overall brightness, total energy, which regime the system is in).
    """

    arch = "A2_pooled"

    def __init__(self, spatial, c_data, c_state: int = 16, c_ctx: int = 32,
                 scale: float = 1.0, **kw):
        super().__init__(spatial, c_data, c_state, scale=scale, **kw)
        self._has_global = True
        self.c_ctx = c_ctx
        h = max(8, int(round(64 * scale)))
        self.global_net = nn.Sequential(
            nn.Linear(self.c_state * 3, h), nn.GELU(), nn.Linear(h, c_ctx), nn.GELU(),
        )
        self.mlp = CellMLP(
            self.nd, self.c_state * _perception_mult(self.nd) + c_ctx, self.c_state, scale
        )

    def rule(self, s):
        dims = tuple(range(2, s.ndim))
        stats = torch.cat([s.mean(dims), s.amax(dims), s.std(dims)], dim=1)
        ctx = self.global_net(stats)                                    # (B, c_ctx)
        ctx = ctx.reshape(ctx.shape[0], self.c_ctx, *([1] * self.nd)).expand(
            -1, -1, *s.shape[2:]
        )
        return self.mlp(torch.cat([_neighbourhood(s, self.nd),
                                   self.global_gain * ctx], dim=1))


# ---------------------------------------------------------------------------------------
# A3 - local + a global summary each cell queries for itself
# ---------------------------------------------------------------------------------------

class AttentiveGlobalNCA(_BaseNCA):
    """Adds a parallel branch that lets different cells hear different global information.

    The lattice is squeezed to at most 64 tokens, those tokens attend to one another, and
    each cell reads back the token covering its own region. So the global context is
    position-dependent rather than a single broadcast sentence, and the cost is fixed at
    64x64 attention no matter how large the lattice gets.
    """

    arch = "A3_attentive"

    def __init__(self, spatial, c_data, c_state: int = 16, c_ctx: int = 32,
                 n_heads: int = 4, scale: float = 1.0, **kw):
        super().__init__(spatial, c_data, c_state, scale=scale, **kw)
        self._has_global = True
        self.c_ctx = c_ctx
        self.tok = nn.Linear(self.c_state, c_ctx)
        self.attn = nn.MultiheadAttention(c_ctx, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(c_ctx)
        self.mlp = CellMLP(
            self.nd, self.c_state * _perception_mult(self.nd) + c_ctx, self.c_state, scale
        )
        # token grid: at most MAX_TOKENS in total, laid out over the lattice
        if self.nd == 1:
            self.tgrid = (min(MAX_TOKENS, spatial[0]),)
        else:
            side = max(1, int(math.isqrt(MAX_TOKENS)))
            self.tgrid = (min(side, spatial[0]), min(side, spatial[1]))

    def rule(self, s):
        pool = F.adaptive_avg_pool1d if self.nd == 1 else F.adaptive_avg_pool2d
        up = F.interpolate
        t = pool(s, self.tgrid)                                         # (B, C, *tgrid)
        B, C = t.shape[:2]
        seq = t.reshape(B, C, -1).transpose(1, 2)                       # (B, T, C)
        seq = self.tok(seq)
        a, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm(seq + a)
        ctx = seq.transpose(1, 2).reshape(B, self.c_ctx, *self.tgrid)
        ctx = up(ctx, size=s.shape[2:], mode="nearest")                 # back to every cell
        return self.mlp(torch.cat([_neighbourhood(s, self.nd),
                                   self.global_gain * ctx], dim=1))


# ---------------------------------------------------------------------------------------
# A4 - local and global alternating, rather than running side by side
# ---------------------------------------------------------------------------------------

class _GlobalMix(nn.Module):
    """One global mixing step: squeeze the lattice to tokens, let the tokens talk, put it back.

    Deliberately not attention. The tokens are mixed by a single learned matrix over token
    positions - the token-mixing idea from MLP-Mixer - because it is cheap, it has no
    softmax to saturate, and it makes the global path a plain linear operator whose
    contribution is easy to measure. The output layer is zero-initialised, so at the start
    of training the global path contributes exactly nothing and has to earn its way in.
    """

    def __init__(self, nd: int, c: int, tokens: tuple, c_g: int = 16):
        super().__init__()
        self.nd, self.tokens = nd, tokens
        T = int(np.prod(tokens))
        self.down = _conv(nd, c, c_g)
        self.token_mix = nn.Linear(T, T)
        self.up = _conv(nd, c_g, c)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        pool = F.adaptive_avg_pool1d if self.nd == 1 else F.adaptive_avg_pool2d
        t = self.down(pool(h, self.tokens))                    # (B, c_g, *tokens)
        B, C = t.shape[:2]
        flat = t.reshape(B, C, -1)
        t = self.token_mix(flat).reshape(B, C, *self.tokens)   # every token sees every token
        g = self.up(t)
        return F.interpolate(g, size=h.shape[2:], mode="nearest")


class InterleavedNCA(_BaseNCA):
    """Local, then global, then local, then global - alternating instead of in parallel.

    A2 and A3 hand a cell its global context once, at the input, and then let a stack of
    per-cell layers work on it. This one interleaves: a couple of local layers, a global
    mixing step, a genuinely local 3x3 hop, another global mixing step, then the rest. That
    is the arrangement a diffusion UNet uses - convolution blocks with global attention
    inserted at intervals - and it is a different hypothesis about *where* global
    information is useful, not merely how much of it there is.

    The extra 3x3 layer is depthwise-separable, so the second local stage buys a real extra
    hop of propagation for very few parameters.
    """

    arch = "A4_interleaved"

    def __init__(self, spatial, c_data, c_state: int = 16, c_g: int = 16,
                 scale: float = 1.0, **kw):
        super().__init__(spatial, c_data, c_state, scale=scale, **kw)
        self._has_global = True
        nd = self.nd
        w = widths(scale)
        c_in = self.c_state * _perception_mult(nd)

        if nd == 1:
            tok = (min(16, spatial[0]),)
        else:
            side = max(1, min(4, min(spatial)))
            tok = (side, side)
        self.tokens = tok

        self.local1 = nn.Sequential(_conv(nd, c_in, w[0]), nn.GELU(),
                                    _conv(nd, w[0], w[1]), nn.GELU())
        self.gmix1 = _GlobalMix(nd, w[1], tok, c_g)
        # depthwise 3x3 (one more hop of *local* propagation) then pointwise
        pad = 1
        self.local2 = nn.Sequential(
            _conv(nd, w[1], w[1], k=3, padding=pad, padding_mode="circular", groups=w[1]),
            _conv(nd, w[1], w[2]), nn.GELU())
        self.gmix2 = _GlobalMix(nd, w[2], tok, c_g)
        self.local3 = nn.Sequential(_conv(nd, w[2], w[3]), nn.GELU(),
                                    _conv(nd, w[3], w[4]), nn.GELU())
        head = _conv(nd, w[4], self.c_state)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.head = head

    def rule(self, s):
        h = self.local1(_neighbourhood(s, self.nd))
        h = h + self.global_gain * self.gmix1(h)
        h = self.local2(h)
        h = h + self.global_gain * self.gmix2(h)
        return self.head(self.local3(h))


ARCHITECTURES = {
    "A1_local": LocalNCA,
    "A2_pooled": PooledGlobalNCA,
    "A3_attentive": AttentiveGlobalNCA,
    "A4_interleaved": InterleavedNCA,
}


def _all() -> dict:
    """A1-A4 plus the layer-family automata in models_v2, imported lazily to avoid a cycle."""
    out = dict(ARCHITECTURES)
    try:
        from src.common.models_v2 import NEW_ARCHITECTURES
        out.update(NEW_ARCHITECTURES)
    except ImportError:
        pass
    return out


def build(arch: str, spatial: tuple, c_data: int, **kw) -> _BaseNCA:
    return _all()[arch](tuple(spatial), c_data, **kw)


def scale_for_budget(arch: str, spatial: tuple, c_data: int, target: int = 13_000,
                     lo: float = 0.10, hi: float = 5.0, **kw) -> float:
    """Find the width scale that puts this architecture closest to `target` parameters.

    The global branches cost a fixed ~5k on top of the shared body, so a single scale
    applied to all four would leave the local-only one materially smaller - and then any
    difference between them would be capacity, which is the confound the whole comparison
    exists to avoid. Sizing each architecture to the same budget separately is what makes
    the comparison about *shape*.
    """
    for _ in range(40):
        mid = (lo + hi) / 2
        n = build(arch, spatial, c_data, scale=mid, **kw).n_params
        if n < target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def demo() -> None:
    """Self-check: every architecture runs on every lattice shape, and starts as the identity."""
    torch.manual_seed(0)
    for spatial in [(64,), (3,), (8, 8), (16, 16), (30,)]:
        for name in ARCHITECTURES:
            m = build(name, spatial, 1)
            obs = torch.rand(4, 1, *spatial)
            mask = (torch.rand(4, 1, *spatial) > 0.3).float()
            m.eval()
            with torch.no_grad():
                y0 = m(obs, mask, steps=1)
                y8 = m(obs, mask, steps=8)
            assert y0.shape == obs.shape, f"{name}{spatial}: {y0.shape} != {obs.shape}"
            assert torch.isfinite(y8).all(), f"{name}{spatial}: non-finite after 8 steps"
            # zero-initialised head => an untrained automaton must not move at all
            assert torch.allclose(y8, obs, atol=1e-6), f"{name}{spatial}: untrained rule moved"
            # gradients must reach the parameters
            m.train()
            loss = m(obs, mask, steps=3).square().mean()
            loss.backward()
            g = [p.grad for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
            assert g, f"{name}{spatial}: no gradient reached any parameter"

    for name, cls in ARCHITECTURES.items():
        m = build(name, (64,), 1)
        print(f"   {name:14s} {m.n_params:>7,} params on a 64-cell line")
    m2 = build("A3_attentive", (16, 16), 1)
    print(f"   A3 on a 16x16 grid: {m2.n_params:,} params, tokens={m2.tgrid}")
    print("   budget-matched to 13k:")
    for name in ARCHITECTURES:
        sc = scale_for_budget(name, (64,), 1, 13_000)
        m = build(name, (64,), 1, scale=sc)
        assert 11_500 <= m.n_params <= 14_500, f"{name} missed budget: {m.n_params}"
        print(f"      {name:16s} scale {sc:5.3f} -> {m.n_params:>7,} params")
    print("self-check passed")


if __name__ == "__main__":
    demo()
