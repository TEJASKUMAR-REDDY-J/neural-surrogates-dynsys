"""Composable local x global automata, with each path independently silenceable.

The eight architectures so far each fused one local mechanism with one global mechanism and
were compared as wholes. That answers "which architecture wins" but not "which *part* is
doing the work", and it cannot tell you whether a good global mechanism is being wasted on a
poor local one.

So this factorises them. Any local block can be paired with any global block, and each path
carries its own gain, so at evaluation time the same trained weights can be run three ways:

    both        the model as trained
    local only  global_gain = 0
    global only local_gain  = 0

Three errors from one fit, which is what makes per-path attribution possible at all. The
local-only number is the one that matters most: if it equals the both number, the global
path is decorative.

    local blocks    fixed     the classic NCA stencil - identity, gradient, Laplacian, then
                              a per-cell MLP. Perception is hand-chosen, only the MLP learns.
                    graph     learned messages per edge from (cell, neighbour, difference).
                              Same one-hop reach, but perception itself is learned.
                    depthwise a learned 3x3 depthwise convolution then a pointwise mix.
                              Cheap, and gives a second hop of reach per step.
                    gru       a gated recurrent cell per site, so state carries explicitly.

    global blocks   none      control. Pure local.
                    pooled    mean/max/spread over the lattice, broadcast to every cell.
                              Permutation-invariant, so it cannot encode position.
                    attention self-attention over at most 64 pooled tokens.
                    spectral  rFFT, keep low modes, complex weights, inverse. Smoothness.
                    tokenmix  a learned matrix over token POSITIONS. The only global block
                              here that contains an explicit position-to-position map -
                              which is the one that survived the shuffled-twin test.

Every global block is zero-initialised, so training starts purely local and the global path
has to earn its way in. Everything is sized to a parameter budget so combinations are
comparable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.models import _BaseNCA, _conv, _neighbourhood, _perception_mult, widths

LOCALS = ("fixed", "graph", "depthwise", "gru")
GLOBALS = ("none", "pooled", "attention", "spectral", "tokenmix")


def _h(scale: float, base: int) -> int:
    return max(4, int(round(base * scale)))


def _tokens(nd: int, spatial: tuple, cap: int = 16) -> tuple:
    if nd == 1:
        return (min(cap, spatial[0]),)
    side = max(1, min(4, min(spatial)))
    return (side, side)


# ---------------------------------------------------------------------------------------
# local blocks: input is the raw state, output is a feature map of width w
# ---------------------------------------------------------------------------------------

class _LocalFixed(nn.Module):
    def __init__(self, nd, c_state, w, scale):
        super().__init__()
        self.nd = nd
        self.net = nn.Sequential(
            _conv(nd, c_state * _perception_mult(nd), _h(scale, 48)), nn.GELU(),
            _conv(nd, _h(scale, 48), w), nn.GELU())

    def forward(self, s):
        return self.net(_neighbourhood(s, self.nd))


class _LocalGraph(nn.Module):
    def __init__(self, nd, c_state, w, scale):
        super().__init__()
        self.nd = nd
        m = _h(scale, 32)
        self.msg = nn.Sequential(_conv(nd, 3 * c_state, m), nn.GELU())
        self.upd = nn.Sequential(_conv(nd, c_state + m, w), nn.GELU())
        self.shifts = [(1,), (-1,)] if nd == 1 else [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def forward(self, s):
        dims = tuple(range(2, s.ndim))
        agg = None
        for sh in self.shifts:
            nb = torch.roll(s, shifts=sh, dims=dims)
            m = self.msg(torch.cat([s, nb, nb - s], dim=1))
            agg = m if agg is None else agg + m
        return self.upd(torch.cat([s, agg], dim=1))


class _LocalDepthwise(nn.Module):
    def __init__(self, nd, c_state, w, scale):
        super().__init__()
        self.nd = nd
        c = c_state * _perception_mult(nd)
        self.net = nn.Sequential(
            _conv(nd, c, c, k=3, padding=1, padding_mode="circular", groups=c),
            _conv(nd, c, w), nn.GELU())

    def forward(self, s):
        return self.net(_neighbourhood(s, self.nd))


class _LocalGRU(nn.Module):
    def __init__(self, nd, c_state, w, scale):
        super().__init__()
        self.nd, self.c = nd, c_state
        e = _h(scale, 40)
        self.enc = nn.Sequential(_conv(nd, c_state * _perception_mult(nd), e), nn.GELU())
        self.gates = _conv(nd, e + c_state, 2 * c_state)
        self.cand = _conv(nd, e + c_state, c_state)
        self.out = nn.Sequential(_conv(nd, c_state, w), nn.GELU())

    def forward(self, s):
        x = self.enc(_neighbourhood(s, self.nd))
        r, z = torch.sigmoid(self.gates(torch.cat([x, s], dim=1))).chunk(2, dim=1)
        n = torch.tanh(self.cand(torch.cat([x, r * s], dim=1)))
        return self.out((1 - z) * n + z * s)


LOCAL_CLS = {"fixed": _LocalFixed, "graph": _LocalGraph,
             "depthwise": _LocalDepthwise, "gru": _LocalGRU}


# ---------------------------------------------------------------------------------------
# global blocks: feature map in, same-shape correction out, zero-initialised
# ---------------------------------------------------------------------------------------

class _GlobalNone(nn.Module):
    def forward(self, h):
        return torch.zeros_like(h)


class _GlobalPooled(nn.Module):
    def __init__(self, nd, w, spatial, scale):
        super().__init__()
        self.nd = nd
        g = _h(scale, 24)
        self.net = nn.Sequential(nn.Linear(3 * w, g), nn.GELU(), nn.Linear(g, w))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        dims = tuple(range(2, h.ndim))
        stats = torch.cat([h.mean(dims), h.amax(dims), h.std(dims)], dim=1)
        g = self.net(stats)
        return g.reshape(*g.shape, *([1] * self.nd)).expand(-1, -1, *h.shape[2:])


class _GlobalAttention(nn.Module):
    def __init__(self, nd, w, spatial, scale, heads=4):
        super().__init__()
        self.nd, self.tok = nd, _tokens(nd, spatial)
        d = max(heads, (_h(scale, 32) // heads) * heads)
        self.down = _conv(nd, w, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.up = _conv(nd, d, w)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, h):
        pool = F.adaptive_avg_pool1d if self.nd == 1 else F.adaptive_avg_pool2d
        t = self.down(pool(h, self.tok))
        B, C = t.shape[:2]
        seq = t.reshape(B, C, -1).transpose(1, 2)
        a, _ = self.attn(seq, seq, seq, need_weights=False)
        t = a.transpose(1, 2).reshape(B, C, *self.tok)
        return F.interpolate(self.up(t), size=h.shape[2:], mode="nearest")


class _GlobalSpectral(nn.Module):
    def __init__(self, nd, w, spatial, scale):
        super().__init__()
        self.nd = nd
        c = _h(scale, 16)
        if nd == 1:
            self.modes = (max(2, min(spatial[0] // 2, _h(scale, 10))),)
        else:
            self.modes = (max(2, min(spatial[0] // 2, _h(scale, 5))),
                          max(2, min(spatial[1] // 2, _h(scale, 5))))
        self.down = _conv(nd, w, c)
        self.weight = nn.Parameter(torch.zeros(2, c, c, *self.modes))
        self.up = _conv(nd, c, w)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, h):
        t = self.down(h)
        wt = torch.complex(self.weight[0], self.weight[1])
        if self.nd == 1:
            n = t.shape[-1]
            f = torch.fft.rfft(t, dim=-1)
            m = min(self.modes[0], f.shape[-1])
            o = torch.zeros_like(f)
            o[..., :m] = torch.einsum("bcx,cdx->bdx", f[..., :m], wt[..., :m])
            t = torch.fft.irfft(o, n=n, dim=-1)
        else:
            n1, n2 = t.shape[-2:]
            f = torch.fft.rfft2(t, dim=(-2, -1))
            m1, m2 = min(self.modes[0], f.shape[-2]), min(self.modes[1], f.shape[-1])
            o = torch.zeros_like(f)
            o[..., :m1, :m2] = torch.einsum("bcxy,cdxy->bdxy", f[..., :m1, :m2],
                                            wt[..., :m1, :m2])
            t = torch.fft.irfft2(o, s=(n1, n2), dim=(-2, -1))
        return self.up(t)


class _GlobalTokenMix(nn.Module):
    """The only global block containing an explicit map between specific positions."""

    def __init__(self, nd, w, spatial, scale):
        super().__init__()
        self.nd, self.tok = nd, _tokens(nd, spatial)
        c = _h(scale, 16)
        T = int(np.prod(self.tok))
        self.down = _conv(nd, w, c)
        self.mix = nn.Linear(T, T)
        self.up = _conv(nd, c, w)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, h):
        pool = F.adaptive_avg_pool1d if self.nd == 1 else F.adaptive_avg_pool2d
        t = self.down(pool(h, self.tok))
        B, C = t.shape[:2]
        t = self.mix(t.reshape(B, C, -1)).reshape(B, C, *self.tok)
        return F.interpolate(self.up(t), size=h.shape[2:], mode="nearest")


GLOBAL_CLS = {"none": _GlobalNone, "pooled": _GlobalPooled, "attention": _GlobalAttention,
              "spectral": _GlobalSpectral, "tokenmix": _GlobalTokenMix}


# ---------------------------------------------------------------------------------------

class HybridNCA(_BaseNCA):
    """One local block, one global block, each with its own gain so either can be silenced."""

    def __init__(self, spatial, c_data, local: str = "fixed", glob: str = "none",
                 c_state: int = 16, scale: float = 1.0, **kw):
        super().__init__(spatial, c_data, c_state, scale=scale, **kw)
        self.local_kind, self.global_kind = local, glob
        self.arch = f"{local}+{glob}"
        self._has_global = glob != "none"
        self.local_gain = 1.0
        w = _h(scale, 64)
        self.w = w
        self.local = LOCAL_CLS[local](self.nd, self.c_state, w, scale)
        self.glob = (_GlobalNone() if glob == "none"
                     else GLOBAL_CLS[glob](self.nd, w, tuple(spatial), scale))
        head = _conv(self.nd, w, self.c_state)
        nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.head = head

    def rule(self, s):
        h = self.local(s)
        g = self.glob(h)
        # Both gains multiply here, so silencing either path needs no retraining and no
        # change of weights - it is the same model asked to run with one hand tied.
        return self.head(self.local_gain * h + self.global_gain * g)


def build_hybrid(local: str, glob: str, spatial, c_data: int, **kw) -> HybridNCA:
    return HybridNCA(tuple(spatial), c_data, local=local, glob=glob, **kw)


def scale_for_budget(local: str, glob: str, spatial, c_data: int, target: int = 13_000,
                     lo: float = 0.05, hi: float = 6.0) -> float:
    for _ in range(40):
        mid = (lo + hi) / 2
        n = build_hybrid(local, glob, spatial, c_data, scale=mid).n_params
        if n < target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def demo() -> None:
    """Self-check: every combination builds, starts as the identity, and both gains bite."""
    torch.manual_seed(0)
    n_ok = 0
    for spatial in [(64,), (8, 8)]:
        for lo in LOCALS:
            for gl in GLOBALS:
                m = build_hybrid(lo, gl, spatial, 1, scale=0.5)
                obs = torch.rand(3, 1, *spatial)
                mask = (torch.rand(3, 1, *spatial) > 0.3).float()
                m.eval()
                with torch.no_grad():
                    y = m(obs, mask, steps=4)
                assert y.shape == obs.shape, f"{lo}+{gl}{spatial}: {y.shape}"
                assert torch.allclose(y, obs, atol=1e-5), (
                    f"{lo}+{gl}{spatial}: untrained rule moved by "
                    f"{(y - obs).abs().max():.2e}; it must start as the identity")
                m.train()
                m(obs, mask, steps=2).square().mean().backward()
                assert any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in m.parameters()), f"{lo}+{gl}: no gradient"
                n_ok += 1

    # after perturbation, silencing a path must actually change the update - otherwise the
    # per-path attribution the whole experiment rests on is measuring nothing
    for gl in ("pooled", "attention", "spectral", "tokenmix"):
        m = build_hybrid("fixed", gl, (64,), 1, scale=0.5)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.1 * torch.randn_like(p))
        s = m.seed(torch.rand(2, 1, 64), torch.ones(2, 1, 64))
        with torch.no_grad():
            both = m.rule(s)
            m.global_gain = 0.0
            loc = m.rule(s)
            m.global_gain, m.local_gain = 1.0, 0.0
            glb = m.rule(s)
            m.local_gain = 1.0
        assert (both - loc).norm() > 1e-6, f"{gl}: silencing global changed nothing"
        assert (both - glb).norm() > 1e-6, f"{gl}: silencing local changed nothing"

    print(f"   {n_ok} combinations x lattice shapes built, identity at init, gradients flow")
    print("   budget-matched to 13k on a 64-cell line:")
    for lo in LOCALS:
        row = []
        for gl in GLOBALS:
            sc = scale_for_budget(lo, gl, (64,), 1)
            row.append(f"{gl}:{build_hybrid(lo, gl, (64,), 1, scale=sc).n_params:>6,}")
        print(f"      {lo:10s} " + "  ".join(row))
    print("self-check passed")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    demo()
