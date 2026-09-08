"""Small MLP surrogates, sized by parameter count rather than by width.

Experiments sweep *capacity*, so the natural knob is total parameters. `mlp_for_budget`
inverts the width->parameter relation so a requested budget lands within a few percent.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Plain tanh MLP. Predicts the increment, so the output is the same size as input."""

    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HorizonMLP(nn.Module):
    """Horizon-conditioned surrogate: predicts x(t+H) - x(t) in one forward pass.

    The horizon enters as two extra inputs, log(H) and H/H_max, both scaled to O(1). This
    is the cheap version of the continuous horizon conditioning in Hybrid Neural World
    Models (arXiv:2605.28317) - one network covers every horizon instead of one network
    per horizon.
    """

    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int = 2, h_max: int = 64):
        super().__init__()
        self.h_max = h_max
        self.body = MLP(in_dim + 2, out_dim, width, depth)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h = h.reshape(-1, 1).float()
        feat = torch.cat([torch.log(h) / math.log(self.h_max), h / self.h_max], dim=1)
        return self.body(torch.cat([x, feat], dim=1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _params_for_width(in_dim: int, out_dim: int, width: int, depth: int) -> int:
    """Closed form: first layer + (depth-1) hidden layers + output layer, with biases."""
    n = (in_dim + 1) * width
    n += (depth - 1) * ((width + 1) * width)
    n += (width + 1) * out_dim
    return n


def width_for_budget(in_dim: int, out_dim: int, budget: int, depth: int = 2) -> int:
    """Smallest width whose parameter count is closest to `budget`. Bisection on width."""
    lo, hi = 1, 4
    while _params_for_width(in_dim, out_dim, hi, depth) < budget:
        hi *= 2
        if hi > 1 << 20:
            break
    best, best_err = lo, float("inf")
    for w in range(max(1, hi // 2 - 1), hi + 2):
        err = abs(_params_for_width(in_dim, out_dim, w, depth) - budget)
        if err < best_err:
            best, best_err = w, err
    return best


def mlp_for_budget(
    in_dim: int, out_dim: int, budget: int, depth: int = 2, seed: int = 0, horizon: bool = False,
    h_max: int = 64,
) -> nn.Module:
    """Build a model with approximately `budget` parameters."""
    torch.manual_seed(seed)
    eff_in = in_dim + 2 if horizon else in_dim
    w = width_for_budget(eff_in, out_dim, budget, depth)
    if horizon:
        return HorizonMLP(in_dim, out_dim, w, depth, h_max=h_max)
    return MLP(in_dim, out_dim, w, depth)


class RefineMLP(nn.Module):
    """A corrector applied over and over to its own output.

    Given where the system is now, a horizon, and a current guess at where it will be, this
    proposes a correction to the guess. Run it once and it is an ordinary direct predictor;
    run it k times, each pass refining the last, and it becomes iterative computation at
    inference time - the mechanism recursive-reasoning models like TRM are built on.

    The interesting question is not whether pass 2 beats pass 1. It is whether the sequence
    keeps improving, settles, or turns around and gets worse - and whether it stays sane
    beyond the number of passes it was trained for.
    """

    def __init__(self, dim: int, width: int, depth: int = 2, h_max: int = 256):
        super().__init__()
        self.h_max = h_max
        self.dim = dim
        # inputs: current state, current guess, guess-minus-state, and two horizon features
        self.body = MLP(3 * dim + 2, dim, width, depth)

    def _feat(self, x, guess, h):
        h = h.reshape(-1, 1).float()
        hf = torch.cat([torch.log(h) / math.log(self.h_max), h / self.h_max], dim=1)
        return torch.cat([x, guess, guess - x, hf], dim=1)

    def step(self, x, guess, h):
        return guess + self.body(self._feat(x, guess, h))

    def forward(self, x, h, passes: int = 1, return_all: bool = False):
        guess = x
        outs = []
        for _ in range(passes):
            guess = self.step(x, guess, h)
            outs.append(guess)
        return outs if return_all else guess


def refine_for_budget(dim: int, budget: int, depth: int = 2, seed: int = 0,
                      h_max: int = 256) -> RefineMLP:
    torch.manual_seed(seed)
    w = width_for_budget(3 * dim + 2, dim, budget, depth)
    return RefineMLP(dim, w, depth, h_max=h_max)


def demo() -> None:
    # closed-form parameter count must match the real model
    for depth in (1, 2, 3):
        for width in (4, 17, 64):
            m = MLP(3, 3, width, depth)
            assert count_params(m) == _params_for_width(3, 3, width, depth), (depth, width)

    # budget targeting: within 10% across three orders of magnitude
    for budget in (1_000, 10_000, 100_000):
        m = mlp_for_budget(3, 3, budget, depth=2)
        got = count_params(m)
        assert abs(got - budget) / budget < 0.10, (budget, got)

    # same seed -> same weights; different seed -> different
    a = mlp_for_budget(3, 3, 1000, seed=0)
    b = mlp_for_budget(3, 3, 1000, seed=0)
    c = mlp_for_budget(3, 3, 1000, seed=1)
    pa = torch.cat([p.flatten() for p in a.parameters()])
    pb = torch.cat([p.flatten() for p in b.parameters()])
    pc = torch.cat([p.flatten() for p in c.parameters()])
    assert torch.allclose(pa, pb) and not torch.allclose(pa, pc)

    # horizon model runs and is sensitive to the horizon input
    hm = mlp_for_budget(3, 3, 5000, horizon=True)
    x = torch.randn(8, 3)
    o1 = hm(x, torch.ones(8))
    o2 = hm(x, torch.full((8,), 32.0))
    assert o1.shape == (8, 3) and not torch.allclose(o1, o2)

    # refinement model: repeated passes must actually change the guess
    rm = refine_for_budget(3, 8000, seed=0)
    xx, hh = torch.randn(6, 3), torch.full((6,), 32.0)
    o1 = rm(xx, hh, passes=1)
    o3 = rm(xx, hh, passes=3)
    allo = rm(xx, hh, passes=3, return_all=True)
    assert o1.shape == (6, 3) and len(allo) == 3
    assert torch.allclose(allo[-1], o3) and not torch.allclose(o1, o3)

    print(
        "models ok - "
        + ", ".join(
            f"{b}->{count_params(mlp_for_budget(3,3,b))}" for b in (1000, 10000, 100000)
        )
    )


if __name__ == "__main__":
    demo()
