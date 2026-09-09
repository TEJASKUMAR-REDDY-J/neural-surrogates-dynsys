# 03-relevance-dilution — notes

## What this spike is for

Every experiment before this one studied systems where **every measured variable was part of
the dynamics**. That is not the situation a world model is in. A world model sees an enormous
number of signals, most irrelevant to any particular prediction, arriving on different
clocks, and its hard problem is not "fit the dynamics" but "work out which of these things
matter here".

So: hold the true dynamics completely fixed (a 3-variable chaotic system), and only add junk
alongside. Four kinds, because *irrelevant* is not one thing.

| distractor | what it is |
|---|---|
| **white** | independent noise — unstructured, unpredictable |
| **drift** | slow smooth sinusoids — irrelevant but highly predictable |
| **foreign** | a second independent chaotic system, phase-shifted — structured, on its own clock |
| **echo** | lagged copies of the real variables — redundant, not confusing. The control |

---

## 2026-09-09 — 84 cells, zero failures

Four predictions were written down before running. Three held, one was wrong, and the way it
was wrong is the most useful thing here.

### Prediction 1 — lookup collapses faster than the network. **HELD**

Error at one step on the real coordinates, as junk is added (d = 0 to 96):

| | lookup | network |
|---|---|---|
| white | 0.019 → **0.640** (34x worse) | 0.001 → 0.112 |
| foreign | 0.019 → **0.734** (39x worse) | 0.001 → 0.031 |
| echo | 0.019 → 0.035 | 0.001 → 0.005 |

Nearest-neighbour retrieval computes distance over **every** column, so junk columns swamp
it. A network can put near-zero weight on them. Under white and foreign noise the network's
relative advantage holds or improves as junk accumulates.

### Prediction 2 — `drift` hurts more than `white`. **HELD, and dramatically**

The network's error under drift goes 0.001 → **0.896**, worse than the lookup baseline's
0.348. From six drift columns onward **the network loses**, by 2.6x to 3.5x.

### Prediction 3 — `foreign` is the worst case. **WRONG**

Foreign is the *mildest* for the network (0.031 at 96 distractors) and the case where its
relative advantage is **best** (ratio 0.04). Structured junk on its own timescale turns out
to be easy to ignore, presumably because it is easy to recognise as unlike the target.

### Prediction 4 — `echo` barely hurts. **HELD — the kill condition passed**

Redundant columns cost almost nothing: lookup 0.019 → 0.035, network 0.001 → 0.005, across a
32-fold increase in dimension. So **dimensionality alone is not the problem**. If echo had
hurt like white, the whole relevance framing would have been a red herring. It did not.

---

## Why `drift` is catastrophic — and it is not the reason I guessed

The stated hypothesis was "predictable junk tempts a model to spend capacity where error is
cheap". That is **wrong**. The generalisation gap gave it away — test loss over train loss on
the real coordinates:

| distractors | drift | white | foreign | echo |
|---|---|---|---|---|
| 6 | **38,415** | 1.05 | 2.11 | 0.98 |
| 96 | **26,819** | 0.95 | 3.19 | 0.80 |

A gap of four orders of magnitude is not misallocated capacity, it is memorisation. So the
train and test distributions were compared directly, in the junk columns only:

| distractor | mean shift from train to test, as a fraction of the column's own SD |
|---|---|
| white | 0.01 |
| foreign | 0.01 |
| echo | 0.03 |
| **drift** | **0.83** |

**Drift is the only non-stationary distractor, and the only one where the network loses.**
The sinusoids run over the whole series, so the training block and the test block sit at
different phases; the model latches onto values that never recur, and in test those columns
are somewhere it has never been.

**This is the clock-coordinate artefact again.** The very first battery found that about 20%
of the standard benchmark carried monotone time coordinates which made prediction an
extrapolation problem and manufactured a spurious R-squared of 0.59. The same failure has now
reappeared from the opposite direction: introduced deliberately, as a distractor, it is the
single most damaging thing you can add.

The general statement: **the dangerous irrelevant feature is not the noisy one or the
structured one. It is the one whose distribution moves between training and deployment** -
timestamps, counters, slowly-changing covariates, seasonality, anything with a trend.

---

## Can the model tell what matters?

Share of input sensitivity landing on the 3 real columns, against what a model that
*cannot* tell junk from signal would give (3/(3+D)):

| distractors | white | drift | foreign | echo | blind would be |
|---|---|---|---|---|---|
| 6 | 0.557 | 0.253 | 0.835 | 0.520 | 0.333 |
| 24 | 0.239 | 0.067 | 0.503 | 0.226 | 0.111 |
| 96 | **0.014** | 0.064 | 0.186 | 0.114 | 0.030 |

As selectivity (actual over blind, so >1 means it is genuinely selecting):

```
foreign   1.0  1.9  2.5  3.5  4.5  6.4  6.2     rises throughout - easiest to ignore
echo      1.0  1.7  1.6  1.6  2.0  2.2  3.8     steady
white     1.0  1.8  1.7  1.7  2.1  5.6  0.5     works to 48, then COLLAPSES at 96
drift     1.0  1.3  0.8  0.7  0.6  1.8  2.1     below 1 for much of the range
```

Two things worth keeping. White noise is handled well until 96 distractors, at which point
selectivity falls **below** blind - the model is doing worse than ignoring the inputs
uniformly. And drift is anti-selective across most of its range: the model is actively drawn
to the misleading columns.

---

## Delayed generalisation

Per-epoch train and held-out curves were logged specifically to look for this. Epochs the
held-out loss keeps improving *after* the training loss settles:

| distractors | echo | foreign | white | drift |
|---|---|---|---|---|
| 12 | **+210** | +120 | +25 | **-25** |
| 96 | **+235** | +60 | +60 | **-40** |

Echo shows the largest delayed improvement - the held-out loss goes on improving for 200+
epochs after training has settled, which is the delayed-generalisation shape. Drift is
consistently **negative**: its best held-out epoch comes *before* training settles, which is
the signature of overfitting onset, not grokking.

**This is suggestive, not a grokking claim.** Classic grokking is a sharp phase transition on
algorithmic tasks; what is here is a slow tail. What can be said is that the *sign* separates
the conditions cleanly, and it separates them the same way every other metric does.

First-layer effective rank rises with dimension in every condition, but echo uses the fewest
directions at 96 distractors (63 against white's 92), which is what redundancy should do.

---

## What this changes

The rule this project has been circling - "measure coverage, decide whether to train" - is
incomplete. Coverage says how densely the data covers the space. It says nothing about
whether the columns are *relevant*, and nothing about whether they are *stationary*. On this
evidence the second matters more than either:

1. **relevance** decides whether retrieval is viable at all (lookup dies at 34-39x; the
   network does not)
2. **stationarity** decides whether learning is viable at all (the network dies at 896x, and
   loses to the baseline it otherwise beats by 16x)
3. **dimension alone** decides very little (echo, 32x more columns, almost no cost)
