# B19 (#568) — what a cell knows against what it can say

**Both, and the second half is learned.** A cell genuinely has little to say — and
the restriction maps, which *could* say what little there is and which a random map
of their own shape and band *does* say, instead level every cell's emission to a
constant that is uncorrelated with what the cell holds.

Three findings, in the order they constrain each other:

1. **The state is nearly rank one.** At horizon a predicting cell's own node stalk
   turns through a median of **1.318 directions** (`p = 8`, centred, 20k, seed 42)
   out of 32 available and 24 its mask exposes. Uncentred it reads **1.000** and the
   question cannot be asked at all, so [B18](https://github.com/NGL321/patchworks/issues/567)'s
   correction is load-bearing here rather than decorative.
2. **Transport discards nothing.** A cell's interior maps *collectively span its
   entire exposed block* — `rank(interior_rowspace) = k_v` at **150 of 150 cells in
   all three arms** — so the stacked map out of a cell is **injective on everything
   the mask exposes**. There is no direction a cell holds in its exposed block that
   its neighbourhood cannot read. The clean loss measure, `readable/state` (same
   ambient, an orthogonal projector and nothing else), reads **0.998**.
3. **But the learned maps are not faithful messengers of *how much* a cell has.**
   Against a **shape-, mask- and band-matched Haar control** — an independent
   partial isometry per edge, the same widths, the same `k_v`, drawn at random —
   the control tracks state rank across cells at **r = +0.974**, while the learned
   maps track it at **r = −0.046**. Over state-rank quartiles the state doubles
   (1.054 → 2.167) and the random map follows it (1.059 → 2.003) while the learned
   emission is flat (1.603 → 1.555).

So *the specialist* is the right reading of the typical cell, and *the bottleneck*
is wrong as stated — nothing is lost, because nothing can be. What is real, and what
neither horn anticipated, is a third thing: **the emitted spectrum is a learned
constant rather than a readout of the cell.** A rich cell cannot say that it is
rich, though the information to reconstruct it is formally still on the wire.

## The instrument

`b19_state_emitted.py`. Per predicting cell, over one 1000-tick window
(`T0`'s own `WINDOW`) ending at each checkpoint:

| quantity | what it is |
| --- | --- |
| `state` | participation ratio of the node-stalk stream `h_c`, split by mask block exactly as `T0/run.py:226-229` splits it |
| `readable` | the same stream projected by `Blocks.interior_rowspace` (`T0/excitation.py:111`) — the projector onto what a cell's interior neighbours can read. **Unweighted**, so it isolates *which directions survive* from *how they are scaled* |
| `emitted` | participation ratio of the concatenation of what the cell puts on **all** of its edge ends, `[F_e h_c]_e` |

so that the path from state to edge factors in two:

```
state --(subspace selection)--> readable --(singular-value weighting)--> emitted
```

Emission is recomputed as the engine computes it — `tick.py:827-828`'s
`maps.restrict(stalks[layout.pair_positions])` — but on the stalks **as they stand
at the observation instant**. `sheaf.broadcast` was not read, because it holds the
previous message-passing phase's `outgoing`, taken before reconciliation subtracted
its displacement: it is half a tick away from the state recorded beside it. Six
checks in `check_568.py` pin the gather to the engine's own (max |diff| 3.81e-06 at
scale 2.61e+01, float32 `bmm` noise), the pad to exact zeros, the interior mask to
the edge kinds, the rank ceilings, and `interior_rowspace` to being a symmetric
idempotent with no leak into the private block.

**Both moments throughout**, per [B18](https://github.com/NGL321/patchworks/issues/567):
uncentred, every state reading here is **1.000** at horizon and the question is
unanswerable; the centred readings below are the reading.

## 1. State against emitted, per cell, over one window

`p = 8`, seed 42, 150 predicting cells, medians over cells:

| ticks | window | state | private | exposed | readable | emitted | emitted (interior) | e/s | r/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 100 | 2.589 | 2.322 | 2.506 | 2.508 | 2.497 | 2.278 | 0.987 | 0.960 |
| 300 | 200 | 1.340 | 1.314 | 1.290 | 1.294 | 2.204 | 1.937 | 1.474 | 0.995 |
| 1 000 | 700 | 1.143 | 1.158 | 1.138 | 1.135 | 1.325 | 1.236 | 1.066 | 0.996 |
| 3 000 | 1000 | 1.194 | 1.179 | 1.189 | 1.191 | 1.350 | 1.307 | 1.063 | 0.996 |
| 10 000 | 1000 | 1.286 | 1.198 | 1.286 | 1.281 | 1.413 | 1.384 | 1.092 | 0.996 |
| 20 000 | 1000 | **1.318** | 1.229 | 1.310 | **1.288** | **1.433** | 1.397 | 1.023 | **0.998** |

Uncentred, the same run reads state **1.000** and emitted **1.000** at 20k, with
`mean_share` **1.0000** — B18's correction is load-bearing here, not decorative.

The state-side trajectory independently reproduces B18 on a differently scoped
population (all 150 predicting cells over a trailing window, against its chain
relay cells): **2.589 / 1.340 / 1.143** at 100 / 300 / 1000 against its **2.52 /
1.29 / 1.14**, uncentred **1.079 / 1.002 / 1.002** against its **1.08 / 1.002 /
1.002**.

### `emitted / state` is not a loss fraction, and must not be read as one

The headline ratio in the table is **1.023** — emission reads *above* state — and
the honest thing to say is that this is a property of the statistic, not evidence
of information gained. Three facts fix its interpretation:

* **The participation ratio is not a rank.** `y = F_c h_c` gives
  `rank cov(y) ≤ rank cov(h)` always, but `(Σσ²)²/Σσ⁴` can *rise* under a linear
  map that flattens a skewed spectrum. Emitted PR exceeding state PR means the
  emission's spectrum is more even, not that it carries more directions.
* **The emission is over-complete.** `Σ_e m_e` has median **38.5** (max 63) against
  a readable block of `k_v` = **24**, so a cell's edge ends are a redundant
  re-encoding of at most 24 directions across 38.5 lane dimensions. State and
  emission do not live in comparable ambients.
* **So the clean loss measure is `readable / state`** — same ambient, same
  spectrum, an orthogonal projector and nothing else. It reads **0.998**.

### And even that 0.2% is a re-weighting, not a discard

`rowspace_568.py` measures `rank(interior_rowspace)` against `k_v`:

| arm | `p` | `k_v` | median `Σ interior m_e` | rank(rowspace) | cells reading the whole exposed block |
| --- | --- | --- | --- | --- | --- |
| reserve | 8 | 24 | 38 | 24 (24–24) | **150 / 150** |
| reserve_p12 | 12 | 20 | 38 | 20 (20–20) | **150 / 150** |
| reserve_p16 | 16 | 16 | 35 | 16 (16–16) | **150 / 150** |

A cell's interior maps *collectively span its entire exposed block*, at every cell
of every arm. The stacked map is therefore **injective on what the mask exposes**:
there is no direction a cell holds in its exposed block that its neighbourhood
cannot read. Whatever the maps do to the spectrum is invertible, and ADR-0032's
band (`σ_min/σ_max` closing to 1.0000 by 100 ticks, per
[B11](https://github.com/NGL321/patchworks/issues/555)) keeps it close to an
isometry per edge.

The `readable/state` gap of 0.998 is thus the *private block's* contribution, not
a transport loss at all.

## 2. By mask block — the richness is exposed, not hidden

Medians at 20k, share of the window's **variance** (and energy) by block:

| arm | `p` | energy private | energy interior | variance private | variance interior |
| --- | --- | --- | --- | --- | --- |
| reserve | 8 | 0.279 | 0.721 | **0.192** | **0.802** |
| reserve_p12 | 12 | 0.421 | 0.579 | 0.279 | 0.718 |
| reserve_p16 | 16 | 0.606 | 0.393 | 0.395 | 0.600 |

At `p = 8` **80% of a cell's variation lives in the exposed block**, and the
private block's own rank (1.229) is no higher than the exposed block's (1.310).
This closes the reading the ticket asked to distinguish: a cell whose richness is
entirely private would be a different finding, and that is not what the surface
does. Even at `p = 16`, where the private block is half the stalk, 60% of the
variance is still exposed.

`p` moves the split monotonically, as it must — it *is* the private width — but it
does not move the conclusion.

## 3. By level, degree and lane width

`p = 8` at 20k, medians within group:

| level | cells | state | emitted | e/s |
| --- | --- | --- | --- | --- |
| 1 (rim) | 70 | 1.125 | 1.434 | 1.107 |
| 2 | 20 | 1.317 | 1.310 | 0.963 |
| 3 | 16 | **2.047** | 1.382 | 0.779 |
| 4 | 14 | 1.932 | 1.451 | 0.709 |
| 5 | 12 | 1.821 | 1.573 | 0.817 |
| 6 | 10 | 1.409 | 1.412 | 1.063 |
| 7 (apex) | 8 | **1.062** | **2.244** | 1.899 |

The state's rank is **non-monotonic and peaks in the middle**: the mid-levels
L3–L5 hold ~1.8–2.0 directions while the rim holds 1.125 and **the apex holds the
least of anything on the dome, 1.062** — while emitting the most, 2.244. The same
shape holds at `p = 12` (L3–L5 1.79/1.76/1.60, apex 1.107) and `p = 16` (1.63/1.32/1.67,
apex 1.074).

This is worth putting beside the user's ruling, recorded on
[B17](https://github.com/NGL321/patchworks/issues/565), that **the apex is not
privileged**: measured on traffic rather than on the operator, the apex is the
*poorest* cell on the dome, and the "rim-to-apex" chain runs toward the narrowest
state in the graph rather than toward an integrating one.

### Emitted rank does not track state rank at all

| arm | `corr(state, emitted)` | Q1 state → emitted | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- | --- |
| `p = 8` | **−0.068** | 1.048 → 1.635 | 1.151 → 1.433 | 1.498 → 1.340 | 2.255 → 1.457 |
| `p = 12` | **+0.044** | 1.080 → 1.586 | 1.201 → 1.252 | 1.471 → 1.348 | 2.065 → 1.466 |
| `p = 16` | **+0.054** | 1.047 → 1.373 | 1.154 → 1.254 | 1.432 → 1.338 | 2.048 → 1.286 |

Across state-rank quartiles the state doubles (1.05 → 2.26) while the emission
stays flat within 1.3–1.6. **Emission has its own operating point of about 1.4,
essentially independent of what the cell holds.** For the richest quartile
`emitted/state` reads **0.644 / 0.711 / 0.642** — those cells really do emit a
flatter spectrum than they hold — while their `readable/state` is still **0.962 /
0.956 / 0.952**, so the divergence is entirely in the weighting and not in the
selection. Whether that flatness is a property of *maps of this shape* or of *these
maps* is exactly what §5's control decides, and the answer is the latter.

By degree and by lane width the answer is the same picture read through a
correlate: `e/s` runs 1.899 (degree 5) to 0.841 (degree 6) to 1.006 (degree 9) at
`p = 8` with no monotone trend, and over `Σ interior m_e` from 14 to 63 it moves
between 0.418 and 2.382 tracking the group's state rank rather than its width
(the `Σ = 38` group has state 3.494 and `e/s` 0.418; the `Σ = 40` group has state
1.118 and `e/s` 2.382). **Width is not the variable**; state rank is, inversely,
which is the ceiling effect above.

## 4. Two `p` settings — and the number that should worry B27

| arm | `p` | state | readable | emitted | composed **operator** ER ([B13](https://github.com/NGL321/patchworks/issues/560)) |
| --- | --- | --- | --- | --- | --- |
| reserve | 8 | 1.318 | 1.288 | 1.433 | 1.4386 |
| reserve_p12 | 12 | 1.353 | 1.323 | 1.398 | **2.1309** |
| reserve_p16 | 16 | 1.254 | 1.255 | 1.324 | **2.9171** |

The ticket asked for at least two `p` settings because `p` moves the exposed block
directly. It does — and it moves **nothing** in the traffic. Across the range over
which B13's composed rim-to-apex operator rank climbs **1.44 → 2.13 → 2.92**, more
than doubling and clearing #540's bar of 2.0 for the first time on this map, the
state a cell actually holds reads **1.318 / 1.353 / 1.254** and what it puts on its
edges reads **1.433 / 1.398 / 1.324**. Neither is even monotone in `p`.

This is [B17](https://github.com/NGL321/patchworks/issues/565)'s thesis measured
rather than argued. `composed_reads` multiplies learned maps and never touches the
stalks, and here is the consequence with numbers on it: **the metric #532's
destination is written against triples while the thing it is supposed to be a proxy
for does not move at all.** `p = 16` is not carrying three directions of signal to
the apex; it is an operator with rank 2.92 carrying a state of rank 1.25.


## 5. The null: the flattening is **learned**, not structural

Two arms were re-run with a **shape-, mask- and band-matched Haar control** computed
on the *same window, same cell, same state stream*: per edge an independent random
partial isometry of that edge's own `m_e × k_v`, so ADR-0032's band holds exactly and
the only thing differing from the surface is **which subspace each lane selects**. A
second control (`haar_scaled`) additionally rescales each edge to the learned map's
own mean singular value, leaving direction as the sole difference. The two agree
throughout, so scale is not the story.

The control validates the instrument first: at `p = 8`, 20k, the random map emits
**1.364** against the readable state's **1.369**. A random isometry neither adds nor
removes spread — exactly as it should — which means any departure the learned maps
show is theirs.

They depart, and the departure is not in the median but in the **correspondence**:

| | `p = 8` | `p = 16` |
| --- | --- | --- |
| `corr(state, emitted)` — **learned** | **−0.046** | **+0.052** |
| `corr(state, emitted)` — Haar control | **+0.974** | **+0.930** |
| `corr(state, emitted)` — Haar, learned scales | +0.923 | +0.906 |

and quartile by quartile at `p = 8`:

| state-rank quartile | state | learned emission | Haar emission |
| --- | --- | --- | --- |
| Q1 | 1.054 | 1.603 | 1.059 |
| Q2 | 1.244 | 1.507 | 1.244 |
| Q3 | 1.557 | 1.533 | 1.501 |
| Q4 | 2.167 | 1.555 | 2.003 |

**A random map of the learned maps' own shape, mask and band is a faithful
messenger** — it reports a cell's rank back almost exactly (r = +0.97). **The learned
maps are not.** They lift the poorest quartile from 1.054 to 1.603 and press the
richest from 2.167 down to 1.555, arriving at a constant near 1.5 that carries no
information about the cell at all. Levelling in both directions, not compression.

This is the ticket's question answered in a way neither of its horns anticipated.
*The bottleneck* is wrong — the map is injective, nothing is discarded, and the
control proves a map of this shape could report the state faithfully. *The
specialist* is right about the typical cell but silent about the tail. What is
actually happening is that **training makes emission stop being a function of the
state**.

It also matches [B16](https://github.com/NGL321/patchworks/issues/564)'s mechanism
from a new direction: the transport rule descends *make the two ends of a lane
agree*, which is an objective on the **edge**, not on the cell — nothing in it asks
the emission to remain a readout of the sender. Where B16 found the collapse to be
junction-carried, this finds the sender-side signature of the same objective.

And the **learned/random ratio is smallest exactly where the operator scores best**:
per-cell median `learned/haar` reads **1.108** at `p = 8` and **1.050** at `p = 16`
at 20k. At `p = 16`, the arm [B13](https://github.com/NGL321/patchworks/issues/560)
scores at 2.92, the learned maps are the *closest to random* of any arm measured
— emitted 1.307 against the control's 1.309. B17 observed that at `p = 16`
[B16](https://github.com/NGL321/patchworks/issues/564)'s coherence never builds and
called it *the better-scoring candidate is the one where training does less*. Here
is the same statement on an instrument that touches the stalks: at `p = 16` the
learned transport is indistinguishable from a random isometry.

## What was *not* measured

Stated as plainly as what was:

* **One seed (42) and one horizon (20k).** No second seed, no 100k arm. B11 carried
  its reserve arm to 100k and found the band still closing; this reading cannot say
  whether 1.318 is a floor or still climbing — it rose 1.143 → 1.194 → 1.286 → 1.318
  over the last four checkpoints and had not flattened.
* **Joint span across chains is not here** — that is
  [B20](https://github.com/NGL321/patchworks/issues/569)'s, and it is the reading
  that decides whether per-cell rank 1.3 is a population code working. Every number
  above is per-cell, aggregated as an order statistic over cells, which is exactly
  the aggregation B17 warned about. **A median over 150 cells cannot see a
  population code**, and this ticket does not claim to.
* **No composed rank, no principal angles, no regional consistency, no ripple test.**
  B16, B21 and B22 own those.
* **No architecture was changed**, per the ticket's own note.
* The **drive** piece is reported by energy share only, not by rank, because a
  one-dimensional stream's participation ratio is degenerate — `T0`'s own caveat,
  inherited.
* `interior_rowspace` and the mask blocks are read **at the checkpoint** while the
  window spans the preceding 1000 ticks, so both drift slightly within the window.
  The window was kept at `T0`'s 1000 partly for this reason; the maps are applied
  per-tick, not assumed constant.

## Limits of the reading itself

* **The null was read at two `p` settings (8 and 16), not three**; `p = 12` has the
  learned reading only.
* **Run-to-run reproducibility is ~3–7% at horizon, and this bounds every number
  above.** The primary and null passes are the *same* arm, seed and tick schedule,
  differing only by an extra read-only diagnostic, and they agree exactly through
  3000 ticks (state 1.194, readable 1.191, emitted 1.350) then diverge: at 20k,
  `p = 8` reads state **1.318** against **1.365** and emitted **1.433** against
  **1.538**. Nothing in the agent's inputs differs, so this is float-level
  perturbation amplified by the dynamics. Differences of a few percent between arms
  in this readout should not be read as signal; the effects claimed here — injectivity
  at 150/150, r = +0.97 against r = −0.05, operator 2.92 against state 1.25 — are one
  to two orders of magnitude clear of it.

* The emitted participation ratio is **not** comparable like-for-like to the state's,
  for the over-completeness reason given above. Everywhere a loss claim is made here
  it rests on `readable/state` and on injectivity, never on `emitted/state`.
* Injectivity is a statement about the **stacked** map out of a cell — what the
  cell's whole neighbourhood can jointly read. A *single* neighbour reads only its
  own lane (`m_e`), which is far narrower. That distinction is real and this reading
  does not price it; whether the neighbourhood ever reassembles what it jointly holds
  is B20's and B22's question, and is the one place the bottleneck horn could still
  return.
* `rowspace_568.py` reads at 400 ticks, not at horizon. Rank of a span is generically
  determined by dimensions (`min(Σ_e m_e, k_v)`, and B1 found the learned subspaces
  indistinguishable from Haar), so it is not expected to move; it was not re-read at
  20k.
