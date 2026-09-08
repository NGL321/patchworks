# B23 — Is the dome earning its constraint, or is the hierarchy imposed and inert?

Reading for [#572](https://github.com/NGL321/patchworks/issues/572), on the map
[Transport whose composed rank exceeds one](https://github.com/NGL321/patchworks/issues/532).

**Instrument:** `prototypes/cold-start/T6/b23_dome.py`, read by `b23_analyse.py`.
**Surface:** `arms.build_arm` — the `reserve` arm (`p = 8`, budget 63) and the `shipped`
arm — seeds 42 and 43, baseline condition, 20,000 ticks, checkpoints on T3's ladder.
Raw JSON beside this file as `572-dome-<arm>-baseline-seed<n>-20000.json`, structure-only
reads as `572-structure-<arm>.json`.

Nothing here edits `src/`. This is a reading and a scoping note; it does not rule on the
dome's fate, which is [B27](https://github.com/NGL321/patchworks/issues/576)'s.

---

## The short answer

**No. On every axis the wager names, a cell's level predicts either nothing or the
opposite of what the dome assumes — and training makes the inversion worse.**

The one strong, reliable correlation with depth in the whole reading runs *down* the
hierarchy: rim cells speak loudest (`emission_gain`, Spearman −0.88 with level). The
quantities abstraction would live in — timescale, state rank, private share, what the
cell can be shown to represent — are flat, weakly inverted, or non-monotone with a break
at exactly the apex.

Three things found on the way that are larger than the question asked:

1. **The apex is faster than the median cell in every run at every checkpoint, and
   training makes it more so.** Apex spectral retention against the graph median: 0.35,
   0.36, 0.49, 0.50 at 100 ticks, falling to **0.12, 0.20, 0.25, 0.27** at 20,000 across
   the four arm-seeds. On `reserve`/s42 that is `tau` 24.2 → **4.4 ticks** against a graph
   median of 36.4 and a p90 of 204.
2. **On the arm this map has been reading since #555, the private-dimension gradient
   does not exist.** The `reserve` mask sets `k_v = n - p` at every predicting cell, so
   `p_v = 8` rim to apex — flat. #548's inversion is the `shipped` arm's story; the
   reserve arm has no gradient at all to invert.
3. **The body stalls by 2,000 ticks in all four runs and is still stalled at 10,000.**
   Peak proprioceptive standard deviation over a 1,000-tick window falls from ~1.7 at 100
   ticks to ~2e-4 at 2,000 and stays there; touch and puck read zero. One run of the four
   (`reserve`/s43) is moving again by 20,000. Corroborated independently by #518's own
   `travel_window` (8.43 over the first 100 ticks, ~0.001–0.01 per window thereafter).
   Most trained readings on this map are therefore taken against a motionless world.

---

## 1. Does level predict anything?

`Spearman(level, X)` over the 150 predicting cells. A dome that forces abstraction
predicts a strong **positive** on timescale, state rank and private share.

| measured | reserve/s42 | reserve/s43 | shipped/s42 | shipped/s43 |
|---|---|---|---|---|
| `tau_e` (stalk autocorrelation) | −0.299\* | −0.067 | −0.134 | −0.275\* |
| `tau_int` (integrated) | −0.341\* | −0.123 | −0.173 | −0.331\* |
| `lag1` (one-tick) | −0.368\* | −0.522\* | −0.232\* | −0.353\* |
| `tau_spectral` (`-1/ln rho`) | +0.178 | +0.301\* | +0.243\* | +0.266\* |
| state rank, centred | +0.263\* | +0.180 | +0.120 | +0.303\* |
| chart rank, centred | +0.268\* | +0.192 | +0.114 | +0.279\* |
| `mean_share` | +0.378\* | +0.566\* | +0.286\* | +0.460\* |
| `emission_gain` | **−0.843**\* | **−0.858**\* | **−0.839**\* | **−0.849**\* |

\* two-sided *p* < 0.01. `p_v` and `k_v` carry no variance on the `reserve` arm and are
reported under *Private share* below.

**Nothing in the first six rows is both strong and stable.** The three autocorrelation
rows are **negative in all four runs** — deeper cells are, if anything, faster. The
spectral row is weakly positive but non-monotone, and §*Timescale* below is where that
matters. The one row that is strong and stable across arms and seeds is `emission_gain`,
and it runs **down** the hierarchy.

### Timescale

**No gradient, and what signal there is runs backwards.** Two independent instruments:

* **Trajectory autocorrelation** (the centred vector autocorrelation of the cell's node
  stalk, `tau_e` = lag at which it crosses `1/e`). **Negative in all four runs** at
  20,000 ticks (−0.07 to −0.30; the one-tick statistic −0.23 to −0.52): deeper cells are,
  weakly, *faster*. Per-level medians are not worth quoting — the population is bimodal,
  some cells near-white and others past 250 ticks, and a median lands on whichever mode
  is heavier, which is why the rank correlation is the statistic reported.
* **Spectral retention** `rho(used)` — `body.py:789`, *"the quantity timescale reads
  (#143)"* — as `tau = -1/ln rho`. Weakly positive with level (+0.17), but the shape is
  what matters, and it is not a ramp:

On `reserve`/s42, per level (`tau` in ticks):

| ticks | `tau` apex | `tau` L1 | `tau` L2–L6 | apex / graph median |
|---|---|---|---|---|
| 100 | 24.2 | 49.8 | 138.2 | 0.36 |
| 1,000 | 29.1 | 44.8 | 96.9 | 0.54 |
| 5,000 | 26.9 | 35.6 | 81.0 | 0.53 |
| 10,000 | 10.1 | 33.8 | 68.9 | 0.23 |
| 20,000 | **4.4** | 28.9 | 61.4 | **0.12** |

`tau` **rises** from the rim to L5 and then **breaks at the apex** — a shape a rank
correlation reads as a weak positive and which is nothing of the kind. On
`reserve`/s43 at 20,000 ticks: 13.6, 32.4, 48.4, 66.1, 72.5, 32.5, **6.5**.

And the break is in every run:

| apex `tau` / graph median | reserve/s42 | reserve/s43 | shipped/s42 | shipped/s43 |
|---|---|---|---|---|
| at 100 ticks | 0.36 | 0.35 | 0.49 | 0.50 |
| at 20,000 ticks | **0.12** | **0.25** | **0.20** | **0.27** |

The apex is the one cell the hierarchical story is loudest about. It is faster than the
median cell in **every run at every checkpoint**, and training roughly halves the ratio.
At 20,000 ticks all eight apex cells sit in the faster half of the graph in all four runs
— on `reserve`/s42, all eight in the fastest 30 of 150.

This is the direction [#271](https://github.com/NGL321/patchworks/issues/271) read on a
different instrument entirely (retention 0.91 at the apex against 0.99 at the rim). Two
instruments, one shape.

### State rank

**Weakly positive, small, and not stable.** Centred participation ratio of the node-stalk
stream rises from ~2.0 at L1 to ~3.0 at L7 at 1,000 ticks, and the whole population
collapses toward 1.1–1.4 by 20,000. Spearman with level at 20,000 runs +0.12 to +0.30
across the four runs, significant in two of them. Against an ambient of `n = 32` this is
a graph of near-rank-one cells at every level, tilted very slightly.

The *uncentred* statistic reads **1.00 at every level at every checkpoint**, which is
[B17](https://github.com/NGL321/patchworks/issues/565)'s warning reproduced exactly: it
would have shown a flat rank-one graph and said nothing about the variation.

`mean_share` is ≥ 0.99 at every level from 1,000 ticks on — the state is almost entirely
a static offset with a thin film of variation on top.

### Private share and lane width

**There is no gradient to predict from on the arm this map reads.** `apply_reserve` sets
`permitted = stalk - p` at every predicting cell, so on the `reserve` arm `p_v = 8` and
`k_v = 24` everywhere — the correlations against `p_v` and `k_v` are undefined for want
of variance. The `shipped` arm carries #548's allocation, and there the gradient is
non-monotone exactly as `06-graph-topology.md` records:

| | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|
| `p_v` (shipped, median) | 7 | 4 | 10 | 2 | 2 | 2 | 13 |
| `p_v` (reserve, median) | 8 | 8 | 8 | 8 | 8 | 8 | 8 |

### Reach

**Strongly graded, and downward.** `emission_gain` — the summed leading singular value of
every restriction map a cell sends on — correlates −0.88 with level. Rim cells push
hardest; the interior pushes least. The causal question, how far a perturbation actually
travels, is [B21](https://github.com/NGL321/patchworks/issues/570)'s and is not attempted
here.

---

## 2. Is there an abstraction gradient at all?

**Not by the best proxy available — and the proxy is flat, not inverted. The stronger
claim does not survive the null.**

The reading: a linear decode of world state from each cell's own chart over the window,
with a circularly-shifted target as an empirical null beside it. At 1,000 ticks, while
the world was still moving:

| decode target | L1 | L2 | L3 | L4 | L5 | L6 | L7 | Spearman w/ level |
|---|---|---|---|---|---|---|---|---|
| proprioception (`R²`) | 0.715 | 0.720 | 0.602 | 0.669 | 0.591 | 0.712 | 0.603 | **−0.22** |
| — its null | 0.089 | 0.062 | 0.057 | 0.033 | 0.039 | 0.036 | 0.056 | |
| touch (`R²`) | 0.297 | 0.288 | 0.240 | 0.257 | 0.209 | 0.283 | 0.276 | **−0.22** |
| — its null | 0.012 | 0.010 | 0.007 | 0.004 | 0.008 | 0.004 | 0.007 | |
| puck pose (`R²`) | 0.588 | 0.595 | 0.541 | 0.461 | 0.460 | 0.482 | 0.479 | −0.28 |
| — its null | 0.528 | 0.485 | 0.490 | 0.453 | 0.406 | 0.402 | 0.457 | |

Proprioception and touch decode far above their nulls everywhere, and the raw `R²` falls
with depth. **But the null falls with depth too**, and the honest statistic is the excess
over it — which is small and whose sign is **not stable across seeds**:

| Spearman(level, decode excess over null), 1,000 ticks | reserve/s42 | reserve/s43 | shipped/s42 | shipped/s43 |
|---|---|---|---|---|
| proprioception | −0.114 | +0.072 | −0.066 | +0.359\* |
| touch | −0.166 | — | −0.181 | −0.221\* |

So the reading is **no gradient**, in either direction, rather than an inverted one. An
earlier draft of this readout quoted the raw −0.22 from a single run as an inversion;
that is withdrawn — it was the null moving, not the content.

**Puck pose is the one target that has to be inferred** — `puck_pose` is the demo's
privileged accessor and the pucks reach the graph only through the rendered image — and
its fit sits at its own null at every level (0.588 against 0.528 at L1; 0.479 against
0.457 at L7). **No cell in the graph demonstrably represents it**, and the apex no more
than the rim.

A linear decode is a lower bound on content, and this is stated as such: it can miss a
nonlinear code. What it can say is that there is **no level at which readable content
rises**, on the one axis where the rig has a privileged variable to ask about.

---

## 3. The standing counter-evidence, in one place

| | reading | direction |
|---|---|---|
| [#548](https://github.com/NGL321/patchworks/issues/548) | Private dimension inverts after per-edge reallocation — L1 vision 7–9 against L4–L6's 1–3, apex 11–15. Recorded in `06-graph-topology.md`: *"the gradient is no longer monotone in depth"* | against |
| [#271](https://github.com/NGL321/patchworks/issues/271) | Private width is a relay aperture, not a stability gradient, and buys no retention: `corr(p_v, Δρ) = −0.019`, range −0.107..+0.047, retention 0.91 at the apex against 0.99 at the rim | against |
| [#276](https://github.com/NGL321/patchworks/issues/276) | **No run had ever carried a placed timescale gradient**, so every earlier flat reading was evidence about neither placement nor learning. `05-timescales.md`: #143's claim *"stands unchecked"* | neither |
| **this ticket** | On four trained arm-seeds: no timescale gradient (autocorrelation negative in all four), apex faster than the median cell everywhere and pulling further away with training, decodable content flat, private gradient absent on the reserve arm | against |
| [#556](https://github.com/NGL321/patchworks/issues/556) / this ticket | The reserve mask flattens `p_v` to a constant, so the map's own live arm has no private-dimension gradient at all | against |

### The pre-registered falsifier fires

`05-timescales.md`, *The gradient is learning's job*:

> **The falsification is pre-registered, and it is this section's own:** nothing
> guarantees the gradient appears. Learning may simply not produce it. […] If learning
> cannot produce the gradient, that redirects #127 rather than deadlocking it.

The same section records that this had never been checked — every flat reading to date
was taken where nothing was placed (#276), so flatness was uninformative. **This is the
first reading of the depth↔timescale correspondence on a trained arm**, and it does not
merely fail to find the gradient: the trajectory statistic is negative in all four runs,
and training takes the apex from ~0.4 to 0.12–0.27 of the graph's median retention in
every one of them.

Read against `05`'s own terms, the falsifier has fired. That is a finding this ticket
reports; what it costs is B27's to rule.

### And two structural facts, from the code rather than the run

* **No per-cell rate exists in the running architecture.** `patchworks.timescale.ClockDivisor`
  is an instrument held by nothing outside its own module and its tests;
  `cell.index.level` is read at runtime by nothing on the live path — only by
  `bias_selection`'s retired go/no-go and by `graph.py` at construction. The dome is
  entirely a construction-time object.
* **Nothing sparsifies.** There is no prune, grow, or rewire anywhere in `src/`. The
  graph is fixed at construction and the restriction mask *"closes and never re-opens"*.
  The wager's stated referent — a dense undifferentiated network that sparsifies toward
  semi-hierarchy — has no mechanism in this codebase. The dome is not a stand-in for that
  process; it is a hand-drawn picture of where the process was supposed to end up.

---

## 4. What a dome-free arm would cost to stand up

Rim plus unconstrained connectivity, sparsifying. **Not built, not priced beyond this.**

**What is free.** The running surface never reads the dome's shape. `tick.py`, `body.py`,
`restriction.py`, `learning.py`, `agent.py` and `diagnostics.py` contain no reference to a
level, a column or an apex: they take a cell list, an edge list, per-edge widths and a
private mask. Any graph satisfying `Σ_e m_e ≤ privacy_budget` at each predicting cell
runs today, unchanged. `dim_h0`/`dim_h1` are computed over the whole predicting
subcomplex and are shape-agnostic.

**What has to be written.**

1. **A connectivity rule to replace the taper.** `build_graph`'s `_covers` proportional
   covering, the per-level lateral fill and the degree targets all go. Whatever replaces
   them must still hand back cells, edges and degrees.
2. **A lane allocator that does not key on levels.** `allocate_lane_widths` detects a
   lateral edge as `level[u] == level[v]` and gives it `lateral_m`. With no levels,
   "lateral" is undefined, and #548's ruling — laterals at `m = 1` because *no lateral
   edge lies on any rim-to-apex chain* — is explicitly contingent on this dome's lateral
   count. It must be re-derived, not inherited.
3. **A sparsification mechanism, which does not exist.** This is the real cost. Growth or
   pruning of edges means the restriction mask reopening, and `restriction.py` re-applies
   the mask after every transport step *specifically so a learning rule cannot re-open
   it*. Whatever admits sparsification is a change to that invariant, and it is the same
   object [B17](https://github.com/NGL321/patchworks/issues/565)'s *"what a lane should
   be"* patch is circling.
4. **A drive attachment point.** ADR-0009 attaches the drive *deep* — the apex is the
   only cell it can hang from today, and it is why a plain boundary BFS reports the apex
   as one hop from the world. Without an apex the ADR needs a new rule for what "deep"
   means.

**What the rig loses.** `t2.rim_chains` needs an apex set to build a chain to; every
composed rim-to-apex reading on this map — B1, B6, B11, B13, B14, B16 — is defined
against it, as is `construction_grading`. Ten instruments under `benchmarks/` reference a
level or an apex. This is the same conclusion B17 reached from the other side: the map's
bar presumes the shape.

**ADRs that would need amendment,** by how heavily they lean on a depth axis:
ADR-0026 (rim–core influence as a conduction ratio), ADR-0027 (the demo's depth criterion
as a conduction time), ADR-0021 (rim-to-core detectability as a bottleneck ratio),
ADR-0009 (a drive is a motor edge attached deep), ADR-0024 (depth decimates in time and
not in space). ADR-0015's one-global-band and ADR-0028's learned spectrum are unaffected —
they were already written against placement.

**Cheapest honest first step, if B27 wants one:** the sandbox stalls from 2,000 ticks in
all four runs, so a dome-free arm would be compared against a largely motionless baseline
on every dynamical axis. Whatever the shape question, **the body's stall is upstream of
it**, and it is #517's by this map's own scoping.

---

## What was not measured

* **Causal reach.** No perturbation was injected and nothing was traced.
  [B21](https://github.com/NGL321/patchworks/issues/570) owns it; `emission_gain` is an
  operator-side proxy and nothing more.
* **Why the body stalls.** Whether the commanded stalk slice goes constant or the arm
  rests against a limit is not established here. The arm's command is #517's, by this
  map's own scoping.
* **Nonlinear content.** The decode is linear, so it lower-bounds what a cell represents.
* **Anything past 20,000 ticks.** The composed-rank record's 100k arms are not replicated
  here.
* **The shallow dome.** Every reading is the full dome; the map has never stood a shallow
  surface up.
* **Two seeds, one condition.** Baseline only, seeds 42 and 43, no pinned-drive arm.
