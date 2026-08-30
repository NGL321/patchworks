# Effective resistance on the dome, and whether the gauge band dominates the hop

[#150](https://github.com/NGL321/patchworks/issues/150). Two measurements, no decision in either,
both graph-side and neither needing a training run. Reproduced by

    .venv/bin/python benchmarks/graph_transmission.py       # 0.4 s wall

on the real dome: 414 cells (150 predicting, 264 boundary), 682 edges, `chi = 1036`. Both estimators
are checked against closed forms in `tests/test_graph_transmission.py` — Foster's theorem for
effective resistance, `n/(n-1)` on a complete graph for balanced Forman curvature — because the whole
ticket is arithmetic and a plausible wrong number would not be caught downstream.

---

## The verdict, in three lines

1. **The gauge band is not the dominant term, and the resemblance to `2^9 = 512` is a coincidence.**
   The band's *entire* dynamic range is `rho^2 = 4x` against a per-hop loss of ~866x, it currently
   contributes exactly **1.00** because every map is initialised at Frobenius 1, and raising `rho`
   is **asymptotically self-cancelling**: `rho` sits in the numerator through the two map norms and
   in the denominator through `gain_v = gamma / max(sum_e m_e, rho^2 deg(v))`. At `rho = 16` the hop
   is **1.008x** what it is at `rho = 2`.

2. **The dominant terms are both construction-time dimensioning, and neither is learned.** Of the
   per-hop loss, **52% is the reconciliation gain** and **35% is dimensional dilution**
   (`1 / sqrt(d_p m_e)`), against 12% for the body and **0% for the gauge**.

3. **The topological term is small, so rewiring is not the lever here.** Rim to apex is **7 hops of
   graph distance but only 1.82 unit-resistance edges**. The dome's parallel routes have already
   bought back nearly all of its depth. Every remedy in `docs/research/148`'s §10.3 table attacks a
   term worth under 2x, while the loss is ~866x per hop in the model term.

The third is the one that changes what [#142](https://github.com/NGL321/patchworks/issues/142) should
do, and it was not predictable from the record.

---

## 1. Effective resistance and commute time

Di Giovanni et al. (ICML 2023, arXiv:2302.02941) Theorem 5.5 bounds the Jacobian obstruction above
and below by commute time `tau(v,u) / 2|E|` — that is, by effective resistance — **independent of
depth**. So the topological half of the sensitivity factorisation is computable without running
anything, and it had never been computed on this graph.

Unweighted (`volume = 1364`), to the apex (the eight L7 predicting cells):

| from | cells | hops | R mean | R max | tau mean | tau max | R in unit edges |
|---|---|---|---|---|---|---|---|
| L1 | 70 | 6 | 0.8281 | 1.0524 | 1129.5 | 1435.5 | 0.83 |
| L2 | 20 | 5 | 0.6702 | 0.8177 | 914.1 | 1115.4 | 0.67 |
| L3 | 16 | 4 | 0.5799 | 0.6226 | 791.0 | 849.2 | 0.58 |
| L4 | 14 | 3 | 0.5533 | 0.6072 | 754.7 | 828.2 | 0.55 |
| L5 | 12 | 2 | 0.5159 | 0.5874 | 703.7 | 801.2 | 0.52 |
| L6 | 10 | 1 | 0.4690 | 0.5566 | 639.8 | 759.2 | 0.47 |
| actuator | 1 | 7 | 1.1967 | 1.2145 | 1632.3 | 1656.6 | 1.20 |
| drive | 1 | 1 | 0.3355 | 0.3509 | 457.7 | 478.6 | 0.34 |
| **patch** | 256 | 7 | **1.8163** | 2.0524 | **2477.4** | 2799.5 | **1.82** |
| proprioceptive | 3 | 7 | 1.9484 | 1.9683 | 2657.6 | 2684.7 | 1.95 |
| touch | 3 | 7 | 1.9607 | 1.9810 | 2674.4 | 2702.1 | 1.96 |

Stalk-weighted (`w_e = m_e`, `volume = 7528`) gives the same ordering with everything scaled: patch
to apex `R = 0.3432`, `tau = 2583.8`.

**The last column is the finding.** A sensory patch is seven hops from the apex and **1.82
unit-resistance edges** from it. One of those 1.82 is the patch's own leaf edge, which no rewiring
short of moving the sensor can remove; the remaining ~0.82 is the entire six-level climb from L1 to
the apex, because each level is a lattice with many parallel routes into the next.

The consequence for #142 is blunt. The topological term the over-squashing literature exists to
attack is **already near its floor** on this graph. Adding a virtual node, a fully-adjacent last
layer, or an expander rewiring can reduce `R` from 1.82 toward at best ~1.0 — the leaf edge is
irreducible — and would buy under a factor of two, against a deficit the map prices at ~1e14. Those
remedies are aimed at a term the dome does not have a problem with. **`docs/research/148` §10.3's
table should be read with this column beside it**: it correctly lists what each remedy buys in
general, and on *this* graph the answer for all of them is "almost nothing".

### The worst cuts: every patch cell is a bridge

`w_e * R(u,v)` on an edge is that edge's spanning-tree probability — 1 exactly for a bridge, low
where many parallel routes exist. They sum to `n - 1` by Foster's theorem, which is the global check
on the computation.

| edges | count | mean share | median | min |
|---|---|---|---|---|
| sensory (m=8) | 262 | **1.0000** | 1.0000 | 1.0000 |
| motor (m=8) | 3 | 0.4184 | 0.4183 | 0.4183 |
| interior (m=4) | 409 | 0.3596 | 0.3459 | 0.2548 |
| drive (m=1) | 8 | 0.3355 | 0.3347 | 0.3216 |

**262 of 682 edges are bridges, and they are exactly the sensory edges.** Every patch cell has degree
one, so its information has precisely one route out and that route is a cut of the graph. This is
what `06-graph-topology.md` intends — a patch cell *is* a leaf — but it is worth having it named:
the rim is not merely the highest-resistance region, it is the only region where a single edge
failing or saturating disconnects a cell outright.

The worst pairs in the graph are patch-to-patch across the render (`R = 3.27`, `tau = 4464`), which
is two leaf edges plus the climb. No pair involving the apex is anywhere near the top.

---

## 2. The gauge's share of one hop

The check is exact rather than statistical. One hop's two map factors are

    restrict_p = |F_p dx| / |dx|            edge_r = gain_r * |F_r^T d| / |d|

and for a uniformly drawn direction, `E|F u| ~= |F|_F / sqrt(d)`. So the hop factorises into a term
the gauge sets, a term construction sets, and the body — each readable separately. Over the 1091
hop-carrying endpoints (the same population `benchmarks/untrained_fixed_point.py` reads):

| | `|F|/sqrt(d)` | drawn | ratio (mean / min / max) |
|---|---|---|---|
| restrict | 0.19035 | 0.18105 | 0.9497 / 0.8789 / 1.0415 |
| edge | 0.01418 | 0.01408 | 0.9931 / 0.9537 / 1.0215 |

The identity holds, so the split below means something. The predicted hop is
`0.1811 * 0.0141 * 0.4529 = 0.001155`, against **#120's measured 0.001086 — a ratio of 1.063**,
reached here analytically, from the graph and the maps alone, with **no sandbox and no ticks**. That
agreement is the strongest evidence in this document that the decomposition is the right one.

| factor | value | share of the loss | what sets it |
|---|---|---|---|
| gauge `|F_p| |F_r|` | 1.00000 | **0.0%** | the gauge band, in [1/4, 4] |
| dilute `1/sqrt(d m)` | 0.09303 | **35.1%** | the mask width and `m_e`, at construction |
| gain `gamma/max()` | 0.02944 | **52.1%** | `gamma`, `m_e` and degree, at construction |
| body | 0.45290 | 11.7% | a frozen random MLP; bounded by nothing |

### Why raising `rho` cancels

`gain_v = gamma / max(sum_e m_e, rho^2 deg(v))`. At `rho = 2`, **80 of the 150 receiving cells are
already on the `rho^2 deg` arm** (interior cells, where `sum_e m_e = 4 deg` ties it exactly), and
every cell moves onto that arm as `rho` grows. On that arm the gain falls as `rho^-2` precisely as
the two map norms rise as `rho^2`.

| rho | restrict | edge | hop | vs rho=2 |
|---|---|---|---|---|
| 1 | 0.19035 | 0.01438 | 0.00124 | |
| 1.5 | 0.26202 | 0.02157 | 0.002559 | |
| **2** | 0.33369 | 0.02837 | **0.004287** | 1.000x |
| 3 | 0.47702 | 0.02175 | 0.004699 | **1.096x** |
| 4 | 0.62036 | 0.01631 | 0.004583 | 1.069x |
| 8 | 1.19370 | 0.00816 | 0.00441 | 1.029x |
| 16 | 2.34037 | 0.00408 | 0.004323 | 1.008x |

The `rho = 1` to `rho = 2` step buys **3.46x**, which independently reproduces the record's
`0.001086 -> 0.003747` (3.45x) for saturating the band — by a different route, since nothing here
runs the transport rule. Past `rho = 2` the curve is flat: the best any `rho` reaches is **1.096x**,
at `rho = 3`, and it decays back toward 1.0 from there.

**So the answer to the ticket's second question is no, and the reason is stronger than the number.**
It is not that the gauge happens to be worth little today. It is that `rho` is structurally
prevented from buying transmission by its own appearance in the gain, which is ADR-0010 and
`02-tick-semantics.md` interacting in a way neither document notices. Candidate 3 of #142 — the
gauge — is closed by this, and closed on arithmetic rather than on a measurement that could come out
differently on a trained surface: the two `rho^2` cancel identically, at any operating point.

---

## 3. What the stalk widths are worth

`m` enters the hop three times and helps in none of them, because ADR-0010 fixes a Frobenius norm
**independent of `m`**: the same norm over more rows is less per row (`1/sqrt(m)` in the dilution),
and `sum_e m_e` is the gain's other arm (`1/m` above `m = rho^2`). Above that threshold the two
compose to `m^-3/2`. Rebuilding the dome at other widths and pricing each with the identity above:

| interior_m | boundary_m | hop | vs built | private dim | chi |
|---|---|---|---|---|---|
| 2 | 8 | 0.00212 | **1.734x** | 10.75 | 1854 |
| 4 | 4 | 0.0015 | 1.227x | 4.13 | 2096 |
| **4** | **8** | **0.001223** | **1.000x** | 3.95 | 1036 |
| 4 | 16 | 0.00104 | 0.851x | 3.95 | -1084 |
| 6 | 8 | 0.0007214 | 0.590x | 0.40 | 218 |
| 8 | 8 | 0.0005114 | 0.418x | **0.00** | -600 |

**Widening an edge stalk costs transmission.** This is not an argument against `m = 8` at the rim,
because width and gain buy different things: width buys **rank** — how many of a patch cell's 48
directions can leave at all — and costs **gain per direction**. It is an argument that the trade was
never priced, and this is the price. Narrowing interior edges to `m = 2` is worth **1.734x per hop**
and *also* returns private dimension (10.75 against 3.95), which is `H^0`, which is slow state. Both
halves point the same way, and the direction is the opposite of the instinct that a transmission
problem is solved by widening the pipe.

The `private dim` column is the hard constraint on the other end: at `interior_m = 8` the graph has
**no private features at all**, so `01-cell-and-sheaf.md`'s `H^0`-as-private-features and
ADR-0005's timescale-by-persistence both lose their substrate. Widening was never free and is now
known to be doubly costly.

Over seven hops, `1.734^7 = 47x`. Against a ~1e14 deficit that settles nothing — this is a lever,
not a solution, and #142 should treat it as one.

---

## 4. Balanced Forman curvature

Topping et al. (ICLR 2022, arXiv:2111.14522) Definition 1, which they prove identifies the
negatively curved edges *"responsible for the over-squashing issue"*.

| edges | count | mean | median | min | max | negative |
|---|---|---|---|---|---|---|
| all | 682 | 1.2952 | 0.6667 | -1.4167 | 8.4722 | 6.7% |
| drive (m=1) | 8 | 0.7000 | 0.8500 | -0.4500 | 1.5500 | 25.0% |
| interior (m=4) | 409 | 2.1454 | 2.1071 | **-1.4167** | 8.4722 | 10.8% |
| motor (m=8) | 3 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0% |
| sensory (m=8) | 262 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0% |

**The sensory row is silence, not a clean bill of health.** Definition 1 is stated for
`d_i, d_j > 1` and every patch cell has degree 1, so curvature is identically zero there *by
convention*. The cut-share table above is what speaks about the rim instead, and it says every one
of those edges is a bridge.

**The suspects are the L2 to L3 edges**, and they are `m = 4`:

    L2/vision(2, 0)     -> L3/core(8,)     interior m=4  Ric -1.4167   R(u,v) 0.3785
    L2/vision(1, 0)     -> L3/core(4,)     interior m=4  Ric -1.4167   R(u,v) 0.3786
    L2/vision(1, 3)     -> L3/core(7,)     interior m=4  Ric -1.4167   R(u,v) 0.3808
    L2/somatomotor(0,)  -> L3/core(0,)     interior m=4  Ric -1.3333   R(u,v) 0.4870

This is the vision-to-core junction, and it is **independently the graph's narrowest waist**:
`Dome.cut_capacities` reports `L1 -> L2` at 280 and `L2 -> L3` at **80**, a 3.5x contraction in one
level, tied with `L6 -> L7` for the narrowest cut in the dome. Two diagnostics that share no
arithmetic — a local curvature and a global capacity count — name the same edges.

Curvature and per-edge effective resistance correlate at **r = -0.6002** over the 682 edges, which
is the expected sign and a reasonable magnitude: they are related diagnostics, not the same one.

**So the design's `m = 8` at the rim is relocated rather than justified.** The rim is where the
bridges are and the junction is where the squeeze is, and they want different treatments — the rim
wants rank, the junction wants either width or more routes. Whether either is worth doing is #142's
call, and §3 above says width is not free.

---

## What this leaves for #142

The ticket must produce *"a target per-hop transmission, derived rather than chosen"*. What is now on
the table for that derivation:

- **The budget is fully attributed.** `hop = gauge * dilution * gain * body`, the four measured, and
  their product reproduces #120's reading to 6% from the graph alone. A target on the hop can now be
  translated into a target on a named factor.
- **Two of the four are closed.** The gauge cannot move the hop (§2, structurally). The topological
  term is already near its floor (§1), so rewiring is not the lever the literature suggests it is.
- **One is a construction parameter with a measured price and a known cost** (§3): narrower interior
  edges, worth 1.734x per hop at `m = 2`, paid for in rank and returned in `H^0`.
- **One is the reconciliation gain, and it is 52% of the loss** — the largest single term, set by
  `gamma / max(sum_e m_e, rho^2 deg)` with `gamma` already at its ceiling of 1.0. Its `deg` and
  `sum_e m_e` are construction choices nobody has priced against transmission, and after this
  document `rho` is known not to be a route into it.
- **The body remains the only unbounded factor**, at 11.7%, and remains what the Koopman conversion
  turns into a design variable. That case is untouched by anything here.

## What was not done

**The sheaf Laplacian's effective resistance was not computed**, though `docs/research/148` §10
flags the cross-reference to Hansen & Ghrist (arXiv:1808.01513) as the unexploited link. The
graph-side computation is what the ticket asked for and is done; the sheaf-side one is a different
and larger object (~17,000 stalk dimensions) and needs a pair of stalk *directions*, not a pair of
cells, before "resistance between rim and apex" even has a referent.

It is worth stating why it is still open rather than cheap. The gauge bounds each restriction map's
**magnitude** to within `rho^2`, so the sheaf resistance cannot differ from the graph resistance by
more than a bounded factor **in scale**. It says nothing about **rank**: a map that is
rank-deficient in a direction makes that direction infinitely resistive, and the structural mask
creates exactly such deficiency by construction — that is what private features *are*. So the sheaf
resistance can be arbitrarily worse than the graph resistance in specific directions, and which
directions those are is a real question this document does not answer.
