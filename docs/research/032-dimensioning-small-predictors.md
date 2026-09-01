# Citation pass: dimensioning small predictors and bottleneck width (patchworks#32)

Validates the dimensions closed in [#8](https://github.com/NGL321/patchworks/issues/8) and recorded
in [`06-graph-topology.md`](../spec/06-graph-topology.md): `n = 32` (node stalk of a predicting cell),
`k = 12` (chart), `m = 4` typical interior edge stalk, `m = 8` on boundary edges. Citations validate
after the fact per the map's Notes; this document does not revise the closed design — it flags where
a source threatens a claim already made. Vocabulary follows `CONTEXT.md`: Patchworks' side is
described in its own terms (chart, node stalk, edge stalk, predicting cell, boundary cell, cell
body, restriction map, piece), the prior art's in its own field's. Where a source could not be
reached, that is stated rather than papered over. Reservoir computing was established as the closest
structural precedent by [#13](https://github.com/NGL321/patchworks/issues/13) and that work is not
redone here — only the *sizing* question it left open is asked.

## Amendment, 2026-08-30: this pass defends `k_piece`, and there is no second `k`

Added on the resolution of [#145](https://github.com/NGL321/patchworks/issues/145), which was opened
to ask whether this pass's defence of `k = 12` also covers the Koopman conversion's use of the chart.
**It does not, and it does not need to.**

The pass defends `k = 12` as the width at which a chart **injectively names** the states of a piece,
via the fractal generalisation of Takens. That is a claim about *representation*. The conversion
raised a second and different question — how many observables make the piece's *dynamics linear* —
which #145 named `k_lift`, and which the [#148](https://github.com/NGL321/patchworks/issues/148)
citation pass priced at one to two orders of magnitude **above** the state, not below it. Read
against that scale, `k = 12` under `n = 32` would be indefensible.

**#145 resolved that the design has no `k_lift`, because it has no lift.** The chart *persists*, and
`encode` fuses it with new evidence every tick, so `K` is a linear recurrence driven by a nonlinear
input map rather than an EDMD dictionary over an instantaneous state. See
[ADR-0023](../adr/0023-the-chart-is-not-a-koopman-lift.md).

Two consequences for this document specifically:

- **Nothing below is withdrawn, and the delay-embedding reading is *strengthened*.** A persisting
  chart is a delay embedding, which is the object this pass's Sauer–Yorke–Casdagli argument was
  already about. #148 §4 identified generating the dictionary from delays as *"the cheapest fix
  available to this architecture specifically"* and noted that this pass's argument and that one are
  the same argument. They are — and the architecture was already making it structurally.
- **The size×delay trade-off below is load-bearing, not colour.** Duan et al.'s result that reservoir
  size trades against delay depth is what licenses a small per-cell `k` in a graph with unit delay on
  every edge. Under the reframe it stops being corroboration and becomes part of the defence.

**What this pass still does not cover** is the chart's *double duty*: `k = 12` must now name the
piece **and** carry the cell's memory, and a linear recurrence's recoverable short-term memory is
bounded by its state dimension — the same bound §"`n = 32`" below applies when it caps one cell's
linearly recoverable memory at 32 delay taps. The headroom argument here covers the naming job alone.
That gap is its own ticket, blocked on transmission.

## Headline verdict, stated plainly

**Nothing in the literature speaks to `n = 32` / `k = 12` / `m = 4` as a set, and the two bodies of
work that come closest speak past each other by two orders of magnitude in opposite directions.** The
capacity literature (useful width, autoencoder bottlenecks, intrinsic dimension of representations)
was measured at widths of 10²–10⁴ and has no calibrated statement about a 32-unit stalk; the
delay-embedding literature, which *is* dimensionally commensurate, says the governing quantity is
twice the box-counting dimension of the thing being charted, which for a cell's piece of this sandbox
is small enough that `k = 12` has real headroom.

Three specific results, in order of weight:

- **`k = 12` survives, and for a better reason than the arithmetic that set it.** The strict
  local-chart reading of ADR-0004 has a theorem behind it: an embedding is generic once the
  coordinate count exceeds **twice the box-counting dimension of the attractor**, not the ambient
  state dimension (Duan et al. 2023, quoting the fractal generalisation of Takens; Sauer, Yorke &
  Casdagli 1991). `k = 12` charts a piece of box-dimension up to ~6 in a world whose whole state is
  about twenty numbers. That is not tight.
- **`n = 32` is not challenged, but it is also not defended by reservoir computing.** Read naively,
  RC practice says the opposite of `n = 32` — "use as big a reservoir as you can afford", sizes of
  order 10⁴ commonplace (Lukoševičius 2012). The naive reading does not transfer, because a
  reservoir's size trades directly against **time delays** (Duan et al. 2023: 40 neurons × 5 lags ≡
  200 neurons; "a single neuron reservoir with time delays is sometimes sufficient"), and Patchworks'
  graph is delay-rich by construction — unit delay on every edge, diameter ~9 hops. The whole-graph
  state (≈150 predicting cells × 32 = 4,800 node-stalk dimensions; 1,800 chart dimensions) is
  squarely inside the range RC practice recommends. The per-cell number is not the comparable one.
- **`m = 4` is the number with the least headroom, and it is the one no source vindicates.** Under
  the same embedding criterion an interior edge stalk carrying 4 numbers can faithfully carry a
  shared piece of box-dimension below 2. Nothing found says that is wrong; nothing found says it is
  right either. `m` is also the *first* thing the #14 ladder permits flexing, so this is the cheap
  exposure rather than the expensive one — which is the good outcome.

**Nothing found argues the numbers are wrong by an order of magnitude in either direction.** The one
place where an order-of-magnitude gap appears — RC reservoir sizes of 10⁴ against `n = 32` — closes
once the delay/size trade-off and the whole-graph accounting are applied.

## 1. Useful width of very small MLPs, and where capacity collapses

### Sources

- Lu, Z., Pu, H., Wang, F., Hu, Z., Wang, L. (2017). "The Expressive Power of Neural Networks: A View
  from the Width." NeurIPS 2017. [arXiv:1709.02540](https://arxiv.org/abs/1709.02540).
- Park, S., Yun, C., Lee, J., Shin, J. (2021). "Minimum Width for Universal Approximation." ICLR 2021.
  [arXiv:2006.08859](https://arxiv.org/abs/2006.08859).

### Findings

There *is* a sharp width floor, and it is a theorem rather than a heuristic. Lu et al.: "width-(n+4)
ReLU networks, where n is the input dimension, are universal approximators. Moreover, except for a
measure zero set, all functions cannot be approximated by width-n ReLU networks, **which exhibits a
phase transition**." Park et al. sharpen this to an exact constant: "the minimum width required for
the universal approximation of the Lᵖ functions is exactly **max{d_x+1, d_y}**", and note it does
*not* hold for uniform approximation with ReLU alone.

**This does not bear on `n` or `k`.** Both results constrain the **hidden width of a map**, and `n`
and `k` are the *interface* dimensions of the cell — the input and output of `encode`, `step`,
`decode` — not their internal width. Where it does bite is a quantity `01-cell-and-sheaf.md` leaves
open: the shared frozen cell body's hidden layers. For `encode : ℝ³² × ℝ¹² → ℝ¹²`, Park et al. gives
`w_min = max(45, 12) = 45`; for `step : ℝ¹² → ℝ¹²`, `w_min = 13`; for `decode : ℝ¹² → ℝ³²`,
`w_min = max(13, 32) = 32`. **A cell body whose hidden layers are narrower than ~45 is provably not
a universal approximator of its own encoder**, and Lu et al.'s phase transition says the failure is
abrupt, not graceful. The spec fixes no hidden width, so this constrains an undetermined parameter
rather than contradicting a committed one — but it is a floor that should be written down before the
body is built.

> **Corrected — this pass first wrote `w_min = 33` for `decode`.** An arithmetic slip: the `+1` was
> applied to whichever term won, where Park et al. put it on the `d_x` term alone. `encode` (45) and
> `step` (13) are cases where `d_x + 1` wins and are unaffected; `decode` is the one map where the
> `d_y` term wins, and the one that came out one too high — at exactly `d_y + 1`. The 33 reached
> [`01-cell-and-sheaf.md`](../spec/01-cell-and-sheaf.md) and
> [`05-timescales.md`](../spec/05-timescales.md) by way of
> [#49](https://github.com/NGL321/patchworks/issues/49) and was ruled back to 32 in
> [#84](https://github.com/NGL321/patchworks/issues/84). Nothing measured moved with it: `decode` is
> off the chart's round trip, so no fold margin in the record was computed at a `decode` width. The
> line above and the recommendation below now read 32.

Lu et al. also record that "there exist classes of wide networks which cannot be realized by any
narrow network whose depth is no more than a polynomial bound" — depth does not buy back width.
**No source found** addresses the useful width of very small MLPs empirically at the 12–32 scale;
these floors are all that exists, and they concern representability, not trainability and not the
carrying of a nonlinear dynamical step.

## 2. Autoencoder bottlenecks: is the ratio `n/k` principled?

### Sources

- Ansuini, A., Laio, A., Macke, J.H., Zoccolan, D. (2019). "Intrinsic dimension of data
  representations in deep neural networks." NeurIPS 2019.
  [arXiv:1905.12784](https://arxiv.org/abs/1905.12784).
- Pope, P., Zhu, C., Abdelkader, A., Goldblum, M., Goldstein, T. (2021). "The Intrinsic Dimension of
  Images and Its Impact on Learning." ICLR 2021.
  [arXiv:2104.08894](https://arxiv.org/abs/2104.08894).
- Loche et al. (2025). "Intrinsic Dimension Estimating Autoencoder (IDEA) using CancelOut layer and a
  projected loss." [arXiv:2509.10011](https://arxiv.org/abs/2509.10011).

### Findings

**The ratio `n/k` has no principled basis anywhere in this literature — the ratio is not the object
the field reasons about.** Every method found selects a bottleneck from an *absolute* estimate of the
data's intrinsic dimension, never from a fraction of the input dimension. IDEA's criterion is
"continuously assessing the reconstruction quality under the removal of an additional latent
dimension" — shrink until reconstruction degrades. The common practice around it is the elbow on
reconstruction-error-vs-bottleneck-size, inherited from PCA scree plots. **No source found** states,
recommends, or derives a target `n/k`.

That is a *supporting* finding for the spec as written. `01-cell-and-sheaf.md` already says "the
*degree* of compression (`n/k`) is a hyperparameter; the spec commits to `k < n` and nothing more,"
and that "a useful `k` turns out to be much smaller than `n` is a finding the proof-of-concept
reports, not a number fixed here." The literature agrees that the ratio is not the thing to defend.

The absolute numbers place Patchworks outside the regime these papers measured. Ansuini et al.: "the
ID is orders of magnitude smaller than the number of units in each layer," rising then falling across
depth, and "the ID of the last hidden layer predicts classification accuracy on the test set."
Patchworks' `n/k ≈ 2.7` is not orders of magnitude. **Nothing in Ansuini et al. says 2.7 is too
small** — the result describes what trained large networks do, it does not require it — but their
ratios cannot be transplanted, and `k < n` reads better as a shape invariant than a compression
claim.

Pope et al. give the only usable absolute calibration. Their MLE estimates: MNIST 7–13, CIFAR-10
13–26, CIFAR-100 11–23, SVHN 9–19, ImageNet 26–43, CelebA 9–26 — whole datasets of natural images, in
ambient dimensions of 10³–10⁵. **A 4×4 RGB patch of a three-puck arena is a far smaller thing than
MNIST**, so a boundary cell's 48-dimensional stalk plausibly carries an intrinsic dimension well
under 8, and `m = 8` on boundary edges has headroom against the closest available yardstick. Their
scaling result — "learning a manifold requires a number of samples that grows **exponentially** with
the manifold's intrinsic dimension" — cuts *in favour* of small `k`: every dimension added to the
chart is paid for exponentially in experience, which is the expensive currency for a continually
learning system with no dataset.

## 3. Reservoir computing: size per unit of task complexity

### Sources

- Lukoševičius, M. (2012). "A Practical Guide to Applying Echo State Networks." In *Neural Networks:
  Tricks of the Trade*, 2nd edn, LNCS 7700, 659–686. (PDF stream unparseable to the fetcher; the
  §"Size of the Reservoir" text was recovered by search over the same primary PDF.)
- Jaeger, H. (2002). "Short term memory in echo state networks." GMD Report 152 — **not read
  directly**; its memory-capacity bound is quoted from Wang, S. et al. (2023), *Int. J. Imaging
  Systems and Technology* 34(1), [10.1002/ima.22940](https://doi.org/10.1002/ima.22940).
- Duan, X.-Y., Ying, X., Leng, S.-Y., Kurths, J., Lin, W., Ma, H.-F. (2023). "Embedding Theory of
  Reservoir Computing and Reducing Reservoir Network Using Time Delays."
  [arXiv:2303.09042](https://arxiv.org/abs/2303.09042).

### Findings

**There is no rule of the form "N nodes per unit of task complexity."** Lukoševičius' guide — the
field's standard practical reference, and the one place such a rule would live — gives only a
monotone recommendation: "The bigger the space of reservoir signals x(n), the easier it is to find a
linear combination of the signals to approximate y_target(n). The reservoir can be too big only when
the task is trivial and there is not enough data available T < 1 + N_u + N_x. For challenging tasks
use as big a reservoir as you can afford," with sizes of order 10⁴ described as unremarkable. A
Scholar Gateway sweep for a task-complexity-indexed sizing rule returned nothing better.

The one hard bound is on memory rather than capacity: "the reservoir's memory capacity does not
exceed its size, that is, **MC ≤ N**" (Wang et al. 2023, restating Jaeger). Applied to a predicting
cell, `n = 32` caps the linearly recoverable short-term memory of one cell's node stalk at 32 delay
steps. This is *consistent with*, and independently arrived at by, the spec's own private-dimension
arithmetic: at degree 6 with `m = 4`, `max(0, n − Σ_e m_e) = 32 − 24 = 8` dimensions are private and
exactly invariant under reconciliation, which `05-timescales.md` identifies as the capacity to hold
slow state. Eight dimensions of guaranteed-private slow state, inside a 32-dimensional memory
ceiling, is internally coherent. **No source found** says eight is too few for anything in
particular.

**Duan et al. is the finding that defuses the 10⁴-vs-32 gap, and it is a proof rather than a
heuristic.** They "rigorously prove that RC is essentially a high dimensional embedding of the
original input nonlinear dynamical system" (Theorem 1: an embedding is generic once
`m ≥ 2·dim(ℳ) + 1`, `dim(ℳ)` the box-counting dimension), and derive from it "a trade-off relation
between the time delays and the number of neurons in RC," reporting that "40 neurons with uniformly 5
lags" matched 200 undelayed neurons and that "only using a **single neuron reservoir with time
delays** is sometimes sufficient." Reservoir *size* is therefore not the invariant; the product of
size and delay depth is. Patchworks buys delay depth structurally — unit delay on every edge, ~9-hop
diameter, a chart that persists across ticks — so the per-cell `n` is not the quantity the RC
literature's 10⁴ is commensurate with. This is a genuine transfer from the precedent #13 identified,
and it lands in the spec's favour.

## 4. The dimension of a learned local chart (the strict ADR-0004 sense)

### Sources

- Sauer, T., Yorke, J.A., Casdagli, M. (1991). "Embedology." *J. Statistical Physics* 65, 579–616.
  [10.1007/BF01053745](https://doi.org/10.1007/BF01053745) — **unreachable in readable form**; three
  fetch attempts (SFI preprint PDF, Springer, an arXiv restatement) all failed to yield parseable
  text. Its criterion is quoted verbatim below from Duan et al., a primary source that restates it.
- Duan et al. 2023, as above.

### Findings

This is the area the ticket flags as least obvious, and it is the one where the literature is most
directly applicable to a number this small — because the governing quantity is *not* a capacity but a
dimension count, and dimension counts of a few units are exactly what these theorems are about.

The criterion, verbatim from Duan et al.'s statement of the fractal generalisation of Takens: an
observation map "is generically an embedding as long as **Σᵢ dᵢ > 2·dim(ℳ)**," where "dim(ℳ) denotes
box-counting dimension of the manifold." Sauer, Yorke & Casdagli's original result (per its abstract,
which could not be read in the paper itself) is that when the coordinate count exceeds twice the
box-counting dimension of the attractor `A`, almost every delay-coordinate map is one-to-one on `A`
and an embedding on smooth manifolds inside it — and, when it does not, the paper characterises the
**self-intersection set** that results. Self-intersection is precisely the failure mode Patchworks
would see: two genuinely distinct situations receiving the same chart coordinates, indistinguishable
to `step`, and appearing downstream as irreducible disagreement — the same signature ADR-0004 and
ADR-0007 are already trying to read.

Applied to the committed numbers, with the sandbox's world state at about twenty numbers
(`06-graph-topology.md`):

| stalk | dimension | supports a piece of box-dimension up to |
|---|---|---|
| chart `k` | 12 | < 6 |
| node stalk `n` | 32 | < 16 |
| interior edge `m` | 4 | < 2 |
| boundary edge `m` | 8 | < 4 |

A cell's **piece** — one 4×4 patch, or one deep abstraction over a few pucks — has a box-counting
dimension far below 6 in a world of about twenty numbers total. `k = 12` is therefore comfortable in
the strict ADR-0004 sense, and the ADR's framing (`k` is the dimension of the piece, `n` is the room
needed to talk about it with neighbours) is the framing this literature actually supports. `m = 4` is
where the margin thins: an overlap of box-dimension 2 or more is not faithfully embeddable in a
4-dimensional edge stalk, and two pucks' worth of shared structure would already be at that limit.
The mitigation is real but indirect — Duan's delay/size trade-off means many delayed 4-dimensional
views across ticks recover what one static 4-dimensional view cannot, and Patchworks' edges are
delayed by construction. **No source found** settles whether that mitigation is sufficient.

**Gap, stated plainly.** The manifold-learning primaries for local charts — Roweis & Saul (Science
2000) and Saul & Roweis (JMLR 2003) — could not be read: `science.org` returned 403 and both mirror
PDFs were unparseable. Two attempts each, then abandoned per protocol. Their contribution would have
been the *estimation* question (how one reads a piece's dimension off data), not the *criterion*,
which Duan et al. supplies verbatim. The criterion is what the spec needed.

## Are the numbers wrong by an order of magnitude?

**No — in neither direction, on the evidence found.**

- **Too small?** The only order-of-magnitude case against is RC's 10⁴ reservoirs versus `n = 32`. It
  fails on two independent grounds: the delay/size trade-off (Duan et al.), and the whole-graph
  accounting — ~150 predicting cells × 32 is ~4,800 dimensions of state, and the comparable object to
  an echo-state reservoir is the graph, not one cell.
- **Too large?** Pope et al.'s exponential sample-complexity scaling is the only argument for
  shrinking `k`, and it argues from a principle, not from a number for a world this small. Nothing
  found says 12 is extravagant for a piece of a three-puck arena.
- **The genuinely open number is `m = 4`**, and it is open because no literature addresses it, not
  because literature contradicts it.

## Revision tickets recommended (not created)

1. **Record a minimum hidden width for the shared cell body.** Park et al.'s `max{d_x+1, d_y}` gives
   `w_min = 45` for `encode`, `32` for `decode`, `13` for `step`, and Lu et al. show the failure at
   sub-minimum width is a phase transition rather than a degradation. `01-cell-and-sheaf.md` leaves
   the body's hidden width unspecified; this is a free, provable floor that belongs in the spec
   before the body is built. Does not touch `n` or `k`.
2. **Add the self-intersection reading to ADR-0004's falsification test.** ADR-0004 already names
   persistent structured disagreement as the signature of curvature the linear restriction map cannot
   follow, and ADR-0007 adds the lag floor as a confounder. Sauer/Yorke/Casdagli name a *third* cause
   with the same surface appearance: a stalk too narrow to embed its piece, producing
   self-intersection. It has a distinguishing test the other two lack — widen the stalk and the
   residual falls — and the criterion (`dim > 2·boxdim`) makes it predictable rather than merely
   diagnosable. Documentation-only; commits nothing.
3. **Note in `06-graph-topology.md` that `m = 4` is the dimension with the least theoretical
   headroom**, with the `2·boxdim` criterion as the reason and the delay/size trade-off as the
   mitigation. `m` is the first rung on #14's flex ladder, so this is cheap to record and cheap to
   act on later.

None of these revises a committed number. `n = 32`, `k = 12`, `m = 4`, `m = 8` stand unchallenged.
