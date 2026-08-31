# The fold margin: right space, right quantity, re-run against the new gain (#195)

`02-tick-semantics.md` carries a per-cell check that a standing reconciliation
offset must not walk a cell across an activation boundary:

```
gain_v x floor_v  <  margin_v
```

[#189](https://github.com/NGL321/patchworks/issues/189) found it **well-formed
but needlessly tight**, and [#190](https://github.com/NGL321/patchworks/issues/190)
then changed `gain_v` out from under it. This directory measures both
corrections at once, because both land on the same per-cell pass in
`bias_selection.py` and doing them separately means paying for the
100,000-tick surface twice.

## How to run it

Inside the supported container (ADR-0012):

```
docker run --rm -v "$PWD:/work" -w /work --entrypoint python patchworks:headless \
    prototypes/fold-margin-195/margin.py <subcommand>
```

| subcommand | what it does |
|---|---|
| `construction` | the two margins read off `bias_selection`'s construction sweep — where `02`'s published `0.3502` comes from. Seconds, no surface run. |
| `surface` | one agent taught to 100,000 ticks, read at four budgets: every predicting cell's floor and both margins, on the same surface at the same instant. ~35 minutes. |

`_divergence.py`, `_readonly.py` and `_compare.py` are the instrument checks
described under *What is not trustworthy* below.

## The two defects

**Defect 1 — the wrong space.** `bias_selection._fold_margin` reads Hanin &
Rolnick's distance to the nearest region boundary, `min_i |z_i| / ||g_i||`, with
the gradient over `encode`'s *whole* input. `encode` eats `cat(chart, node_stalk)`
in `R^k x R^n`, so that row norm runs over all `k + n = 44` directions. But
reconciliation displaces the **node stalk alone** — the message-passing phase
writes `stalks`, never `charts` — so the displacement the check guards against
lives in the `n = 32` subspace the stalk enters through. Restricting the gradient
to `encode_hidden_weight[:, k:]` is the margin in the space the offset moves in.
It is never smaller, so the standing check is **conservative and never unsafe**.

Measured on the built body: **mean 1.183, median 1.164** across the four budgets
(per-budget means 1.187 / 1.197 / 1.183 / 1.183), against #189's mean 1.176 and
the isotropic expectation `sqrt(44/32) = 1.1726`. #189's number reproduces.

**Defect 2 — the wrong quantity.** `02`'s prose writes the bound as
`gamma x floor < margin_v`. What multiplies the floor in the code and in the run
is `gain_v = gamma / denominator_v`. See *The ruling on defect 2* below: it is a
shorthand in `02`'s prose alone, not a defect in the mechanism.

## What the corrections are worth, at construction

`construction`, seed 42, 8192 draws — the same configuration `02` records:

| reading | cap on `gamma x floor` |
|---|---|
| **published** (full space, old denominator) | **0.3502** — reproduces `02` exactly |
| defect 1 only (stalk space, old denominator) | 0.4107 |
| defect 2 only (full space, new denominator) | 0.1167 |
| **both** | **0.1369** |

Both readings of the demo's horizons (`onset`, `duration`) give the same
numbers: the margin check runs on drawn candidates at this configuration, and
the drawn population does not depend on the target range.

**Net, the check tightens 2.56x.** Defect 1 loosens by 1.17x and #190's
denominator tightens by 3.00x at the binding cell, and the second is the larger.

## What it permits, per cell, on the surface it runs on

`surface`, real dome, `train`, seed 42, both rules on, 400-tick quiescent hold at
each budget (ADR-0007's protocol, as #158 and #178 run it). Margins are the
**median over the hold**, which is the estimator `bias_selection.measure` uses
and the one `02`'s published number comes from.

| ticks | floor med | margin med | cap (both) | binding cell | level | `gamma_cap` | cells < 1 |
|---|---|---|---|---|---|---|---|
| 30,000 | 0.0489 | 0.0248 | 0.00242 | 62 | 1 | 0.0913 | 39 |
| 50,000 | 0.0381 | 0.0217 | 0.00202 | 86 | 2 | 0.0443 | 37 |
| 75,000 | 0.0428 | 0.0244 | 0.00027 | 53 | 1 | 0.0111 | 28 |
| 100,000 | 0.0325 | 0.0192 | 0.00126 | 33 | 1 | 0.0375 | 24 |

**Read this way the check binds hard: `gamma <= 0.0375` at 100,000 ticks.**

**And the number is not a measurement.** Three things say so at once:

1. **The binding cell is a different cell at every budget** — 62, 86, 53, 33.
2. **`gamma_cap` wanders 8.2x with no trend** (0.0111 to 0.0913) across budgets
   over which the *population* is steadily improving.
3. **The population statistics are stable.** The median margin sits in a 1.3x
   band (0.0192–0.0248) across the same budgets, and the count of cells inside
   their own margin at `gamma = 1` falls monotonically, 39 → 37 → 28 → 24.

The systematic quantity is stable and improving; the extremal quantity re-rolls.
A global cap set by the tightest of 150 cells is an **order statistic**, not a
property of the surface — which is the same shape of finding #178 recorded when
the apex "wandered 3.8x with no trend", one level up.

The estimator makes this concrete. Same cells, same window, three readings:

| reading | med margin | min margin | min cap | `gamma_cap` |
|---|---|---|---|---|
| median over the hold (`02`'s) | 0.01924 | 0.00016 | 0.00126 | 0.0375 |
| min over the hold | 0.01759 | 0.00001 | 0.00009 | 0.0001 |
| one instant | 0.01873 | 0.00016 | 0.00126 | 0.0242 |

The medians agree to 10%; the minima differ by **16x** and the caps by **14x**.
Everything the global cap says is a statement about the estimator.

## `02`'s systematic claim does not survive, and #190 is why

`02` claims the check "binds hardest exactly where timescale matters most",
because `sum_e m_e` falls with depth and `gain_v` is therefore largest at the
apex — and records level medians of the margin falling 1.25 at level 1 to 0.52
at the apex.

Under #190 the denominator is a **flat 8** at every predicting cell, so the cap
is `8 x margin_v` and carries no depth information at all. What is left is the
margin, and on the built surface it has no depth trend:

```
margin, level medians      L1      L2      L3      L4      L5      L6      L7
  30,000               0.0253  0.0268  0.0268  0.0196  0.0264  0.0201  0.0186
  50,000               0.0199  0.0288  0.0254  0.0270  0.0635  0.0174  0.0109
  75,000               0.0232  0.0208  0.0283  0.0217  0.0313  0.0059  0.0517
 100,000               0.0198  0.0166  0.0182  0.0227  0.0139  0.0112  0.0540
```

At 100,000 ticks the **apex is the loosest level, not the tightest** (0.0540
against a graph median of 0.0192). #190 predicted this in the abstract —
"`02:113` dissolves with it" — and it is confirmed measured: the sentence has
nothing left to be true of.

## Where the 100x goes: the margins the rig places are not the margins the surface has

The construction sweep's binding margin is about `0.1369 / 8 = 0.017`. The built
surface's binding margin at 100,000 ticks is `0.00016` — **100x tighter** — while
the surface's *median* margin, 0.0192, sits right where the construction sweep's
tightest draw sat.

The cause is not subtle: **`bias_selection` places biases for a target timescale
and reads their margins at construction; the prediction rule then moves those
same biases for 100,000 ticks, and nothing re-checks the margin.** The margin is
a function of the bias, and the bias is the thing the prediction rule learns.

[#160](https://github.com/NGL321/patchworks/issues/160) asks whether a
construction-time check can stand against a floor that moves 144x in a run. This
is the same question from the other side, and the answer is worse: the **margin**
moves too, and further. `02` says the check is run once, at construction; at
construction it passes, and on the surface it runs on it fails, and both are
statements about the same check.

## The ruling on defect 2

**`gain_v` is the right subject, `gamma` is a shorthand, and it lives in `02`'s
prose alone.** The implementation has always used the gain:

- `bias_selection.FoldMarginCheck` says so in terms — *"The bound is read with
  the per-cell gain in it, `gain_v x floor < margin_v` ... Read without the gain
  the bound would be the same number at every cell and could not bind anywhere
  in particular"* — and `fold_margin_check` computes `margin x denominator`,
  which is that form.
- `prototypes/disagreement-floor/floor.py` writes `gain_v * floor < margin_v` in
  its own docstring and computes `offset = gain_v * floor`.

So this is **not a third defect in the mechanism**; it is one sentence of `02`
that never matched its own code. But it was never harmless either: under the old
denominator `gamma` and `gain_v` differed per cell by up to 2.6x, and the
shorthand is what hid that `02`'s "binds hardest at the apex" was a claim about
the *denominator's shape* rather than about the check.

**#190 makes the shorthand correct and its consequence false in the same
stroke.** With a flat denominator the two forms differ by a single global factor
of 8, so which one is written no longer changes *which* cell binds — only the
number. Correct `02` to `gain_v` anyway, and strike the depth claim with it.

## The actuator: the check does not reach it, and cannot

#190 asked that whatever ruled next "should say what happens at the actuator, or
say explicitly that it is leaving it". It is not left — it is out of reach, by
construction. `01-cell-and-sheaf.md:456`: *"A boundary cell runs no body."* No
body means no `encode`, no folds, and no fold margin. `fold_margin_check` runs
over `dome.predicting` and structurally cannot include the 264 boundary cells.

So #190's 12.0x at the actuator is untouched by this check in either direction:
the check neither licenses nor threatens it. Whatever bounds the actuator's
standing offset, it is not this.

## What is not trustworthy, and how far

**The surface run does not reproduce at a fixed seed.** Four independent runs,
all seed 42, `train`, real dome, same code, all read at 30,000 ticks:

| run | floor med | margin med | margin min | cap | binding cell | level | `gamma_cap` | cells < 1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04887 | 0.02484 | 0.00030 | 0.00242 | 62 | 1 | 0.0913 | 39 |
| 2 | 0.06382 | 0.02163 | 0.00009 | 0.00071 | 11 | 1 | 0.0189 | 39 |
| 3 | 0.05574 | 0.02035 | 0.00022 | 0.00176 | 146 | **7** | 0.0157 | 45 |
| 4 | 0.04859 | 0.02186 | 0.00057 | 0.00455 | 2 | 1 | 0.0381 | 31 |

**Four different binding cells and a 5.8x spread in `gamma_cap`, at one seed and
one budget.** The median margin over the same four runs spans **1.22x** and the
count of cells inside their own margin spans **1.45x**. This is the same split
the budgets gave, measured a second and independent way: the extremum is noise,
the population is signal.

`_divergence.py` rules out the checkpointing: a budget read with and without an
earlier checkpoint in front of it is bit-identical, so #178's claim that its
holds do not disturb the trajectory holds here too. `_readonly.py` rules out the
margin instrument the same way — a hold with the per-tick read and one without
agree exactly. What is left is float reduction order — torch runs 6 intra-op
threads here — amplified by 30,000 ticks of a chaotic surface.

**This is a fact about the surface, not about this rig**, and it bears on
anything in the record that quotes a per-cell extremum off a long run. #178's
own "wanders 3.8x with no trend" is the same phenomenon one level up, read as a
property of the budget when it is also a property of the run.

**One draw.** Seed 42, `train` split. #178's sweep found the split makes no
difference and the seed does, and nothing here re-measures that.

**#178's floors were read on a different body.** Its branch,
`worktree-apex-floor-asymptote-178`, has no `CellOperators` in `body.py` at all —
its 100,000-tick trajectory predates the conversion. Its apex 0.087 and its worst
cell 0.175 do not describe the body `action` ships, and the 30,000-tick control
cannot reproduce on it. The floors here are re-read on the current body for that
reason; the **horizon** #178 established is what this borrows, not its numbers.
