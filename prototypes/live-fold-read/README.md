# The live fold read, run (#202)

[ADR-0019](../../docs/adr/0019-construction-nominates-the-run-decides.md) moved the fold-margin check
off construction and onto the run; [#197](https://github.com/NGL321/patchworks/issues/197) built the
instrument, `patchworks.tick.FoldRead`, and deliberately did not run it.
[`02-tick-semantics.md`](../../docs/spec/02-tick-semantics.md) states the bound as holding **after a
burn-in** and names no count. This directory runs the instrument and reports the count.

**It reports that there is no count.**

```
PYTHONPATH=src python prototypes/live-fold-read/read.py --ticks 100000
```

One run, `real` dome, `split=train`, `seed=42`, both rules on, 150 predicting cells, 21 minutes.
Results in `202-read.json` (checkpoint tables) and `202-per-tick.npz` (per-tick and per-cell arrays).

## The bound has no second gain in it

`02` writes the bound `gain_v × offset < margin_v`. That form is inherited from the demoted
`γ × floor` bound, in which the divisor was the **un-gained** disagreement floor.
`FoldRead.offset` is read off the **gained** displacement — `tick.py`'s message-passing phase applies
`_gain_per_component` and *then* observes — so the live comparison is `offset < margin` directly, which
is what `FoldRead.reconciliation_reaches` computes. Applying `gain_v` a second time would divide by 32
again and report a burn-in belonging to no run.

`read.py`'s `check_gain_convention` asserts this on the run rather than trusting a reading of the
source: it reconstructs the un-gained delta the way #178's `per_cell_floor` does and checks the ratio.

```
FoldRead.offset / un-gained delta = 0.031222,  gain_v = 0.031250   -> gained
```

**Both sides implement the same inequality; only the word collides.** `bias_selection.FoldMarginCheck`
forms `product_cap = margin × denominator`, so its `gamma_cap(offset)` expects the **un-gained**
quantity, while `FoldRead.offset` holds the **gained** one. The two are a factor of `gain_v` = 1/32
apart, and passing one into the other silently mis-states a nomination by 32x. Nothing in the tree does
that today. Noted because the two APIs are the nomination and the verdict, and comparing them is the
whole of ADR-0019.

## 1 — The burn-in does not exist

Every one of the 100,000 ticks carried at least one breaching cell. All 150 cells breached at least
once; 109 of 150 were still breaching after tick 90,000; the median cell's **last** breach was at tick
95,976.

| decade | mean breaching cells | ticks with any breach |
|---|---|---|
| 0–1,000 | 32.87 | 100% |
| 1,000–2,000 | 31.15 | 100% |
| 2,000–5,000 | 31.18 | 100% |
| 5,000–10,000 | 29.44 | 100% |
| 10,000–20,000 | 26.03 | 100% |
| 20,000–50,000 | 18.01 | 100% |
| 50,000–100,000 | 16.18 | 100% |

The breach **density** falls — 33 cells to 16 — and then **plateaus**. It does not approach zero. In
the second half of the run the number of breaching cells sits at p05 11, median 16, p95 22, and not one
tick in 50,000 was clean.

So `02`'s sentence is false as written. There is no tick beyond which `offset < margin` holds. A count
can only be recovered by restating the claim as a **tolerance** — *no more than k cells breach* — and
even then the quantity being tolerated settles at ~16/150 rather than decaying.

This is #178's finding a third time: read to 30,000 the breach count is 15 and falling, which invites
exactly the extrapolation to zero that the 100,000-tick read refuses.

**No depth structure**, as #190 and #160 predicted by striking the depth claim. Median breach fraction
by level runs 9.7 / 12.7 / 13.3 / 15.4 / 12.5 / 10.1 / 9.0 % down levels 1–7 — no trend, and the
binding cells are scattered (levels 6, 5, 4, 1, 1, 3, 4, 4, 1, 5 for the ten worst).

## 2 — Reconciliation does cost cells their regions

The comparison ADR-0007 asks for, read **within each cell** so that a cell's own margin cannot produce
the association by itself:

```
P(crossing | breach on the previous tick) / P(crossing)
    median 3.66x    p25 2.31x    p75 6.36x    range 0.77x - 22.5x
    above 1.0x on 145 of 150 cells; above 1.5x on 130
```

When reconciliation reaches, the cell is **~3.7x more likely to lose its activation region on the next
tick**. That is ADR-0007's named failure, observed rather than bounded.

**Read the mechanism before reading the number as a surprise.** A breach *is* the displacement being at
least the distance to the nearest fold, so a crossing following one is partly definitional. The
instrument's own docstring says why the factor is 3.7 and not infinite: the displacement's direction may
point across the region rather than out of it. What the number settles is the question the record left
open — whether the arrangement sliding under the operating point *ever actually* costs a cell its
region. It does, routinely, and it is still doing so at tick 100,000.

## 3 — ADR-0005's precondition holds, and now it is falsifiable

#160 re-sourced the precondition off the margin proxy and onto measured dwell, so this is the first
reading of it on a run. Dwell is measured in a window between checkpoints, against each cell's own
`τ = −1/ln ρ(K · J_encode)` in the region it is in — the same round trip
`bias_selection.measure` reads on the construction sweep.

| ticks | τ median | window dwell median | dwell/τ median | breaching |
|---|---|---|---|---|
| 100 | 1.213 | 1.01 | 0.96 | 39 |
| 1,000 | 1.184 | 1.03 | 1.00 | 29 |
| 2,000 | 1.182 | 1.26 | 1.22 | 29 |
| 5,000 | 1.143 | 2.03 | 1.65 | 32 |
| 10,000 | 1.105 | 7.23 | 6.43 | 25 |
| 20,000 | 1.082 | 12.18 | 11.06 | 23 |
| 30,000 | 1.022 | 35.19 | 35.16 | 15 |
| 50,000 | 1.020 | 57.14 | 56.85 | 17 |
| 75,000 | 1.040 | 68.69 | 71.79 | 15 |
| 100,000 | 1.009 | 85.33 | 82.68 | 16 |

At the horizon **130 of 150 cells clear `dwell ≥ 2.6 τ`** (`DEFAULT_SAFETY_FACTOR`), and **15 of 150 sit
at dwell ≤ 2 ticks** — precisely the *"dwell collapses to a tick or two"* failure `05` names, where the
cell decays at an average rate by averaging over unrelated regions. Eleven of those 15 are among the
109 cells still breaching after 90,000 ticks: the precondition's failures and the bound's are largely
the same cells.

**Early in the run the precondition genuinely fails** — dwell/τ is 0.96 at tick 100 and does not pass
2 until somewhere past 2,000 ticks. It is earned over the run rather than held from the start, which is
the shape ADR-0019 argued for; what ADR-0019 did not anticipate is that the *other* half of the read,
the breach, never finishes arriving.

**Read the ratio's two halves separately.** dwell/τ reaches 83x because **dwell grows 85x**, not because
τ moves: τ is flat at ~1.0–1.2 ticks for the whole run. A τ of one tick is what makes an 85-tick dwell
83 τ. Whether a one-tick τ is what construction meant to place is **not settled here** — it is a
question about `K` and the placed band, which is #143's, and this read only establishes that τ on the
run is ~1 tick and does not drift.

**30,000 ticks would have understated dwell/τ by 2.4x** (35 against 83). #178's horizon finding governs
this quantity too.

## What this does not do

- It does not re-open the burn-in as a ramp or a lower `γ`. ADR-0019 declined both; this is the
  measurement that decision asked for. What it reports is that the count the decision assumed exists
  does not, which is a fact for that decision to be re-read against — not a re-litigation of it here.
- It does not settle the unpinned per-cell bias. It supplies the data that question waited on.
- It does not swap the gain's denominator, which is
  [#195](https://github.com/NGL321/patchworks/issues/195)'s.
