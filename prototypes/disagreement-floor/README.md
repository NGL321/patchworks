# The per-level disagreement floor (#158)

`02-tick-semantics.md` bounds reconciliation with `γ × floor < fold margin`, checked
per cell at construction. [#155](https://github.com/NGL321/patchworks/issues/155)'s
precondition 1 measured the margin side and found the bound resolves not to a factor
but to a function of a number nobody had measured:

```
recoverable = min(5.585, cap / (γ · floor)),   cap = 0.3278 post-#157, apex-binding
```

This directory derives the floor.

## What is measured, and in what units

The bound guards a cell's **operating point**. Reconciliation's standing offset on a
node stalk is

```
offset_v  =  gain_v · ‖ Σ_{e∋v} F_evᵀ (F_ev x_v − y_e(t−1)) ‖
```

and `tick.py`'s message-passing phase computes that vector, multiplies it by `gain_v`
and subtracts it. `floor` is therefore the norm of the sum **before** the gain — which
is exactly what `bias_selection.FoldMarginCheck` divides its `product_cap` by. Nothing
here reconstructs it: the sheaf keeps both terms of the difference (`broadcast`,
`incoming`), so the number read is the one the run applied.

## How to run it

Inside the supported container (ADR-0012):

```
docker run --rm -v "$PWD:/work" -w /work --entrypoint python patchworks:headless \
    prototypes/disagreement-floor/floor.py <subcommand>
```

| subcommand | what it does |
|---|---|
| `hold` | settle, then hold the world still (ADR-0007's protocol) and read the floor per level. `--learn N` reads it on a surface that has had `N` ticks of both rules instead. |
| `jitter` | the same quantity's tick-to-tick variation with the sandbox live. `--impulse` disturbs the arm first, because the untrained fixed point is a locked one. |
| `trajectory` | (#178) the same read at nine budgets out to 100,000 ticks, on **one** run. Repeated `hold --learn N` costs the sum of the budgets; this costs the largest. |
| `sweep` | the hold across splits and seeds — a static floor is positional, and one pose reports on one point of the overlap. |
| `optimum` | the structural control: the whole-graph minimum achievable energy, and what the delta does when the body is suppressed and reconciliation runs alone. |

## What it found

Real dome, `benchmarks/timescale_selection.py`'s seed 42, 5000 ticks to settle, 400
ticks of quiescent hold.

**The floor, per level** (median un-gained delta over cells):

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| seed 42 | 11.03 | 9.01 | 7.39 | 9.10 | 7.29 | 6.30 | 9.16 |
| pooled, 3 seeds × 3 splits | 3.83 | 4.08 | 3.76 | 3.39 | 3.05 | 3.90 | 4.05 |

Per-cell range across the sweep: **2.21 to 33.30**. The verdict is the same at every
level and under every draw — the floor sits in #155's **third row**, `≥ 0.328`, by
between 10x and 100x. `γ_cap` lands at **0.03–0.11**, against the `γ = 1.0` that
`tick.py:71` ships.

**Three secondary findings, each of which is about the instrument rather than the
number:**

1. **Nothing drains during the hold.** The untrained fixed point is *already*
   quiescent — [#120](https://github.com/NGL321/patchworks/issues/120)'s locked arm —
   so the lag floor is zero there before the hold starts, and ADR-0007's separator
   separates nothing at this configuration.
2. **The split makes no difference; only the seed does.** Numbers agree to five
   decimals across `train` / `heldout_pair` / `heldout_sector` at a fixed seed. The
   taper is why: what the split changes is where the pucks are, and that never reaches
   the cells. So at construction the "static floor" is not positional — it is a
   property of the draw.
3. **The jitter check passes by four to six orders of magnitude.** Median tick-to-tick
   change is `3e-6` to `2e-4` against a floor of ~9, still world and disturbed alike.
   A floor under the rig's noise is not a floor; this one is nowhere near it.

## Learning moves it by 144x, and that is the answer

`hold --learn 30000` — the same read on a surface that has had 30,000 ticks of both
rules:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 (apex) |
|---|---|---|---|---|---|---|---|
| untrained | 11.03 | 9.01 | 7.39 | 9.10 | 7.29 | 6.30 | 9.16 |
| after 30k ticks | **0.055** | **0.058** | 0.076 | 0.146 | 0.111 | **0.057** | **0.217** |
| recoverable | 5.59x | 5.59x | 4.31x | 2.25x | 2.95x | 5.59x | **1.51x** |

Median over all 150 cells falls from **8.96 to 0.062**. The control below predicted
exactly this: the untrained number was model error, and the bias rule is what removes
it.

**The medians clear; the tail does not.** The trained distribution is badly skewed at
mid-depth — level 4 reads median 0.146, p95 **2.40**, max **3.85**, and levels 3 and 5
have the same shape. The tightest cell caps the global `γ` at **0.085** against the
apex's cap, or ~0.125 against level 4's own, which is the fairer pairing and still
nowhere near 1. `γ` is one global scalar (`tick.py:71`), so those few cells set it: the
typical cell clears the bound and a handful of mid-depth cells do not.

Two things the trained read shows that the untrained one could not:

- **ADR-0007's quiescent hold works now, and did nothing before.** Driven, the median
  is 0.110; ten ticks into the hold it is 0.055. That difference *is* the lag floor,
  draining. Untrained it drained nothing, because #120's arm never moved — the
  separator needs a graph that transmits before it separates anything.
- **The apex does not drain.** Levels 1–6 fall by about half into the hold; level 7
  sits at 0.212–0.217 throughout. Its floor is static, and it is the level `02` says
  should bind.

## The finding that matters more than the number

`optimum`, run at the same configuration:

```
minimum achievable Dirichlet energy: 30.79
total edge energy now:               32773        (1064x the minimum)

reconciliation alone, the inference phase suppressed:
  step      median delta     total energy
     0           8.9556          32772.8
    10           6.5260          21278.2
   100           1.2748           2232.0
   200           0.5743            649.7
```

The minimum achievable energy is the disagreement no node-stalk assignment can clear —
ADR-0007's static floor, measured. But **the gradient of that energy is zero at its own
minimiser**, so an irreducible *energy* produces no standing *offset* at all. And the
control shows the graph is not near that minimiser: it sits at 1064x it, and the moment
the body stops re-predicting, the delta collapses toward zero.

So the quantity `02`'s bound calls `floor` is **not** ADR-0007's disagreement floor. It
is the per-tick reconciliation step magnitude at the running operating point, and at
construction it is dominated by **model error** — the residue ADR-0007 defines the
floors *against*. ADR-0002 is what sustains it: one step, not a solve, means the body
puts the operating point back off consensus every tick faster than one Jacobi step
pulls it back.

Which is why the trained read is the one to price against, and why the number is not a
construction-time constant. `02-tick-semantics.md` says the check is run once, at
construction, and ADR-0007 was amended by
[#37](https://github.com/NGL321/patchworks/issues/37) to *strike* the running
re-derivation — on the grounds that ADR-0010's gauge stops the **denominator**
drifting. That argument is sound and it does not reach this: what drifts here is the
**floor**, by 144x over 30,000 ticks, in the direction of safety but not by any bound
the record holds.

## Where it settles: 30,000 ticks was not the answer (#178)

Everything above is read at one budget, and the closing paragraph records that the
floor was **still falling** there. [#178](https://github.com/NGL321/patchworks/issues/178)
asked where it lands. `trajectory`, one run to 100,000 ticks — #120's own long-run
budget — read at nine budgets along the way:

| ticks | 1000 | 2000 | 5000 | 10000 | 20000 | **30000** | 50000 | 75000 | 100000 |
|---|---|---|---|---|---|---|---|---|---|
| apex | 0.243 | 0.364 | 0.096 | 0.151 | 0.246 | **0.217** | 0.097 | 0.104 | **0.087** |
| all-cell median | 0.310 | 0.292 | 0.207 | 0.115 | 0.106 | **0.062** | 0.046 | 0.039 | **0.040** |
| tightest cell | 3.02 | 14.9 | 13.6 | 15.4 | 16.1 | **3.85** | 0.837 | 0.263 | **0.175** |
| `γ_cap` | 0.109 | 0.022 | 0.024 | 0.021 | 0.020 | **0.085** | 0.392 | **1.000** | **1.000** |

The 30,000 column is the control and it reproduces the run above to five decimals —
apex 0.21728, all-cell 0.06214, level 4 median 0.14574 with p95 2.40283 and max
3.85384. The checkpointing does not disturb the trajectory it measures.

**Three findings, and the third is the one that moves.**

1. **The apex does not descend; it wanders, then settles.** Between 1k and 30k it
   ranges 0.096 to 0.364 — a **3.8x** spread with no trend, and 30,000 ticks happened
   to land near a local *high*. The last three reads are 0.097 / 0.104 / 0.087, a band
   an order of magnitude tighter. That narrowing, not any single value, is what says
   the quantity has settled.
2. **It settles above the break-even, but not far above.** 0.0866 against 0.0587 is
   **1.47x**, so the apex recovers `0.3278 / 0.0866` = **3.79x** of #155's 5.585x
   rather than the 1.51x that 30,000 ticks reported. Against the post-#157 cap of
   0.3502 the figure is **4.05x**.
3. **The mid-depth tail is gone.** This is the finding above that does *not* survive
   the longer budget. At 30k a handful of mid-depth cells read p95 2.40 and max 3.85
   and capped the global `γ` at **0.085**; by 75k the worst cell in the whole graph is
   0.263 and by 100k it is 0.175, so `γ_cap` is **1.0** and nothing caps `γ` below the
   1.0 that `tick.py:71` ships. The binding level moves to the apex — the level `02`
   says should bind — and the constraint stops being on `γ` at all.

The non-monotonicity is worth carrying to
[#160](https://github.com/NGL321/patchworks/issues/160) on its own account. That ticket
asks whether a construction-time check can stand against a quantity that moves 144x in
a run; the answer here is that it does not even move *monotonically*, so a check run at
one budget has ~3.8x of scatter under it independent of where the floor ends up.

**One draw.** Seed 42, `train` split, and the pre-#157 body — the same base as the run
above, which is what makes the 30k control valid. The sweep found the split makes no
difference and the seed does, so the 100k figures are one trajectory, not a
distribution over draws.

## What this does not settle

The exact check `02` specifies joins each cell's floor to **its own** fold margin.
#155's margins were measured on drawn candidates rather than on the built cells, so
this pairs a measured per-cell floor against the apex's global cap instead. That is the
right comparison for the global `γ` and the wrong one for naming the binding cell.
Closing the gap means running `bias_selection.measure` on the built graph and joining
per cell.
