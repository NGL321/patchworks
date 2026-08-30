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

Median over all 150 cells falls from **8.96 to 0.062**. The control above predicted
exactly this: the untrained number was model error, and the bias rule is what removes
it.

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

## What this does not settle

The exact check `02` specifies joins each cell's floor to **its own** fold margin.
#155's margins were measured on drawn candidates rather than on the built cells, so
this pairs a measured per-cell floor against the apex's global cap instead. That is the
right comparison for the global `γ` and the wrong one for naming the binding cell.
Closing the gap means running `bias_selection.measure` on the built graph and joining
per cell.
