# What the sparsity pressure buys, and what it costs the channel (#393)

The `λ` sweep [#393](https://github.com/NGL321/patchworks/issues/393) asks for, on
the real dome. `sweep.py` runs it, `read.py` prints the curve, and the
`393-lam*-seed*-*.json` records are the readings.

    python prototypes/sparsity-lambda-393/sweep.py --lams 0 0.1 0.2 0.4 0.8 --seeds 0 1 2 --ticks 3000
    python prototypes/sparsity-lambda-393/read.py

**One training run per `(λ, seed)`, three readings off the same surface.** All
three quantities #393 names are reads of a trained dome and training is the
entire cost, so reading them separately would pay for the run three times and
report three numbers from three different surfaces -- not the joint reading the
ticket asks for. The three are `driven_settling`'s `draining_effective_rank`
(with its `..._opening` baseline), `Diagnostics.whole_graph`'s `rank δ` /
`dim H⁰`, and `detectability`'s conduction ratio in both directions.

## The curve, 3,000 ticks, three seeds

| `λ` | eff rank | (opening) | `dim H⁰` | `rank δ` | `dim H¹` | conduction in/out |
|---|---|---|---|---|---|---|
| 0 | **3.191** | 3.661 | 1484 | 3316 | 448 | 0 / 0 |
| 0.1 | **2.877** | 3.661 | 1484 | 3316 | 448 | 0 / 0 |
| 0.2 | **2.437** | 3.662 | 1484 | 3316 | 448 | 0 / 0 |
| 0.4 | **1.844** | 3.660 | 1494 | 3306 | 458 | 0 / 0 |
| 0.8 | **1.290** | 3.659 | 1510 | 3290 | 474 | 0 / 0 |

Per-`λ` rows are the median over seeds; per-seed rows are in `read.py`'s output
and the spread is tight (eff rank varies by <0.06 across seeds at every `λ`).
Every map is transmitting at 1364/1364 at every `λ`, so this is not norm
collapse being misread as rank collapse.

**The rig reproduces the record.** `λ = 0.4` at 3,000 ticks reads **1.844**
against [#324](https://github.com/NGL321/patchworks/issues/324)'s published
**1.85** at the same length. That is the control on the whole sweep.

**`χ` holds exactly at every point.** `dim H⁰ − dim H¹ = 1036` at all 15 records
-- the free correctness check #393 asks for, asserted in `sweep.py` rather than
eyeballed, so a mis-assembled reading fails loudly.

## What it says

**The pressure buys concentration, not cohomology.** Effective rank falls
**2.5x**, 3.19 to 1.29, across the swept range. `dim H⁰` over the same range
moves **1484 to 1510** -- **+1.8%**. Below `λ = 0.3` it buys *literally none*:
`dim H⁰` is 1484 and `rank δ` is 3316 at `λ = 0`, `0.1` and `0.2`, at every
seed, an exact tie.

That is not a paradox, it is what the two instruments are. The participation
ratio is a *continuous* measure of how concentrated a map's singular values are;
`rank δ` counts how many clear a numerical tolerance. A map driven toward one
dominant direction scores near 1 on the first while its small singular values
stay comfortably above the second's threshold. **Near-rank-1 is not rank-1, and
`H⁰` only grows on the genuine rank-deficiency.**

So `06-graph-topology.md`'s stated intent -- a dead edge *"through learned
rank-deficiency still enlarges `H⁰`, which is the effect `05-timescales.md`
actually wanted from sparsity"* -- is **not** what the pressure is delivering at
this length. It delivers the concentration; the enlargement it was wanted for is
1.8% at twice the default and zero below it.

**There is no knee.** Effective rank falls smoothly and near-linearly in `λ`
across the whole range. Nothing in the rank curve marks `0.4` out.

## The long arm, 30,000 ticks, seed 0

`λ = 0`, `0.4` and `0.8` re-run at #237's length, because the 3,000-tick arm
cannot separate "`λ` closed the channel" from "the channel was never open".

| `λ` | eff rank | (opening) | `dim H⁰` | `rank δ` | conduction in/out | bottleneck in/out |
|---|---|---|---|---|---|---|
| 0 | **2.913** | 3.653 | 1484 | 3316 | 0 / 0 | 7.6e-08 / 2.2e-07 |
| 0.4 | **1.002** | 3.652 | 1503 | 3297 | 0 / 0 | 5.9e-07 / 4.5e-06 |
| 0.8 | **1.000** | 3.646 | 1496 | 3304 | 0 / 0 | 9.7e-07 / 2.1e-06 |

**The second control passes.** `λ = 0.4` at 30,000 ticks reads **1.002** against
[#237](https://github.com/NGL321/patchworks/issues/237)'s published **1.0009**.

**`λ` is the cause of the rank collapse.** At `λ = 0` the fleet holds at
**2.913** after 30,000 ticks where the default drives it to **1.002**. #324's
slide toward 1 is the sparsity pressure's doing and not an inevitability of the
transport rule -- turning the pressure off preserves nearly three effective
directions over the same run. That is the cleanest thing this sweep establishes.

**And it still buys almost no `H⁰`, even at total collapse.** At `λ = 0.4` the
median map is at effective rank **1.002** -- as concentrated as the instrument
can read -- and `dim H⁰` is **1503** against 1484 at `λ = 0`. **+19 dimensions
out of 1484, +1.3%**, for a fleet that has given up two thirds of its effective
rank. `λ = 0.8` buys *less* (1496) than `λ = 0.4` despite lower effective rank.
The conversion `dim H⁰ = dim C⁰ − rank δ` is exact, but `λ` does not move
`rank δ`: it moves the *conditioning* of the maps, not their rank, so the
conversion never fires.

## What this sweep cannot settle, and why

**The conduction column reads 0 in every configuration and cannot price the
channel.** 0 at every `λ`, both directions, **0 of 24 trials nonzero** at every
point, at 3,000 *and* 30,000 ticks, and 0 on an untrained surface too.

That is not `λ` closing the channel. `detectability`'s own printout attributes
such a zero to its cause: *"1 of that path's cells hold no private features, so
their `tau_hat` is 0 by construction, not by measurement
([#385](https://github.com/NGL321/patchworks/issues/385))"*. ADR-0026's
reduction is a `min` over the cells of a path, so a single structurally-zero
cell on the widest path pins the whole reading at 0 regardless of what the
sparsity pressure did. **The instrument is saturated at its floor**, and a
column that reads 0 where the independent variable is *off* carries no
information about the independent variable.

So the half of #393 that asks what the pressure *costs the channel* is not
answered here and is not answerable by this instrument as it stands -- it waits
on #385. What can be said is the weaker, real thing: **ADR-0026's bar is
unsatisfiable at every `λ` swept, including 0**, so there is no setting of this
dial at which both ends of the map's bar survive.

The **bottleneck ratio**, ADR-0021's demoted amplitude diagnostic, is not
saturated and moves the *opposite* way from the assumed trade: it *rises* with
`λ`, 7.6e-08 to 5.9e-07 inbound (~8x). Concentrating a map into one direction
makes that direction transmit better. It remains six orders of magnitude below
its bar, so this buys nothing -- but it is evidence against "sparsity is spent
out of transmission" as a mechanism, rather than for it.

## Notes on the rig

**A rig asserts nothing** (`benchmarks/run_reporting.py`). This reports a curve;
it files no cutoffs and touches no register. `detectability`'s `read` would file
against #325/#329/#341, which is why its parts are called directly instead.

**Per edge and per direction, never a graph-wide average** (#127's standing
rule). The rank reading is a median over the population of *maps* on draining
edges -- two per edge, uncollapsed, `driven_settling`'s own convention -- with
per-map quantiles published beside it. Conduction is stated twice.

**This is a prototype, not a published rig.** Promoting `sweep.py` to
`benchmarks/` is an edit, and #127's standing note is that an edit hands off to
its own ticket.
