# The edge scale ratio against ADR-0010's admissible band (#416)

The reading [#416](https://github.com/NGL321/patchworks/issues/416) asks for, on the real dome.
`read.py` runs it, `summarise.py` prints the tables, and the `416-full-seed*-*.json` records are
the readings.

    python prototypes/edge-scale-ratio-416/read.py --seeds 0 1 2 --ticks 30000
    python prototypes/edge-scale-ratio-416/summarise.py --ticks 30000

**The quantity.** Per edge, per direction, `σ(F_u)/σ(F_v)` — the ratio of the two endpoint maps'
scales, which [ADR-0010](../../docs/adr/0010-restriction-map-scale-is-gauge-fixed.md) leaves free on
purpose and [#411](https://github.com/NGL321/patchworks/issues/411) gave an opinion about.

**The scale is `‖F‖_F`, and the rig does not get to choose that.** ADR-0010's gauge is *stated* on
the Frobenius norm, so the Frobenius norm is what the band constrains. And under #411's flat spectrum
the two readings are the same number: `m` equal singular values give `σ = ‖F‖_F/√m`, an edge's two
ends share one `m` because it is a property of the edge stalk, so `σ_u/σ_v = ‖F_u‖_F/‖F_v‖_F`
**identically**. `σ_max` and `σ_min` ratios are carried beside it as the robustness check, because
the spectra are not flat today.

**Spreads are `|log₂ ratio|`.** A ratio and its reciprocal are one mismatch read from the two ends,
so a spread over directed ratios would sit near 1 however large the mismatches got. In these units
1 is a factor of 2 and 2 is a factor of 4 — so the interior band `ρ² = 4` is **2**, and the
boundary-incident band `ρ = 2` is **1**.

## Two bands, not one — and #416's `ρ² = 4` is only the interior one

This is the correction the reading turns on, and it is arithmetic rather than a finding.

| population | edges | free ends | admissible ratio |
|---|---|---|---|
| interior | 409 | both carry the band | `ρ² = 4` |
| boundary-incident (sensory 262, drive 8, motor 3) | 273 | one pinned at the **exact gauge** | `ρ = 2` |

A boundary cell's own maps are pinned at exactly `‖F‖_F = 1`, so on those 273 edges only one end can
move and the admissible ratio is `ρ`, not `ρ²`. Pooling the two populations reports the pinned ones as
using a quarter of their freedom when they are using half of it. The rig never pools them.

**Untrained is 1 by construction, not by measurement.** `INITIAL_NORM = 1.0` draws every map at the
band's geometric centre, in `restriction.py`'s own words *"so no edge starts with a scale ratio built
into it"*. The construction reading is `6e-08` — float32 round-off around exactly 1 — and it is the
zero of the drift axis, not evidence. The whole reading is about what training does.

## The reading, 30,000 ticks, three seeds

`|log₂ ratio|`, per edge, never pooled across the two populations.

| population | seed | median | p75 | p95 | max | band | p95/band | on the face |
|---|---|---|---|---|---|---|---|---|
| interior (`ρ²=4`) | 0 | 0.061 | 0.135 | 0.281 | 0.726 | 2 | **0.140** | 0.0% |
| interior (`ρ²=4`) | 1 | 0.056 | 0.116 | 0.227 | 0.587 | 2 | **0.113** | 0.0% |
| interior (`ρ²=4`) | 2 | 0.053 | 0.110 | 0.220 | 0.469 | 2 | **0.110** | 0.0% |
| boundary-incident (`ρ=2`) | 0 | **0.915** | 1.000 | 1.000 | 1.000 | 1 | **1.000** | 18.3% |
| boundary-incident (`ρ=2`) | 1 | **0.909** | 1.000 | 1.000 | 1.000 | 1 | **1.000** | 18.7% |
| boundary-incident (`ρ=2`) | 2 | **0.914** | 0.998 | 1.000 | 1.000 | 1 | **1.000** | 16.8% |

In ratio terms: an interior edge's median mismatch is **1.04x** against an admissible **4x**; a
boundary-incident edge's is **1.88x** against an admissible **2x**, with ~18% of them sitting
*exactly on* the band face.

**Nothing is outside its band.** The largest excess measured is `2.5e-7` on a norm, which is float32
round-off in the projection's own clamp retaken in float64, not a breach. `summarise.py` separates
*on the face* from *beyond it* for exactly this reason — the first is the gauge binding, the second
would be the gauge failing, and only the first happens.

## Does it drift? Yes on one population, barely on the other

`|log₂ ratio|` median, at every checkpoint, median over the three seeds.

| tick | interior | boundary-incident |
|---|---|---|
| 0 | 6.4e-08 | 6.0e-08 |
| 10 | 0.016 | 0.006 |
| 100 | 0.025 | 0.014 |
| 300 | 0.030 | 0.034 |
| 1,000 | 0.034 | 0.098 |
| 3,000 | 0.036 | 0.210 |
| 10,000 | 0.050 | 0.438 |
| 30,000 | 0.056 | **0.914** |

The boundary-incident ratio **roughly doubles per decade of ticks** and is stopped by nothing except
the band face. The interior ratio adds ~0.01 per decade and is at 2.8% of its own band's width.

## What it says

**On interior edges, ADR-0010's argument holds at the operating point.** The ratio sits at 1.04x of an
admissible 4x, uses 11–14% of the band at the p95, and creeps at a rate that would not reach the face
in any run this project will do. Read alone, that is #416's *record it and close* case.

**On boundary-incident edges it is saturated, and the objective is not what is holding it.** The ratio
drove monotonically from 1 to the band face over 30,000 ticks and stopped there because the projection
stopped it. 273 of 682 edges — the entire sensorimotor rim, the drive, and the actuator — are pinned
against the constraint.

**Why that is the discriminating population, and the interior one is not.** ADR-0010 argues the
objective wants matched scale: *"minimising `‖F_u x_u − F_v x_v‖` on generic states wants matched
scale"*. On an interior edge that hypothesis and its negation predict the same reading — both ends grow
together under Lemma 2.4's monotone growth of the **joint** scale, which is the direction nothing has an
opinion about, so a ratio near 1 falls out whether or not any matching pressure exists. On a
boundary-incident edge the pinned end **cannot move**: it is nailed at 1 by the exact gauge. There the
joint direction is unavailable and every unit of growth at the free end is a unit of *ratio*. That is
the one place the matching pressure is testable against a fixed reference, and it reads **absent** —
the free end climbs to `ρ` regardless.

So the interior edges' stability is not evidence that the objective holds the ratio. It is evidence
that both ends grow at the same rate, which is Lemma 2.4 and not the transport rule having an opinion.

**This is #411 §5's shape, one layer over.** #411 found disagreement descent buys agreement on the
states actually visited, and that rank-1 collapse was agreement achieved by shrinking what has to be
agreed about. The ratio pressure is not dodged here so much as **not there**: ADR-0010's triangle-
inequality argument rules out one-sided *collapse* (sending either map to zero reads 1, the maximum)
and says nothing whatever about the ratio riding to the **ceiling**, which is the direction the
dynamics actually take.

**A third finding against ADR-0010, on the way past.** The ADR states *"the larger end of every interior
edge rides `‖F‖_F = ρ`"*, and that is the premise the `ρ²`-of-freedom figure rests on. Measured, the
banded maps have median norm **1.71** and only **5.5–6.1%** are at `ρ` — growth is real and the ceiling
is reached by a small minority, not by every edge's larger end. The claim is directionally right and
quantitatively false as written.

## Robustness: the same story on `σ_max`, and a much larger number on `σ_min`

`|log₂|` median at 30,000 ticks, seed 1 (0 and 2 agree to three digits).

| scale used | interior | boundary-incident |
|---|---|---|
| `‖F‖_F` (the gauge's own) | 0.056 | 0.909 |
| `σ_max` | 0.079 | 0.566 |
| `σ_min` | 0.162 | 1.338 (p95 **17.1**) |

`σ_max` tells the same story with a smaller boundary figure — the top-direction distortion is **1.49x**,
not the full 2x, because the free end spreads its Frobenius budget over 8 directions where the pinned
end does not. **The `σ_min` p95 of 17 is not this ticket's finding**: `2^17` is a statement that the
spectra are nowhere near flat, which is precisely the hole #411's `σ_min ≥ ‖F‖_F/√m` floor was ruled to
close. It is carried here only so the Frobenius/`σ` identification is not quietly assumed.

Flatness `σ_min/σ_max` per map reads median **0.40–0.42** across the three seeds, which is what makes
that identification an idealisation today rather than an identity.

## By edge kind, and by position on the channel

| kind | m | edges | band | median `|log₂|` |
|---|---|---|---|---|
| drive | 1 | 8 | `ρ` | 0.998 |
| sensory | 8 | 262 | `ρ` | 0.910 |
| motor | 8 | 3 | `ρ` | 0.756 |
| interior | 4 | 409 | `ρ²` | 0.056 |

The split is by **whether an end is pinned**, not by kind as such: every pinned-end kind saturates and
the one unpinned kind does not.

**Position on the channel, stated twice because the pooled figure misleads.** Rank correlation of
`|log₂ ratio|` with `d(edge, rim)`:

- **all 682 edges: −0.64.** This is an artefact — every boundary-incident edge sits at depth 0 *and* at
  its band face, so the number reports that the two populations differ, not that anything varies along
  the channel.
- **interior only, 409 edges: +0.12 to +0.17.** Weak and positive: deeper interior edges carry slightly
  more mismatch. Per edge and keyed to a distance to the rim, never to the dome's imposed level (#181).

## The gate

#416 makes the reading the gate: *a ratio near 1 that holds means nothing is owed; a ratio that drifts
or spreads across the band is a second hole with a measured cost.*

**The reading is split, and the half that saturates is the half that discriminates.** On interior edges
nothing is owed. On the 273 boundary-incident edges the ratio drifts monotonically to the band face and
is held only by the projection — and those edges are the *only* place the objective's claimed matching
pressure can be tested, because they are the only place one end is fixed. So the gate opens: this is a
second hole with a measured cost, and the cost is a **2x** unmatched scale on every rim, drive and
actuator edge, arrived at by drift rather than sitting there from construction.

What it is *not* is a reason to invent a constant. The measured mechanism is specific — ADR-0010's own
two decisions, the exact gauge at the boundary and monotone growth of the free end, compose into
guaranteed ratio saturation on exactly the edges that have one of each — and naming the remedy is a
ruling, not a read. That is the ticket this reading hands on.
