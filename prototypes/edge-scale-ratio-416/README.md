# The edge scale ratio against ADR-0010's admissible band (#416)

The reading [#416](https://github.com/NGL321/patchworks/issues/416) asks for, on the real dome.
`read.py` runs it, `summarise.py` prints the tables, and the `416-full-seed*-*.json` records are
the readings.

    python prototypes/edge-scale-ratio-416/read.py --seeds 0 1 2 --ticks 30000
    python prototypes/edge-scale-ratio-416/read.py --seeds 3 --ticks 100000
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

## The headline: everything ends up at the gauge

The one-sentence result, and the four runs all say it.

**The norms saturate the band, and once they have, the ratio is decided entirely by which ends are
pinned.** By 100,000 ticks **98.7%** of banded maps sit within 0.5% of `ρ` and 75.3% are at `ρ` to
three decimals. An interior edge then has both ends at `ρ` and reads ratio **1**; a boundary-incident
edge has one end at `ρ` and one nailed at 1, and reads ratio **exactly 2 — the band face**. Nothing
about either number is the objective matching anything.

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

Seed 3's own 30,000-tick checkpoint, inside the 100,000-tick record, reads 0.056 and 0.942 — a fourth
seed agreeing with the three.

In ratio terms at this horizon: an interior edge's median mismatch is **1.04x** against an admissible
**4x**; a boundary-incident edge's is **1.88x** against an admissible **2x**.

**Nothing is outside its band.** The largest excess measured is `2.5e-7` on a norm, which is float32
round-off in the projection's own clamp retaken in float64, not a breach. `summarise.py` separates
*on the face* from *beyond it* for exactly this reason — the first is the gauge binding, the second
would be the gauge failing, and only the first happens.

## The long horizon, 100,000 ticks — where the reading resolves

| tick | interior median | boundary-incident median | banded maps at `ρ` (±0.5%) |
|---|---|---|---|
| 0 | 6e-08 | 6e-08 | 0.0% |
| 100 | 0.031 | 0.016 | 0.0% |
| 1,000 | 0.038 | 0.101 | 0.7% |
| 3,000 | 0.043 | 0.213 | 2.2% |
| 10,000 | 0.062 | 0.483 | 7.5% |
| 30,000 | 0.056 | 0.942 | 15.8% |
| 100,000 | **0.0002** | **1.000** | **98.7%** |

At 100,000 ticks **all 273** boundary-incident edges are within 0.2% of the band face (the *minimum*
`|log₂|` over the population is 0.998), and 44% are on it to within `1e-5`. On the other side **97.6%**
of interior edges are within 1% of ratio 1.

The interior median falling from 0.056 at 30k to 0.0002 at 100k is not the ratio being corrected. It is
both ends arriving at the same ceiling.

## What it says

**ADR-0010's argument for leaving the ratio free is not what is holding the ratio.** The ADR argues the
objective handles it — *"minimising `‖F_u x_u − F_v x_v‖` on generic states wants matched scale"*, and
along the ratio *"the objective points away from collapse rather than toward it"*. The second half is
true and irrelevant: it rules out one-sided **collapse**, and the dynamics go the other way, to the
**ceiling**. The first half is not visible in the reading at all. What decides every edge's ratio at
the operating point is which of its ends the gauge pins:

- both ends free → both ride `ρ` → ratio **1**, and no matching pressure is needed to produce it;
- one end pinned → the free end rides `ρ` against a fixed 1 → ratio **`ρ` = 2**, and no matching
  pressure is able to prevent it.

**The boundary-incident edges are the discriminating population and the interior ones are not.** On an
interior edge the "objective matches scale" hypothesis and its negation predict the same reading, because
both ends grow together under Lemma 2.4's monotone growth of the **joint** scale — the direction the ADR
itself says nothing has an opinion about. On a boundary-incident edge that direction is unavailable: the
pinned end cannot move, so every unit of growth at the free end is a unit of *ratio*. It is the one place
the matching pressure is testable against a fixed reference, and it reads **absent** — the free end
climbs to `ρ` and stays.

**So the cost is structural, maximal and permanent.** 273 of 682 edges — the entire sensorimotor rim, the
drive, and the actuator — carry a **2x** unmatched endpoint scale, which is the whole of the freedom their
band allows. It is not a spread across the band; it is the band's face, occupied by the entire population.

**This is #411 §5's shape, one layer over.** #411 found disagreement descent buys agreement on the states
actually visited, and that rank-1 collapse was agreement achieved by shrinking what has to be agreed
about. Here the ratio pressure is not dodged so much as **not there**: the gauge determines the ratio and
the objective does not enter.

**ADR-0010's own drift claim, checked and upheld — at the long horizon only.** The ADR states *"the larger
end of every interior edge rides `‖F‖_F = ρ`"*. That is **true by 100,000 ticks** (98.7% of banded maps
within 0.5% of `ρ`) and **not yet true at 30,000** (15.8%). The claim is a long-horizon truth quoted
without a horizon; the `ρ²`-of-freedom figure that rests on it is sound at the horizon this reading
reaches, and a run stopping at 30k would find it false.

## Robustness: the same story on `σ_max`, and a much larger number on `σ_min`

`|log₂|` median at 30,000 ticks, seed 1 (0 and 2 agree to three digits).

| scale used | interior | boundary-incident |
|---|---|---|
| `‖F‖_F` (the gauge's own) | 0.056 | 0.909 |
| `σ_max` | 0.079 | 0.566 |
| `σ_min` | 0.162 | 1.338 (p95 **17.1**) |

`σ_max` tells the same story with a smaller boundary figure — the top-direction distortion is **1.49x**,
not the full 2x, because the free end spreads its Frobenius budget over 8 directions where the pinned
end does not. **The `σ_min` p95 of 17 is not this ticket's finding**: `2^17` says the spectra are nowhere
near flat, which is precisely the hole #411's `σ_min ≥ ‖F‖_F/√m` floor was ruled to close. It is carried
here only so the Frobenius/`σ` identification is not quietly assumed.

Flatness `σ_min/σ_max` per map reads median **0.40–0.42** at 30k and **0.55** at 100k — rising, but far
from the 1 #411 wants, so the identification remains an idealisation today rather than an identity.

## By edge kind, and by position on the channel

| kind | m | edges | band | median `|log₂|` at 30k |
|---|---|---|---|---|
| drive | 1 | 8 | `ρ` | 0.998 |
| sensory | 8 | 262 | `ρ` | 0.910 |
| motor | 8 | 3 | `ρ` | 0.756 |
| interior | 4 | 409 | `ρ²` | 0.056 |

The split is by **whether an end is pinned**, not by kind as such: every pinned-end kind saturates and
the one unpinned kind does not.

**Position on the channel, stated twice because the pooled figure misleads.** Rank correlation of
`|log₂ ratio|` with `d(edge, rim)`:

- **all 682 edges: −0.63 to −0.65** across all four runs. An artefact — every boundary-incident edge sits
  at depth 0 *and* at its band face, so the number reports that the two populations differ, not that
  anything varies along the channel.
- **interior only, 409 edges: +0.12 to +0.17.** Weak and positive: deeper interior edges carry slightly
  more mismatch at 30k. Per edge and keyed to a distance to the rim, never to the dome's imposed
  level (#181).

## The gate

#416 makes the reading the gate: *a ratio near 1 that holds means ADR-0010's argument is sound at the
operating point and nothing is owed; a ratio that drifts or spreads across the band is a second hole with
a measured cost, and then it wants a grilling to rule on a constraint.*

**The gate opens, and on a finding sharper than either branch anticipated.** The ratio does not sit near 1
and it does not spread across the band. It **drifts monotonically to the band face and stops there**, on
273 of 682 edges, permanently, and the reason is a composition of ADR-0010's own two decisions: the exact
gauge pins one end at 1, Lemma 2.4's monotone growth carries the other to `ρ`, and the projection holds it
there. The cost is a **2x** unmatched endpoint scale on every rim, drive and actuator edge — the whole of
the freedom the band allows those edges.

And the half that looks healthy is healthy for a reason that is not the ADR's reason. The interior ratio of
1 is both ends resting on the same ceiling, not the objective matching them, so it is not available as
evidence that the objective holds the ratio anywhere.

What this reading does **not** do is name the remedy. Whether the answer is `ρ = 1` at boundary-incident
edges, a matched-scale term, or accepting the 2x as the price of the exact gauge is a ruling, and
ADR-0029's deferral rule says a read does not get to make it. That is the ticket this hands on.
