# B44 (#610): the O(1) return does not bend `τ̂`, and `ρ(K)` is pressured the wrong way

Surface: `reserve_p12`, `capacity_budget = 63`, `private_reserve = 12`, `n = 32`,
`k_v = 20`, `ρ_K = 2` (band `σ(used) ∈ [0.5, 1.0]`). Seed 42, 2,000 training
ticks, five trials per arm. Horizon stamped in this run, not inherited: `std_max`
**7.12e-04** over the last 100 ticks, `moving_share` 0.167 — below B38's `MOVING`
floor of `1e-2`, so the world is stalled at the read, which is the standing
condition of every reading on this map. It does not block this one: `det.trial`
holds the world still and differences two branches forked from one state, so the
paired deviation is a property of the sheaf's own dynamics.

Instruments: `b44_return.py` (train + read), `b44_analyse.py` (paired),
`b44_normality.py` (the identity behind reading 2). No `--file` was passed and no
max-min-over-paths conduction verdict is reported — B39 struck that quantifier.

---

## 1. The pre-registered falsifier fired: `τ̂` does not move at `channel_return ≈ 1`

The ticket's own words: *if `τ̂` is unchanged at `channel_return ≈ 1`, the return
term is not what sets the decay and §2 of B39's resolution is wrong.*

**A 2×2, because gain and return are two things and B39's claim is about their
ratio.** All five arms are installed on **one** trained surface — one training
run, one `K`, one body, one dome — so `world_loop(c)` is identical everywhere and
only the transport maps differ. `flat_*` is B42's reserved-frame flat bundle
(frame built *inside* the permitted window, which is B42's correction of B40);
`haar_*` is `holonomy_read.flat_maps`, the same block structure and the same
isometry drawn per edge-side instead of off a shared cell frame, so what
separates the pairs is cycle-consistency alone.

| arm | `ident` | `chan` | cycle `σ_max` | bottleneck | `τ̂` med | ratio med | censored | readable |
|---|---|---|---|---|---|---|---|---|
| `control` | 0.9681 | 0.3395 | 2.285e-06 | 3.395e-11 | **8.0** | 0.8750 | 16 | 10 |
| `flat_unit` | **0.0000** | **1.0000** | 1.000e+00 | 1.366e-08 | **6.0** | 0.6667 | 9 | 17 |
| `haar_unit` | 1.0088 | 0.1903 | 1.102e-07 | 9.764e-14 | **8.0** | 1.0769 | 18 | 8 |
| `flat_banded` | **0.0000** | **1.0000** | 7.716e-04 | 2.449e-11 | **7.0** | 0.8750 | 16 | 12 |
| `haar_banded` | 1.0088 | 0.1903 | 1.102e-07 | 9.760e-14 | **9.0** | 1.0000 | 18 | 8 |

The manipulation landed: `channel_return` runs 0.19 → 1.00 and the cycle's own
`σ_max` runs 1.10e-07 → 1.000, seven orders. ADR-0021's bottleneck runs
**9.76e-14 → 1.37e-08**, five orders, which is the first time on this map's
record the return has been anywhere but B38's negligible regime.

**And `τ̂` sat still through all of it.** Paired per cell, per trial — the arms ran
the same five sources with the same rotation, so cell `i` of trial `j` is the
same cell under the same perturbation in every arm:

| arm | Δ`τ̂` median | Δ`τ̂` mean | IQR | longer | shorter | same | `τ̂` ratio med |
|---|---|---|---|---|---|---|---|
| `flat_unit` | **−1.0** | −2.525 | [−8, +3] | 297 | 407 | 51 | 0.764 |
| `haar_unit` | **0.0** | +0.944 | [−5, +7] | 372 | 346 | 37 | 1.000 |
| `flat_banded` | **0.0** | −0.910 | [−6, +5] | 349 | 361 | 45 | 1.000 |
| `haar_banded` | **0.0** | +0.923 | [−6, +7] | 364 | 353 | 38 | 1.000 |

**The null these have to beat is control's own trial-to-trial spread**, paired the
same way: `|Δτ̂|` median **5.0**, q75 **12.0**, q90 **21.0** over 1,510 paired
samples. Every between-arm shift is well inside it.

**The same read on #224's gate**, restricted to cells readable in *both* arms of a
pair — 85–95% of cells read as unreadable at runtime precision on this map, so a
median over the whole population is largely a median over arithmetic noise:

| arm | cells readable in both | Δ`τ̂` median | Δ`τ̂` mean | longer | shorter | ratio med |
|---|---|---|---|---|---|---|
| `flat_unit` | 45 | **−1.0** | −5.556 | 7 | 25 | 0.714 |
| `haar_unit` | 38 | **0.0** | +1.158 | 15 | 16 | 1.000 |
| `flat_banded` | 43 | **0.0** | −4.070 | 15 | 21 | 1.000 |
| `haar_banded` | 38 | **0.0** | +1.158 | 15 | 16 | 1.000 |

Same verdict on the gate as off it.

**The 2×2 discriminates nothing.** A `τ̂` that tracked the return would move down
the columns (`flat` vs `haar`); one that tracked gain would move across the rows
(`unit` vs `banded`). `flat_unit` −1 and `flat_banded` 0 against `haar` +0.9 at
both gains: no systematic movement in either direction, and nothing above the
noise floor.

**What weak tendency there is runs the wrong way, and the censoring bias runs
against it.** `flat_unit`'s mean is −2.5 (−5.6 on the gate) with 25 of 32 gated
cells *shorter*. A return that sustained the deviation would lengthen `τ̂`, not
shorten it; a plausible reading of the sign is dispersal — unit-gain transport
spreads the perturbation off the cell faster, so the local peak falls sooner.
And a censored `τ̂` is a **lower bound**, so an arm with more censored cells has
its `τ̂` under-reported: `control` censors 16 against `flat_unit`'s 9, so the bias
inflates the gap in the direction of "flat is shorter" rather than manufacturing
the null. The finding survives it.

**Verdict: the falsifying end fired. The return term is not what sets the decay,
and §2 of B39's resolution is wrong** — not in its arithmetic, which stands (a
uniform rescale does cancel exactly against the deviation's own peak), but in the
conditional it opened: the non-uniform part, the ratio of local decay to what
comes back around the world loop, was argued to be what `τ̂` can see, and at O(1)
return it moves `τ̂` by less than one trial's noise.

**Incidental, recorded so it is not re-derived**: `haar_unit` and `haar_banded`
read identically on the cycle surface (`ident` 1.0088, `chan` 0.1903, `σ_max`
1.102e-07) and differ only in the sixth digit of the bottleneck. `project()`'s
band is a no-op on maps that are already isometries, so the two are one arm
wearing two names.

---

## 2. `ρ(K)` moves — and the retention it buys moves the other way

The ticket's expected shape was *flat, therefore the retention lever is present
and unpressured, which is what would license an objective term aimed at it.* The
reading is sharper than that and changes the licensing.

| tick | `ρ(K)` med | `ρ(K)` max | `ρ(used)` med | `σ(used)` med | `τ` from `ρ(used)` |
|---|---|---|---|---|---|
| 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ∞ |
| 1 | 1.000000 | 1.000203 | 0.997101 | 1.000000 | 344.5 |
| 5 | 1.059871 | 1.300968 | 0.991733 | 1.000000 | 120.5 |
| 10 | 1.098375 | 1.327020 | 0.987981 | 1.000000 | 82.7 |
| 25 | 1.113148 | 1.329993 | 0.984833 | 1.000000 | 65.5 |
| 50 | 1.121865 | 1.337954 | 0.985600 | 1.000000 | 69.0 |
| 100 | 1.124386 | 1.355907 | 0.985750 | 1.000000 | 69.7 |
| 150 | 1.126770 | 1.361739 | 0.984734 | 1.000000 | 65.0 |
| 250 | 1.130951 | 1.364284 | 0.984199 | 1.000000 | 62.8 |
| 500 | 1.140224 | 1.366770 | 0.983021 | 1.000000 | 58.4 |
| 1000 | 1.139870 | 1.780192 | 0.981658 | 1.000000 | 54.0 |
| 2000 | 1.149326 | 1.651907 | **0.981259** | 1.000000 | **52.9** |

**The parameter is not flat.** `K = a·I` at construction gives `ρ(K) = 1.000000`
uniformly across all 150 cells; by 2,000 ticks **149 of 150** cells have moved by
more than `1e-3`, median `|Δ| = 0.149`, max `0.652`. One-step prediction error
descends on `K`'s spectrum every tick, exactly as `PredictionRule.step` says it
does. The lever is present, and it is being pressed.

**But `σ(used)` is pinned at exactly 1.000 at every checkpoint** — the band's
upper face — on all 150 cells at 2,000 ticks. So none of `K`'s growth reaches the
operator the cell computes with; the radial rescale in `CellOperators.used`
absorbs all of it.

**And the quantity ADR-0026 reads falls monotonically.** Above the upper face the
rescale gives `ρ(used) = ρ(K)/σ(K)` exactly — verified per cell to a worst
relative error of **1.66e-06** (`b44_normality.py`; `ρ(K)` 1.149326, `σ(K)`
1.192176, ratio 0.981259, `ρ(used)` 0.981259). So the retention constant is not a
fact about how large `K` got. **It is `K`'s departure from normality**, and that
is what the objective is actually moving: 1.000 at construction → 0.981 at 2,000
ticks, monotone after tick 25.

In ADR-0026's own currency that is `τ`: **∞ → 344 → 65 → 52.9**. ADR-0026's
recorded `λ = 0.99 → τ ≈ 99.5` is passed on the way *down*, somewhere between
ticks 5 and 25, and never returns.

This is exactly what ADR-0015 wrote down in advance. `CellOperators.norms`: *"for
a non-normal matrix `ρ = 0.5` is compatible with `σ_max = 50`, and a dense `K`
trained on a temporal objective will find exactly that, because transient growth
is **how** linear systems move content."* It found it. Prediction error buys its
one-step accuracy with transient growth, the band charges that growth to `σ`, and
retention is what pays.

**Verdict: the retention lever is present and it is not unpressured — it is under
a standing pressure that runs away from retention.** The ticket's licensing
argument therefore does not apply as written: an objective term aimed at
retention would not be filling a vacuum, it would be an adversary of the
prediction term, and #532's standing question about whether two terms are
adversaries (B33's) reaches it. Whether such a term is adopted is not this
ticket's decision.

---

## 3. The conduction ratio, without its struck quantifier

Per cell, `τ̂_c / world_loop(c)`, over ADR-0026's outbound population. B39 struck
*"over rim-to-apex paths"*, the per-stratum count and the L1 universal; none of
them appear here, and `Trial.conduction` is carried in the JSON for provenance
only. `world_loop(c)` is identical in every arm — one dome, no relay — so a
difference between arms is a difference in `τ̂` and nothing else.

Median ratio runs **0.667 to 1.077** across the five arms; cells at or above 1 run
**41 to 83** of ~150 per trial. On #224's gate the medians are 0.71 to 1.00. The
distribution does not separate the arms any more than `τ̂` does, which is the same
finding read in the bar's own currency.

---

## What this does not establish

- **One seed, one arm, one training run.** `τ̂`'s insensitivity to the return is
  measured on `reserve_p12` seed 42 at 2,000 ticks. B38 found the stall horizon
  varies 13× between seeds of one arm; nothing here says how much of the noise
  floor is seed.
- **The world is stalled at the read** (`std_max` 7.12e-04). Detectability holds
  the world by construction so this does not invalidate the reading, but the arm
  never traverses during it, and whether a longer-lived world would let training
  move `τ̂` is out of scope on #532 (*Do not reopen the sandbox*).
- **#224's gate is the background condition.** 8–17 of ~150 cells are readable per
  trial. The gated read agrees with the ungated one, which is the most that can be
  said without pricing the gate — still unpriced, and still in this map's fog.
- **The flat bundle is a null, not a proposal.** B42 refused it as architecture
  and retained it as this map's null; that is the only role it plays here.
- **Nothing is said about whether an objective term aimed at retention is
  adopted.** *Plan, don't do* stands.
