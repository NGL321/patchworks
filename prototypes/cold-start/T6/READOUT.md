# T6 (#534): has ADR-0015's structured-`K` fallback trigger fired?

**Verdict: NOT FIRED — and, as worded, it can no longer fire.**

Surface: full dome, `main`'s live build (forward normalisation, `interior_m = 3`,
`boundary_m = 4`), band `[0.5, 1.0]`, 100k ticks, seeds 42/43/44, both T3 arms
(baseline `c = 1`, `η_K = 0.01`; winner `ρ = 1` at the drive edges with `c = 0.1`,
`η_K = 0.001`). Aggregation stated on every row. **Nothing was re-run**: every number
comes from T3's committed checkpoints at `prototypes/cold-start/T3/`, which store
`K_raw` and `grad_K` per cell at ten checkpoints.

Reproduce: `python prototypes/cold-start/T6/band_pressure.py`
(full output in `534-band-pressure.md`, per-row records in `534-band-pressure.json`).

## The quantity, named before it was read

ADR-0015 pre-registers the fallback on *"a standing fight between the gradient and
the projection"*. ADR-0008 restates it as *"the projection and the gradient fight
every step"*. `learning.PredictionRule.step` gives the firing form: *"a projection
that binds every step is instead the observable that calls #138's named fallback"*.

**The literal observable no longer exists.** #433 moved enforcement out of a
post-step projection and into a differentiable forward clamp
(`CellOperators.used`); `benchmarks/projection_firing.py` says so in its own header
and *declines to report a firing rate* rather than report a zero. So the trigger was
restated, in the script's docstring, before any number was read:

| name | definition | what it is |
|---|---|---|
| **engagement** | `σ_max(K_raw) > 1` | the clamp is active — what the cell computes with is not what the gradient wrote. The live analogue of *binds every step* |
| **drift** | `σ_max(K_raw) − 1` | how far the unconstrained parameter has been carried out of band; the integral of the push, if there is one |
| **band pressure `Π`** | `−η_K·⟨∇K, u₁v₁ᵀ⟩ / (1 − 1/ρ_K)` | the first-order change in `σ_max(K)` this step's own gradient makes, in band-widths. `⟨·, u₁v₁ᵀ⟩` **is** `dσ_max/dK`. `Π > 0` is the gradient pushing out, against the clamp |
| **radial share** | `⟨∇K, u₁v₁ᵀ⟩² / ‖∇K‖_F²` | how much of the gradient's energy is the norm-inflating direction at all |
| **persistence** | fraction of checkpoints with `Π > 0`, per cell | the *standing* half. A sign that alternates is not a standing fight |

Pre-registered verdict rule: **fired** needs engagement ≈ 1 **and** `Π > 0`
persistently **and** a radial share large enough to be a real part of the gradient.
**Not fired** is engagement ≈ 0, or `Π ≤ 0`. Anything between is **ambiguous**.

## What it reads

### 1. Engagement is maximal, and the drift is monotone

The raw parameter is out of the band's **upper** face on **150 of 150 cells at every
one of ten checkpoints, in both arms, on all three seeds**, from tick 100 to 100k.
The lower face is never engaged. `σ_max(used)` is therefore exactly `1.000` for every
cell, everywhere — the pinned value #477 reported, but pinned by construction.

Median `σ_max(K_raw)`, mean over seeds:

| arm | stratum | 100 | 1000 | 10000 | 20000 | 30000 | 100000 |
|---|---|---|---|---|---|---|---|
| baseline | soma | 1.1563 | 1.2156 | 1.2423 | 1.2533 | 1.2596 | 1.5365 |
| baseline | vision | 1.1430 | 1.1820 | 1.2464 | 1.2808 | 1.3066 | 1.3562 |
| baseline | core | 1.1159 | 1.1358 | 1.1681 | 1.1854 | 1.1955 | 1.2290 |
| baseline | apex | 1.1413 | 1.1530 | 1.2548 | 1.3325 | 1.3664 | 1.5017 |
| winner | soma | 1.0341 | 1.0848 | 1.1118 | 1.1168 | 1.1200 | 1.1335 |
| winner | vision | 1.0338 | 1.0518 | 1.0957 | 1.1175 | 1.1337 | 1.1916 |
| winner | core | 1.0235 | 1.0313 | 1.0458 | 1.0527 | 1.0576 | 1.0701 |
| winner | apex | 1.0461 | 1.0607 | 1.0804 | 1.1001 | 1.1083 | 1.1369 |

On the letter of *binds every step*, this is as satisfied as it can be.

### 2. But the gradient is not pushing. It cannot.

`cos(∇K, K)` is **O(1e-9)** — float noise — on every cell, every checkpoint, both
arms, all seeds (max absolute value over 150 cells: 2.97e-07).

That is not a coincidence, it is #433's algebra. Where the clamp is engaged,
`used(K) = K/σ(K)` is **exactly scale-invariant**: `J(cK) = J(K)`, hence
`⟨∇J, K⟩ ≡ 0`. The prediction objective is *blind to the scale of `K`*. The gradient
has no radial component to fight the band with.

The first-order pressure `Π` confirms it, and shows no standing direction anywhere:

| arm | 100k, all 150 cells | engaged | frac `Π > 0` | median \|`Π`\| | radial share | persistence |
|---|---|---|---|---|---|---|
| baseline | mean over seeds | **1.0000** | **0.5178** | 1.85e-05 | 0.0078 | **0.500** |
| winner | mean over seeds | **1.0000** | **0.5111** | 6.97e-06 | 0.0271 | **0.600** |

`frac Π > 0` is a coin flip. Persistence — the *standing* half — is a coin flip.
The norm-inflating direction carries 0.8 % / 2.7 % of the gradient's energy and does
not hold its sign.

### 3. The drift is fully accounted for as a step-size artefact

A step orthogonal to `K` lengthens `K`: for a scale-invariant objective,
`Δ‖K‖_F² = η_K²‖∇K‖_F²` per step, unopposed and monotone. Comparing the observed
growth in `‖K‖_F²` against that prediction (integrated with `‖∇‖` sampled at the two
endpoints, so the prediction is a lower bound):

| arm | actual / predicted, median over cells |
|---|---|
| baseline | 2.11 – 4.89 |
| winner | 0.68 – 2.64 |

Same order of magnitude in both arms across a 10× change in `η_K`, which is what the
second-order account requires and what a contested equilibrium would not produce.
The drift is the parameterisation inflating under its own step, not the gradient
straining against a wall.

### 4. It does not separate by stratum

Apex against core at 100k, per seed:

| arm | seed | apex `Π` med | core `Π` med | apex frac>0 | core frac>0 | apex `σ_raw` | core `σ_raw` |
|---|---|---|---|---|---|---|---|
| baseline | 42 | −9.01e-07 | −2.81e-06 | 0.375 | 0.481 | 1.4986 | 1.2364 |
| baseline | 43 | −2.47e-06 | +3.12e-06 | 0.500 | 0.538 | 1.4902 | 1.2402 |
| baseline | 44 | +1.07e-05 | +2.54e-06 | 0.750 | 0.558 | 1.5163 | 1.2103 |
| winner | 42 | −5.34e-06 | +6.68e-07 | 0.125 | 0.519 | 1.2000 | 1.0901 |
| winner | 43 | +6.07e-06 | +1.34e-06 | 0.625 | 0.577 | 1.0748 | 1.0643 |
| winner | 44 | +8.85e-06 | +2.24e-06 | 0.625 | 0.577 | 1.1358 | 1.0558 |

Apex `frac Π > 0` swings 0.125 → 0.750 across three seeds in the same arm: the sign
is seed noise, not a stratum effect. Apex *drift* is the largest of the four strata
in both arms (1.50 / 1.14) and arrives late — flat to ~2000 ticks, then climbing —
which tracks the apex's larger gradient norm, i.e. the artefact, not a fight.

### 5. T1's winner moves the artefact and not the fight

The winner (`c = 0.1`, so `η_K` 10× smaller) cuts the apex's out-of-band excess
**3.7×** (0.502 → 0.137) and the whole-dome median excess **2.5×** (0.311 → 0.122), while leaving
`frac Π > 0` unchanged (0.518 → 0.511) and persistence at chance. Exactly the
signature of a step-size artefact: shrink the step, shrink the drift, and the
"fight" — which was never there — is unchanged.

## Verdict

**NOT FIRED.**

- The trigger's *surface form* — a projection that binds every step — is at its
  maximum: 150/150 cells, every checkpoint, both arms, all seeds.
- The trigger's *content* — a fight between the gradient and the projection — reads
  **zero**, and zero by construction: `⟨∇K, K⟩ = 0` to 1e-9 because #433 made the
  objective scale-invariant in `K`. The clamp binds; nothing resists it.
- ADR-0015's falsification premise is *"if cells prove to need transient growth to
  move content within a piece, the band is wrong"*. No cell can prove that through
  this observable on the current surface, because the objective cannot express a
  preference about `σ_max` at all.

**A record item, not resolved here: the trigger is now unfalsifiable as worded.**
Before #433, *binding* and *fighting* were the same event — a post-step rescale fires
only when the gradient carried the operator out. #433 separated them: binding became
unconditional and free, and the fight went to zero. The docstring's operationalisation
in `PredictionRule.step` is stale relative to its own build; ADR-0015's noun (the
*fight*) is the one that survives, and it reads zero. Whether ADR-0015 wants a new
observable is an ADR-level question that ADR-0029 reserves to a human. Flagged for
the ledger (#520) and for #532; **nothing amended and no fallback adopted here.**

## Scope: `K`, not transport

This reads **`K`** — the object ADR-0015's fallback names — and nothing else. It says
nothing about transport `F`, whose band is ADR-0010's Frobenius gauge and ADR-0032's
isometry and whose enforcement is still a post-step projection, a different mechanism.
R-A's `σ_min/σ_max = 0.9999995` on 1339 of 1364 maps is a reading of *that* band, so
it is not evidence about this trigger; #534's own candidate-evidence note names both
in one breath and they are separate objects. R-A's top-ranked candidate
(channel-heterogeneous transport) restructures `F`, which ADR-0015's fallback does not
authorise. **The cheap route this ticket went looking for is not there.**

Also not evidence: #477's `σ_max` "pinned at 0.9947". Since #433 the used operator's
norm is pinned to the band by construction on every cell; a pinned `σ_max(used)` is
the clamp's output, not a report of pressure on it.
