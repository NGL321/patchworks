# Does construction alone predict the per-hop grading?

[#233](https://github.com/NGL321/patchworks/issues/233). One measurement, no decision in it. It
explains [#214](https://github.com/NGL321/patchworks/issues/214)'s per-hop attenuation and rules on
nothing: the remedy species this grading belongs to is already closed
([#184](https://github.com/NGL321/patchworks/issues/184),
[#230](https://github.com/NGL321/patchworks/issues/230)). Its value is that it tells the retention
recharter whether per-edge variation is a construction fact or a learned one. Reproduced by

    docker run --rm --entrypoint python patchworks:headless \
        benchmarks/construction_grading.py predict     # 1 s wall, no sandbox
    docker run --rm --entrypoint python patchworks:headless \
        benchmarks/construction_grading.py regress     # ~25 min, trains 30k ticks

in the supported container ([ADR-0012](../adr/0012-a-container-is-the-supported-execution-target.md)),
on the real dome: 414 cells, 682 edges, `chi = 1036`, seed 0, 24 trials, both directions — #214's own
configuration. The predictor is checked against a Monte Carlo of the operator it claims to predict in
`tests/test_construction_grading.py`, because the whole ticket is arithmetic and a plausible wrong
number would not be caught downstream.

---

## The verdict, in four lines

1. **Construction does not predict the grading, and it does not have the range to.** Regressed in the
   log over 296 hops: **`R² = 0.139`**, residual **16.5x**, and a fitted slope of **+3.17** where a
   working predictor gives +1. The predictor's *entire* dynamic range over all 7,122 directed hops
   the graph admits is **10.1x** — and **5.0x** across the edges a rim-to-apex path actually uses —
   against a measured transport spread of **151x** from p05 to p95. The slope is that gap: the fit
   has to stretch the prediction threefold in the log to span what was measured.

2. **A third of what #214 reported as attenuation is not transport at all.** The reported per-hop
   figure is a ratio of ratios, `r_i / r_{i+1} = (dev_i/dev_{i+1}) · (floor_{i+1}/floor_i)`, and the
   floor ratio carries **0.47 dex of the 1.37 dex** the reported grading spans — **34% by standard
   deviation**. The floor is where the trained sheaf happens to be standing on two different edges.
   No construction quantity predicts it, and none should be asked to.

3. **Inter-endpoint misalignment is bounded out as the explanation, on the favourable side.** The
   alignment headroom — what a hop is worth to the best-chosen direction against an average one — is
   **at most 2.83x on every hop measured**, median 1.97x. A residual of 16.5x cannot be a direction
   effect that has 2.8x of room. The *unfavourable* side is not read here, and that is what the fog
   patch graduates on.

4. **The direct route through a relay is far weaker than an independent-maps model says.** The exact
   composed operator `F_out · gain_v · F_inᵀ` is **0.0079x** — 127x weaker — than the factorised
   prediction built from the same two maps' norms. A relay's two incident maps do not share their
   dominant directions, so what arrives on one edge barely projects onto what leaves on another.

The second is the one that changes how #214's own numbers should be read, and it was not visible in
that read's output.

---

## 1. What a hop is, and what the predictor is

`Sheaf.message_passing_phase` writes `x_v ← x_v − gain_v · Σ_e F_evᵀ (F_ev x_v − y_e)`, and
`broadcast` is read from the *pre-update* stalk. So a displacement arriving on edge `e_in` at tick `t`
reaches `e_out`'s disagreement at tick `t + 1`, through the body's inference phase on the way, and one
hop through cell `v` is the linear operator

    M(e_in → v → e_out)  =  F_{v,e_out} · J_body(v) · gain_v · F_{v,e_in}ᵀ

Only `v`'s own two maps appear: the disagreement on `e_in` is absorbed through `F_{v,e_in}ᵀ` and
re-emitted through `F_{v,e_out}`, and the far endpoint's map does not enter. For an isotropic arriving
direction the per-direction gain is `‖M‖_F / sqrt(m_in)` — the Frobenius identity
`benchmarks/graph_transmission.py` checks by Monte Carlo, applied to the composition rather than to
one factor.

At a predicting cell `Dome.restriction_mask` reads one `_permitted[cell]` **regardless of the edge**,
so the incoming and outgoing permitted subspaces coincide and nothing is lost to a mask mismatch.
Modelling each map as its Frobenius norm spread over its permitted entries gives the construction-only
predictor

    P(e_in → v → e_out)  =  body · gain_v · ‖F_in‖_F · ‖F_out‖_F / sqrt(m_in · perm_v)

with `‖F‖_F = rho = 2` at a predicting cell, `gain_v = gamma / max(Σ_e m_e, rho² deg(v))`, and
`perm_v = min(n, Σ_e m_e)`. Every term is read off the built graph. `body` is #120's constant, so it
sets the scale and contributes **nothing to the grading**, which is what was asked.

Three tiers are run, because they separate three causes: **construction** as above; **realised norms**,
the same formula with the trained maps' actual Frobenius norms; and the **exact operator**,
`‖F_out gain_v F_inᵀ‖_F / sqrt(m_in)` from the trained maps themselves.

---

## 2. The predictor's range, before anything is measured

If construction is to explain a 9x-to-240x grading it must first *have* one. Over every directed hop
the graph admits:

| | min | p05 | median | p95 | max |
|---|---|---|---|---|---|
| attenuation `1/P` | 45.5x | 129.8x | 299.8x | 459.3x | 459.3x |

**Range 10.09x**, and the 45.5x floor is the eight `m = 1` drive edges, which no rim-to-apex path
uses. Excluding them the range is **5.0x**, from 91.0x to 459.3x.

Only three quantities vary at all, and one of them dominates:

| quantity | distinct values | min | max | range |
|---|---|---|---|---|
| `gain_v` | 9 | 0.01923 | 0.05 | 2.60x |
| `perm_v` | 5 | 17 | 32 | 1.88x |
| `m_in` | 3 | 1 | 8 | 8.00x |

`m_in` carries most of it, and it takes three values. **A quantity with three values cannot grade a
seven-hop path across two orders of magnitude.** That is the verdict before a single tick is run, and
the regression below is the confirmation rather than the finding.

| relay and incoming width | hops | median | min | max |
|---|---|---|---|---|
| predicting, `m_in = 1` | 32 | 45.5x | 45.5x | 45.5x |
| predicting, `m_in = 4` | 5110 | 224.8x | 91.0x | 324.7x |
| predicting, `m_in = 8` | 1980 | 459.3x | 282.6x | 459.3x |

---

## 3. The read corroborates #214 rather than reproducing it bit for bit

Same seed, same trial count, same window and hold, same container. The rim-to-apex **binding edge is
`#450 interior m=4 L6/core(5,) — L7/core(5,)`, which is #214's own**, and both medians land within a
factor of two:

| | this read | #214 | |
|---|---|---|---|
| rim→apex median bottleneck | 4.69e-10 | 8.7e-10 | 1.9x |
| apex→rim median bottleneck | 1.14e-08 | 1.3e-08 | 1.1x |

The per-hop profile along the median path is **39x, 23x, 184x, 23x, 11x, 71x** against #214's *240x,
57x, 16x, 25x, 84x, 9x, 29x* — the same strongly-graded, non-interchangeable shape, a different
draw of it. The trial that lands at the median is not the same trial, so these are two samples of one
population and not two runs of one measurement. **Nothing below depends on which sample it is**: the
regression is over all 296 hops of all 24 trials in both directions, and the range argument in §2
needs no run at all.

---

## 4. The regression

Log-log, over 296 hops. `R²` is the fitted line's; **unfitted bias** is `mean(log measured − log
predicted)`, what is left when the prediction is simply believed with no slope or offset allowed.

| tier | r | `R²` | slope | residual sd | unfitted bias |
|---|---|---|---|---|---|
| construction only | +0.373 | **0.139** | +3.172 | 1.217 dex (**16.5x**) | +0.45 dex |
| + realised norms | +0.450 | 0.203 | +2.715 | 1.171 dex (14.8x) | +0.51 dex |
| + exact operator | +0.158 | 0.025 | +0.182 | 1.295 dex (19.7x) | +2.24 dex |
| construction vs the *reported* ratio | +0.276 | 0.076 | +2.449 | 1.317 dex (20.8x) | +0.52 dex |

Knowing the trained maps' realised norms buys `R²` from 0.139 to 0.203 and nothing more. **The
grading is not in the norms, and it is not in the gauge.**

The last row is worth its place: regressed against the quantity #214 actually printed — the ratio of
ratios — construction does *worse* (`R² = 0.076`), because a third of that quantity is floor variation
which construction has no term for at all.

### The residual is graded by level, which construction is not

| relay | hops | mean residual | sd |
|---|---|---|---|
| L1 | 48 | −0.49 dex | 2.70 |
| L2 | 55 | +0.52 dex | 0.41 |
| L3 | 49 | +0.68 dex | 0.63 |
| L4 | 48 | +0.69 dex | 0.54 |
| L5 | 48 | +0.72 dex | 0.47 |
| L6 | 48 | +0.59 dex | 0.55 |

Reported for shape and never as an index — the map's standing rule. L1 is the outlier in both mean and
spread, and it is where the five dead hops of §6 live.

---

## 5. What #214's reported attenuation is actually made of

| quantity | median | p05 | p95 | spread |
|---|---|---|---|---|
| reported `1/r` ratio | 37.9x | 2.63x | 349.4x | 132.7x |
| **transport** `1/hop` | 39.8x | 1.76x | 266.1x | **151.0x** |
| **floor ratio** | 1.1x | 0.28x | 11.2x | **40.0x** |

The split is an identity, not a model, and closes to `3.3e-16`. The floor ratio carries **0.47 dex of
the 1.37 dex** the reported grading spans — **34% by sd**.

This is the finding with the longest reach. #214's per-edge numbers are bottleneck *ratios*, and a
ratio of two of them is a hop **times** a statement about where the sheaf was standing. Anyone reading
the 9x-to-240x sequence as seven transport measurements is reading a third of something else. The
predicate is unaffected — ADR-0021 asks for the ratio and the ratio is what fails — but a *causal*
account of the grading has to split them first, and until this read nothing had.

---

## 6. Five hops are not graded at all

**5 of 296 hops attenuate by more than 1e6x.** Four of them are one relay, `L1/somatomotor(1,)`,
at ~3.1e11x — nine orders past its 282.6x prediction:

| relay | `m_in` | `m_out` | measured | tier 1 | residual |
|---|---|---|---|---|---|
| L1/somatomotor(1,) | 8 | 4 | 3.10e11 x | 282.6x | −9.04 dex |
| L1/somatomotor(1,) | 8 | 4 | 3.10e11 x | 282.6x | −9.04 dex |
| L1/somatomotor(1,) | 8 | 4 | 3.10e11 x | 282.6x | −9.04 dex |
| L1/somatomotor(1,) | 8 | 4 | 3.09e11 x | 282.6x | −9.04 dex |
| L1/vision(6, 0) | 8 | 4 | 1.41e6 x | 423.9x | −3.52 dex |

A hop three orders past the graded band is not a graded hop, and averaging it in with the rest would
be the graph-wide-average mistake at hop scale. They are counted rather than trimmed because the count
is itself a finding: **the binding path sometimes runs through a relay that is effectively
disconnected**, and the max-min path takes it because every alternative is worse.

---

## 7. The two gaps between the tiers

| gap | median | p05 | p95 |
|---|---|---|---|
| **composition** — exact / independent-maps | 0.0079x | 0.00135x | 4.9x |
| **direction** — measured / exact | 480x | 0.477x | 1.75e4 x |

**Composition, 127x of loss, is real and is not in the predictor.** The factorised formula treats a
relay's two incident maps as independent draws. They are not: composing them destroys two orders of
magnitude more than independence predicts, which says the maps' dominant directions are close to
orthogonal to each other. That is the object
`RestrictionMaps.project` does not yet hold apart — `GAUGE_C`, and `reconciliation_gain`'s own
docstring says the ruled denominator waits on exactly that projection. It is a *construction-adjacent*
term, in that it is a property of the built maps, but it is not predicted by any construction
parameter and it varies per relay.

**Direction, 480x, is an instrument-scale artifact and was checked rather than assumed.** A one-tick
operator cannot be the whole of a peak-to-peak ratio taken over a 64-tick window, so the operator
prediction is a *floor* on the measured quantity. The alternative — that the graph's parallel routes
carry it — predicts the gap grows with how many routes a relay has. It does not:

| relay degree | hops | median gap |
|---|---|---|
| 5 | 8 | 467.9x |
| 6 | 207 | 459.2x |
| 7 | 35 | 747.6x |
| 8 | 23 | 636.4x |
| 9 | 23 | 408.4x |

`r = −0.121` against degree over 296 hops, degrees 5 to 9. **Flat.** The gap is the window's
accumulation, not the graph's routes, and it is a near-uniform level shift rather than a source of
grading — which is why it moves the tier-3 bias to +2.24 dex while leaving `R²` at 0.025.

---

## 8. What this does to #184's parked fog patch

The patch named three candidate causes of the grading and said it was fog *because nobody knew which
of the three it was*. All three are now read:

- **The taper's own structure (construction).** Measured, and it explains `R² = 0.139` with 5.0x of
  range against 151x of spread. **Not the cause.**
- **Per-edge floor variation.** Measured at **34%** of the reported grading's spread. **A real and
  substantial contributor, and now quantified rather than suspected.**
- **Inter-endpoint misalignment.** Bounded on the favourable side: the alignment headroom is **at most
  2.83x** on every hop measured. A 16.5x residual cannot be an effect with 2.8x of room *in that
  direction*.

What is **not** read here is the unfavourable side. Alignment headroom prices what the best direction
buys over an average one; it does not price what a *badly*-chosen direction costs, which is unbounded
below and is the form the candidate would actually take. This read cannot settle it, because the
measured quantity is a windowed peak sitting 480x above the isotropic single-step prediction, and a
level shift that large swamps the sign of a per-hop direction effect.

So the patch graduates with a sharpened question rather than a guess, which is the outcome #233 was
opened to produce: **does what arrives on an edge align worse than average with the next hop's
operator, read tick-aligned and single-step rather than peak-to-peak?** The instrument this ticket
leaves behind makes that read cheap — the operator, the arriving deviation and the pairing are all
already built.

---

## What this ticket does not claim

- **It rules on nothing.** The remedy species is closed and this reopens none of it.
- **`body` is #120's constant, not a per-cell measurement.** It cancels out of every grading statement
  and is carried only to put the tiers on the same scale as that reading. A per-cell body Jacobian
  could contribute grading this read would attribute to the residual.
- **The widest path is a max-min path, not a flow.** It names the edge the predicate binds on, which
  is what ADR-0021 asks of it, and it is not a claim that transport travels only along it.
- **296 hops from 24 trials in two directions are not 296 independent samples.** Trials share a
  trained surface and paths overlap; the regression's `R²` is a description of this population, and
  the §2 range argument — which needs no run — is what the verdict rests on.
