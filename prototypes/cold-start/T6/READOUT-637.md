# B63 (#637): the sensorimotor dependence instrument, and its first reading

**Surface** (#455): `reserve_p12` — `capacity_budget = 63`, `private_reserve = 12`,
which is `b33.ARM` — seed 42, 2,000 ticks, `float32` training cast to `float64`
for the read. Three arms on one surface: `untrained` (construction), `trained`
(the shipped rule), `flat` (B42's retained null, installed as B44 installs it).

**`traditional`, declared** (B43 §8). Every information-theoretic quantity below
is ordinary Shannon on random variables. Nothing here is cohomological.

**Which object** (B49, #616). Sections 1–6 are on **node stalks** — the
actuator's `commanded` block. Section 7 is on the **lane**. No number crosses.

**Horizon, stamped per run** (B38, #599): `world_std_max` **7.09e-04** over the
last 100 of 2,000 ticks, `moving_share` 0.167. The body has stalled. It does not
invalidate the read — `detectability` holds the world through every branch — but
it bounds what may be said about the *trained* surface, exactly as B38 requires.

**`--no-file` throughout**, and it stays: see §8.

---

## The reading

`I(P; Δ)` lower bound in bits, peak feature, `k = 8`, leave-one-out
nearest-centroid, against a matched scramble null of 400 within-configuration
permutations. Ceiling `log₂ 8 = 3.000`.

| arm | stratum | acc | MI | null | excess | p |
|---|---|---|---|---|---|---|
| untrained | patch | 0.089 | 0.276 | 0.194 | 0.082 | 0.070 |
| untrained | proprioceptive | 0.083 | 0.257 | 0.146 | 0.111 | 0.057 |
| untrained | touch | 0.073 | 0.201 | 0.184 | 0.017 | 0.304 |
| **trained** | patch | 0.375 | **0.857** | 0.235 | 0.622 | 0.0025 |
| **trained** | proprioceptive | 1.000 | **3.000** | 0.196 | 2.804 | 0.0025 |
| **trained** | touch | 1.000 | **3.000** | 0.172 | 2.828 | 0.0025 |
| flat | patch | 0.594 | **1.645** | 0.220 | 1.425 | 0.0025 |
| flat | proprioceptive | 1.000 | **3.000** | 0.170 | 2.830 | 0.0025 |
| flat | touch | 0.755 | **2.414** | 0.184 | 2.230 | 0.0025 |

`p = 0.0025` is the floor at 400 permutations, i.e. no null draw reached the
observed value.

**Three things at once.**

1. **The untrained surface does not clear the bar.** No stratum reaches
   significance. B58 (#630)'s warning — *a construction may supply the quantity
   free* — is discharged **in the architecture's favour** for this quantity, and
   it is the first thing on this map that has been.
2. **The trained surface clears it on all three strata.**
3. **So does the flat bundle** — this map's own retained null — and on patch it
   clears it by nearly **2×** the trained surface. `I(P; Δ) > 0` **does not
   discriminate the architecture from a construction refused as architecture.**

B43 §5 named the honest cost as *above zero, above null can be cleared by a
whisker*. On the widest stratum it is not a whisker: the null wins. This is not
the reference failing — it is the reference doing exactly the job B43 gave it,
on the first reading.

**Proprioceptive and touch saturate.** Both read the ceiling exactly on trained,
and proprioceptive on flat too. `log k = 3` bits is not headroom on the narrow
strata; a reading that means to separate arms there needs a larger `k`.

---

## 1. The estimator, and why its bias runs the safe way

With `P` uniform and **chosen**, `H(P) = log k` exactly, so `I(P; Δ) = log k −
H(P|Δ)` is a **recovery** problem, not a density problem. Decode `P` from `Δ`,
form the `k × k` confusion table, take its plug-in MI. Because `P̂ = f(Δ)`, the
data-processing inequality gives

> `I(P; P̂) ≤ I(P; Δ)`

so **every number reported is a lower bound**, and an estimator that degrades
with dimension loses decoder accuracy and *lowers* the estimate. That inverts
the failure mode the ticket named: a plug-in density estimate on a wide
continuous stalk has **positive** bias and reads bias as signal; a bounded
recovery estimate cannot.

The decoder is leave-one-out (within-sample would make the table diagonal by
construction). Two are run and reported separately, never maximised over —
nearest class centroid and 1-NN, both cosine, because `‖Δ‖` varies with the
situation and a metric that read it would let the estimator recover `C` and call
it content.

**The residual plug-in bias is not corrected by a formula.** The matched
scramble null runs the *identical* pipeline — same decoder, same folds, same
table — on labels permuted **within each configuration**, which keeps the design
balanced and destroys only the correspondence. Whatever bias the pipeline carries
is in the null too, and the reported quantity is the excess. Miller–Madow is
recorded as a cross-check and is never the headline.

## 2. The alphabet

A pattern is a unit vector over the stratum's whole **product space**, `dim =
cells × stalk` — B43 §7's *two different things happening in two places in the
visual field* is a **spatial** distinction, not a direction in one cell's stalk.
`k = 8`, orthonormal where the stratum can carry it, maximally-spread where it
cannot. The labels are nested, so `k' = 2, 4, 8` comes free.

| stratum | dim | shape | orthonormal at k=8 | max &#124;cos&#124; |
|---|---|---|---|---|
| patch | 12288 | 256 cells × stalk 48 | yes | 0.0000 |
| proprioceptive | 6 | 3 × 2 | **no** | 0.7615 |
| touch | 3 | 3 × 1 | **no** | 0.9881 |

Touch cannot carry eight distinguishable patterns at all, and the reading quotes
its crowding rather than a ceiling it cannot reach.

## 3. The sweep is the noise model, and it saturates exactly as B43 §3 predicted

24 situations, one `env.reset(seed)` each, every pattern of every stratum
injected off **one shared quiet fork** — so the design is balanced by
construction, one trial per `(C, stratum, pattern)`. 576 trials per arm.

The configuration ladder is the ask, answered:

| arm | stratum | C=4 | C=8 | C=16 | C=24 |
|---|---|---|---|---|---|
| untrained | patch | 0.871 | 0.529 | 0.203 | 0.276 |
| untrained | proprioceptive | 1.051 | 0.553 | 0.016 | 0.257 |
| untrained | touch | 1.296 | 0.326 | 0.258 | 0.201 |
| trained | patch | 1.468 | 0.898 | 0.858 | 0.857 |
| flat | patch | 1.313 | 1.750 | 1.574 | 1.645 |

At `C = 4` the **untrained** arm reads 0.87–1.30 bits — a surface that reads
0.20–0.28 at `C = 24`. That is the vacuous conditioned form leaking back in, and
it is large enough to have carried a false pass on its own. **`C ≥ 16` is where
it settles; `C ≤ 8` is not admissible.** 24 is reported.

**The conditioned form is verified vacuous, not estimated.** B43 §3 argues
`I(P; Δ | C) = log k` by construction from determinism. Estimating a quantity
that is `log k` by construction would be theatre, so the *premise* was checked
instead: a re-run branch's `Δ` is **bit-identical**, `max|d| = 0.0`. Confirmed
on the instrument.

## 4. #224's gate, and the sharpest thing in this reading

| arm / stratum | median &#124;Δ&#124; | / &#124;&#124;state&#124;&#124; | clears eps_f32 | clears eps_f64 |
|---|---|---|---|---|
| untrained / patch | 2.056e-11 | 6.98e-14 | **0/192** | 192/192 |
| trained / patch | 3.253e-13 | **2.74e-15** | **0/192** | 192/192 |
| flat / patch | 4.820e-09 | 4.37e-11 | **0/192** | 192/192 |
| trained / proprioceptive | 9.728e-04 | 8.20e-06 | 192/192 | 192/192 |
| trained / touch | 2.049e-04 | 1.73e-06 | 192/192 | 192/192 |

**The patch stratum's response at the world-read boundary never clears float32,
on any arm.** Its entire dependence reading exists only on the float64 cast.

And **training makes it smaller**: 2.74e-15 against construction's 6.98e-14 —
**26× down** — and against the flat bundle's 4.37e-11, **16,000× down**. Over the
same interval the MI goes **up**, 0.276 → 0.857.

That is B43 §1's argument for an information quantity doing exactly what it
promised — invariance under any invertible reparameterisation of either side, so
a wandering scale is absorbed. It is also the reason to be careful with it:
**the measure is invariant to an amplitude collapse the shipped runtime cannot
represent.** A pass on patch is a statement about arithmetic that, at `float32`,
is not there.

**Presence/absence collapses into this gate**, as B43 §5's floor case was always
going to: under the pairing the unperturbed arm's `Δ` is *exactly* zero, so
presence is decodable at any amplitude and `I = log 2` trivially. The gate table
is what is reported in its place.

## 5. The per-trial reduction is what costs the reading

| trained arm | acc | MI | excess |
|---|---|---|---|
| patch, **peak** (ADR-0021's reduction) | 0.375 | 0.857 | 0.622 |
| patch, **peak**, 1-NN | 0.594 | 1.293 | 1.066 |
| patch, **trace** (the whole window) | 0.958 | **2.810** | 2.582 |

B43 says `Δ` is *"reduced per trial exactly as ADR-0021 already prescribes"*.
ADR-0021's reduction is the peak, and on patch the peak throws away **three
quarters** of what survives to the terminus. The window is not decoration: it is
where most of the content is. **This amends B43 by comment** (#532's standing
practice), and it does not touch the measure — only what `Δ` is reduced to.

## 6. The terminus is three components wide

The actuator's node stalk is **6** and its readable write-complement — the
`commanded` block, `reading_sites`' own rule — is **3**. So `Δ` is a
3-dimensional object, 2-dimensional after the direction normalisation §1
requires.

That is why the profile runs the way it does: **patch, the widest stratum by
four orders, reads lowest of the three, and touch — 3 dims, near-bijective to a
3-wide terminus — saturates.** As specified, the per-stratum profile is closer to
a readout of *stratum dimension against terminus width* than of transport
quality.

**The clamp B43 §3 asks every measure to name is this one**: `spec.joints = 3`,
fixed at construction. It joins the mask (B22, #571), the band (B44, #610) and
`k_v` (B56, #628) — the parameter moves and the quantity the architecture reads
does not, and here the clamp sits on the **bar** rather than on a trained object.

## 7. The per-edge profile — a LANE object, and it runs

Ask 5 was a budget call. It runs: **no extra trials** (the same runs, one
subtraction and one argmax), and **0.1 min per stratum** to estimate, because the
balanced design makes the leave-one-out centroid a closed form that vectorises
across all 682 edges at once. What would have been `edges × permutations`
separate decodes is one pass over `[E, N, m]` per permutation.

| stratum | edges | median | null | excess | max | p ≤ 0.05 |
|---|---|---|---|---|---|---|
| patch | 682 | 2.696 | 0.220 | 2.488 | 3.000 | 655/682 |
| proprioceptive | 682 | 1.043 | 0.167 | 0.970 | 3.000 | 621/682 |
| touch | 682 | 1.168 | 0.173 | 0.976 | 3.000 | 624/682 |

**These are lane numbers and nothing above is comparable with them** (B49, #616).
The feature is the lane deviation **vector** at each edge's own peak tick,
normalised — not `‖·‖`, which is an amplitude and which B43 §7 bars from carrying
the content. A first pass here stored the norm; it was discarded, not reported.

## 8. The re-take, and why `--no-file` stays

B43 ruled `#341 CLEAR at 0.133; #325 and #329 SHUT at 0.133` **not re-filable
and re-takeable** on this instrument. Re-taken: on this instrument the predicate
is `I(P; Δ) > 0` over a matched null, and the trained surface **passes on all
three strata**.

It is not filed, and `--no-file` stays. Not because the reading is disputed —
because **the flat bundle passes too**. Filing a CLEAR that this map's own
refused null clears by a wider margin would file a verdict that does not
discriminate, which is the failure `#379` cost this rig once already. The bar is
not weak for want of a threshold — B43 refused one on the `k = 1` discipline and
that refusal stands. What is missing is a **reference that separates**, and that
is a ticket, not a constant.

---

## Files

- `b63_dependence.py` — the instrument. `train` / `read --arm {untrained,trained,flat}` / `analyse`.
- `b63_lanes.py` — the per-edge profile. `collect` / `estimate`.
- `b63_table.py` — the cross-sections above, re-printable without re-estimating.
- `637-train-seed42-2000.json` — the surface, with its own motion horizon.
- `637-{untrained,trained,flat}-seed42-2000.json` — 576 trials per arm.
- `637-analysis.json` — every reading, every cross-section, with nulls.
- `637-lanes-trained.json` — the lane profile.

**Registers consulted:** `docs/registers/architecture.md` rows `EPS_F32`
(*"not a knob at all: it is the arithmetic the architecture runs in, and #224
ruled that precision is not a design variable"*), `PRECISION_FLOOR` (*"a
reading, and it moves with the state it is read at"*) and `NORM_FLOOR`. The
first two are what §4 is measured against and they sharpen it: precision is not
a design variable, so the patch stratum's 0/192 is a fact about the architecture
as it runs and not a knob anyone may turn to rescue the reading. No register
entry bears on an information statistic at the world-read boundary; the
amplitude rows this rig carries are ADR-0021's and are demoted rather than read
here. *(The rendered registers lag the tracker and the generator is currently
broken — #550; these three rows were read for content, not for currency.)*
