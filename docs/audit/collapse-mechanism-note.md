# The constant-evidence collapse: a mechanism, a toy, and what would confirm it on the real cells

*External audit, 2026-09-05. Companion to
[`2026-09-05-phase-2-pivot-audit.md`](./2026-09-05-phase-2-pivot-audit.md). Confidence tags:
**[measured]** read off the committed record or this session's runs; **[derived]** follows from the
code's own arithmetic; **[toy]** shown in the twelve-dimensional toy only; **[inferred]** an
argument the record supports but nobody has read on the graph.*

## The claim

[#477](https://github.com/NGL321/patchworks/issues/477) found that by 100k ticks a tenth of the
predicting cells hold an operator that retains nothing — `ρ(K)` down to 0.058, stable rank 1.16 to
1.79, non-normality at the rank-one ceiling of `√2` — and that they are exactly the cells whose
world-side evidence is constant: the eight apex cells against a fixed drive and the somatomotor
column against a frozen arm, none of the eighty vision cells. It recorded *constancy* as the leading
candidate, not established, and pre-registered six 100k runs ([#496](https://github.com/NGL321/patchworks/issues/496))
to test it.

The mechanism is available on paper, and it is not exotic. **The prediction rule's update to `K` is
an outer product whose row space is the evidence direction; under constant evidence with a persistent
error the updates are coherent and rank one; and the operator band converts a growing rank-one part
into shrinkage of every other mode.** Oja's rule with one input converges to that input's direction;
this is the same fact with a normalisation attached. The collapse is therefore a property of the
learning rule under low-rank input, not of the band, the floor, the taper, or the dome — which is what
#477 measured from the other side when it ruled the band out.

## 1. The update is an outer product **[derived]**

A predicting cell's prediction is `p = D K h + b`, with `h = encode(chart, evidence) ∈ ℝ¹²` the
frozen body's one nonlinear map, `K ∈ ℝ¹²ˣ¹²` the learned operator, `D ∈ ℝ³²ˣ¹²` the frozen readout
gauge (`body.py`, `CellBody.advance` and `decode`; ADR-0014). The prediction rule descends
`½‖p − target‖²` on `(b, K)` (`learning.py`, `prediction_error`), the target being the node stalk
reconciliation left behind. With `e = p − target`,

```
∂L/∂K = (Dᵀ e) hᵀ
```

— rank one, column space along `Dᵀe`, row space along `h`. After `T` ticks at rate `η`,

```
K_T = a·I − η · Σ_t (Dᵀ e_t) h_tᵀ ,     K_0 = a·I   (body.py, `CellOperators.__init__`, a = 1.0)
```

The whole question is whether the sum is coherent.

## 2. Two regimes **[derived]**

- **Incoherent.** If `h_t` turns over in many directions and `e_t` is uncorrelated with it, the sum is
  a random walk in matrix space: `‖ΔK‖ ~ η√T`, spread across all 144 entries. The operator drifts
  slowly and stays full-rank.
- **Coherent.** If `h_t = h` is constant (constant evidence, chart at its fixed point) and `e_t` has a
  persistent component `ē`, then `ΔK ≈ −ηT (Dᵀē) hᵀ`: rank one, norm growing **linearly** in `T`.

Vision L1 cells are in the first regime: their evidence is reconciled every tick by eight
neighbours whose own predictions move, so `h_t` is chatter. The somatomotor column's boundary
neighbours are three proprioceptive, three touch and the actuator cells, all constant once the arm
stops ([#120](https://github.com/NGL321/patchworks/issues/120)); the apex's eight drive lanes carry
`DRIVE_ASSERTION = 1.0` forever ([#137](https://github.com/NGL321/patchworks/issues/137)). Those are
the second regime, and they are the cells that die.

## 3. The band turns growth into collapse **[derived]**

The cell computes with `K / max(1, σ_max(K))`. Before [#466](https://github.com/NGL321/patchworks/issues/466)
the stored `K` was rescaled after each step (`CellOperators.project`); after it the rescale is
inside the forward path. **The used operator is the same either way**, so the change of enforcement
mechanism does not touch what follows; only the rate differs.

Write the coherent operator as `K = αI + β û hᵀ` with `û = Dᵀē/‖Dᵀē‖`, `‖h‖ = 1`, and
`c = hᵀû` (in general small: `Dᵀē` lives in the readout's coordinates, `h` in `encode`'s). Once
`β ≫ α`, `σ_max ≈ β` and the used operator is

```
K_used ≈ (α/β)·I + û hᵀ
```

For `c = 0` this family has, with `k = 12`:

| statistic | value in the family | as `α/β → 0` |
|---|---|---|
| `σ_max` | pinned at 1 | 1 |
| `ρ(K)` | `α/β` (the rank-one part is nilpotent) | → 0 |
| `σ_min` | `≈ (α/β)²` | → 0 faster than `ρ` |
| stable rank `‖K‖²_F / σ²_max` | `(k(α/β)² + 1) / σ²_max` | → 1 |
| non-normality `‖KᵀK − KKᵀ‖_F / ‖K‖²_F` | `√2 / (k(α/β)² + 1)` | → `√2` |

That is #477's signature exactly: `σ_max` at 0.9947 fleet-wide, `ρ` 3–7x beneath it, stable rank
toward 1, non-normality at the `√2` ceiling [#357](https://github.com/NGL321/patchworks/issues/357)
derived for a rank-one operator. And it says why the band is not the cause while the projection
fires hardest at the apex ([#335](https://github.com/NGL321/patchworks/issues/335) crossing at 20k):
the rescale is the *mechanism of the collapse's expression*, not its source. Remove the band and the
raw `K` would instead grow without bound along `û hᵀ` — a different failure, not health.

**Rates.** Under the post-step projection, once `β` sits at the band's face each step adds
`η‖Dᵀē‖` to `β` and then rescales everything by `1/(1 + η‖Dᵀē‖)`, so `α` decays geometrically at
rate `≈ η‖Dᵀē‖`. Under forward normalisation the raw `β` grows linearly and `α/β ∝ 1/T`. Both reach
the same corner; the projection gets there sooner.

## 4. The toy **[toy]**

`prototypes/audit-collapse-toy/collapse_toy.py`, pure Python, keeps only §1 and §3 and drops
`encode`'s fusion, reconciliation and the chart round trip. One cell, `η = 10⁻²`, `a = 1`, a
persistent error of norm 0.3 with white noise 0.1 on top. The used operator's statistics:

```
constant evidence direction, persistent error
  t      sigma_max   rho     stable_rank   nonnormality
   100       1.0    0.938        9.58        0.008
  1000       1.0    0.368        2.23        0.554
  5000       1.0    0.137        1.06        1.332
  9999       1.0    0.178        1.02        1.381

constant evidence direction, persistent error, a decode bias absorbs it
   100       1.0    0.970       10.18        0.003
  9999       1.0    0.954        9.81        0.005

varying evidence direction, persistent error   (control)
  9999       1.0    0.981        8.44        0.011

constant evidence direction, white error        (control)
  9999       1.0    0.843        5.91        0.046
```

Constant evidence with a persistent error collapses within a few thousand steps; either control
holds. The white-error column drifts too, slowly — a random walk in `K` is not nothing, and the
fleet-wide gentle `ρ` decline [#360](https://github.com/NGL321/patchworks/issues/360) credited to the
band is what a random walk under a one-sided band looks like.

## 5. The real dead cells against the family **[measured]**

The committed 100k checkpoints (`prototypes/chart-per-domain-132/132-postfloor-real-train-seed{42,43,44}-100000.json`)
carry per-cell `ρ`, `σ_min`, stable rank and non-normality, not the operators. That is enough to ask
whether the dead cells lie on the family, whose stable rank and non-normality are functions of `ρ`
alone. Read this session, all cells with `modes_retaining = 0` at 100k, `c = 0` prediction beside the
reading:

| seed | dead cells | `ρ` range | stable rank read → family | non-normality read → family | `σ_min` read (family: `≈ ρ²`) |
|---|---|---|---|---|---|
| 42 | 11 (soma L1 ×5, soma L2 ×2, apex ×4) | 0.14–0.35 | 1.03–2.74 → 1.19–1.98 | 0.52–1.30 → 0.58–1.15 | 0.000–0.058 |
| 43 | 6 (apex ×6) | 0.20–0.32 | 1.01–1.72 → 1.37–1.84 | 0.80–1.37 → 0.64–0.96 | 0.003–0.045 |
| 44 | 8 (soma L1 ×5, soma L2 ×2, apex ×1) | 0.06–0.32 | 1.00–1.70 → 1.03–1.88 | 0.82–1.40 → 0.63–1.36 | 0.000–0.016 |
| live cells, median | 139/144/142 | 0.92–0.93 | 9.5–9.7 | 0.03–0.04 | 0.28–0.37 |

Two of the family's three predictions hold cell by cell: stable rank sits within about 0.5 of the
value `ρ` predicts, and non-normality rises toward `√2` as `ρ` falls (seed 44 cell 67: `ρ` 0.058,
non-normality 1.386 read, 1.359 predicted). The third is over-satisfied: `σ_min` is below `ρ²`
everywhere, often by orders of magnitude, so the collapsed operators are more degenerate than the
pure family — a rank-one part plus a non-isotropic remnant rather than a scaled identity, which is
what a real `K` that drifted before it collapsed would leave. The residual scatter in non-normality
(±0.3 around the prediction) is the `c ≠ 0` and remnant terms. Live cells are nowhere near the
family.

The dead set's median `ρ` falls with a decelerating log-log slope (seed 42: −0.30 at 5k, −0.46 at
20k, −0.24 at 100k) — a slow decay, not the sharp knee the toy shows with a 0.3 offset. Under §3's
rate that puts the coherent error at `‖Dᵀē‖ ≈ 10⁻³` to `10⁻²`, which is the magnitude a standing
reconciliation offset would plausibly have at a cell whose neighbours barely move. **That number is
a prediction, not a reading**; see §7.

## 6. Why the bias does not save the dead cells **[inferred]**

The toy's second row shows that a decode bias absorbing the persistent error protects the operator
completely. The real rule trains that bias at the same `η`. So why does the real cell collapse?
Because the target moves with the prediction: reconciliation edits the stalk by
`gain_v Σ_e F_evᵀ(F_ev p − y_e)`, so `e = p − target` **is** that displacement — the standing offset —
and a bias shift `Δb` moves `e` by `gain_v Σ_e F_evᵀF_ev Δb`, which nulls `e` only where every incident
lane already agrees. Agreement is never reached: the transport rule has no fixed point at agreement
and the maps wander at `~η` forever ([#339](https://github.com/NGL321/patchworks/issues/339),
`learning.py` `NORM_FLOOR`), the neighbours keep learning, and
[#202](https://github.com/NGL321/patchworks/issues/202) found not one tick in 100,000 on which the
fold-margin bound was clean. The persistent error is regenerated each tick faster than the bias can
absorb it. This is the one link in the chain nobody has read at a dead cell, and §7's first read is
designed to read it.

## 7. Predictions, and the reads that decide them

All five are pre-registered here so that a later session cannot fit the story to the reading.

1. **Alignment.** At a dead cell the top right singular vector of `K` (raw or used) aligns with the
   cell's mean `encode` output `h̄` over the run: `|cos| ≫ 1/√12 ≈ 0.29`. At live cells, chance. *Read:*
   one 5k-tick run on the real dome (seed 42, ~6 minutes at #132's rate) logging per cell `K`, `h̄`
   and the mean prediction-error vector `ē` over the last 1,000 ticks. If the dead cells' `K` row
   space is unrelated to `h̄`, this note is wrong.
2. **Excitation rank.** The participation ratio of each cell's evidence stream over a window — the
   instrument [#154 §3](https://github.com/NGL321/patchworks/issues/154) named for edges, applied to
   cells — separates dead from live at least as well as column identity does (#477's `R² = 0.63–0.68`).
   Same run.
3. **The persistent error.** `‖ē‖` at apex and somatomotor cells is of order `10⁻³`–`10⁻²`, direction-stable
   across windows; at vision cells it is smaller and direction-unstable. Same run.
4. **Enforcement mechanism.** The collapse survives #466's forward normalisation unchanged in endpoint
   and somewhat slowed in rate. *Read:* the #335 re-read #466 already owes.
5. **Interventions.** (a) `c = η_K/η < 1` slows the collapse roughly in proportion; (b) a leak
   `K ← K − μ(K − aI)` bounds the rank-one part at `η‖Dᵀē‖/μ` and so bounds `ρ` away from zero; (c)
   an exogenous arm ([#496](https://github.com/NGL321/patchworks/issues/496)) reads `g > 0` at the
   somatomotor column and leaves the apex untouched, because the apex's constancy is the drive's,
   not the arm's ([#481](https://github.com/NGL321/patchworks/issues/481)).

## 8. The same law on the maps **[derived]**, and what it does to composition

The transport gradient at an endpoint is `∂/∂F_v ‖F_v x − y‖/(‖F_v x‖ + ‖y‖) = (…) xᵀ` — an outer product
with the cell's own stalk `x`. Under a static trajectory every map learns one direction. ADR-0032's
floor projects each map onto the nearest scaled co-isometry, which keeps `U` and `V` and flattens
`Σ`: it holds `m` directions **open** but cannot say which `m − 1` directions the trajectory never
excited should be. Those are whatever construction and drift left, and across seven hops
unrelated `m`-dimensional subspaces compose at chance — which is exactly the flat-plus-chance null
[#436](https://github.com/NGL321/patchworks/issues/436) measured at composed effective rank 1.107.
So [#497](https://github.com/NGL321/patchworks/issues/497)'s "a per-map floor that succeeds per map
does not spread it" is not a defect in the floor. It is the law: **composed channel rank is bounded
by the rank of the variation the trajectory excites along the path**, and a floor can only keep
lanes from closing before variation arrives. The remedy for composed rank is variation, not another
per-map constraint.

## 9. What this note does not cover

It does not run the real model. The toy has no `encode`, no reconciliation, no neighbour and no
chart round trip, so it says which way the collapse runs and why, not how fast. §6 is inferred from
#202 and #339 rather than read at a dead cell. The family fit in §5 is consistent-with, not
confirmation: three seeds, twenty-five cells, four statistics that were computed for another purpose.
Prediction 1 is the decisive read, and it costs one short run.
