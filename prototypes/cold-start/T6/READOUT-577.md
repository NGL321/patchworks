# B28 (#577): gauge, overfitting, or real?

Rigs: `prototypes/cold-start/T6/b28_gauge.py`, `b28_generalise.py`.
Raw: `577-gauge-*.json`, `577-generalise-*.json`, `577-transplant-*.json`. The trained maps
(`577-maps-*.pt`, 8.7 MB) are **not committed** — they are what `b28_generalise.py` reads, and
`b28_gauge.py` rewrites them in ~25 min per arm.

Two arms, seed 42, 20,000 ticks: **`reserve_p16`** (the map's best trained arm,
[B13](https://github.com/NGL321/patchworks/issues/560)/[B20](https://github.com/NGL321/patchworks/issues/569);
`k_v = 16`, lanes 1–16) and **`shipped`** (the surface
[B1](https://github.com/NGL321/patchworks/issues/537) measured; `k_v ≈ 18`, lanes 3–4).

**Headline.** The gauge horn is **dead** — every quantity #532 rests on is an exact gauge
invariant, so genericity is a statement about the maps. The overfitting horn is **dead** —
there is no train/held-out gap because there is no in-sample fit to lose. But the
literal claim *"the learned subspaces are indistinguishable from Haar frames"* is
**false on the shipped arm and gets steadily more false with training**, and what it is
false in the direction of is the collapse itself.

---

## 1. The gauge group, stated

A restriction map `F_{v◁e}` lives on the block `[:m_e, :k_v]` — `m_e` live rows of the edge
stalk, and the leading `k_v` columns of the cell's node stalk, the mask prefix **every one of
a cell's incident maps shares** (`RestrictionMaps.support` / `column_mask`; asserted per cell
by `b28_gauge.cell_widths`). The admissible transformations are

    per predicting cell v :  O_v ∈ O(k_v)  on the exposed block
    per edge e            :  Q_e ∈ O(m_e)  on the edge stalk

acting as `F_{v◁e} ↦ Q_e F_{v◁e} O_vᵀ`, with the cell states carried along, `h_v ↦ O_v h_v`.
Every constraint the build imposes is equivariant under it:

| object | why it does not move |
|---|---|
| the support | the rectangular block `[:m_e, :k_v]`, which `O_v` and `Q_e` preserve |
| the band (`gauge_bounds`, `norms`) | Frobenius, orthogonally invariant |
| ADR-0032's floor (`_flatten`) | a function of the singular values alone |
| ADR-0010's cap (`_push_apart`, `gram_peaks`) | `λ_max(Σ_e FᵀF) ↦ λ_max(O (Σ_e FᵀF) Oᵀ)` |
| the transport objective | `‖F_u h_u − F_v h_v‖²`, and `Q_e` is shared by the edge's two ends |

**It is orthogonal, not general-linear, and that is the whole answer to
[B24](https://github.com/NGL321/patchworks/issues/573).** B24 states the freedom as *"act on
each stalk by an invertible map and adjust incident restrictions"*. Under `GL` the composed
spectrum would indeed **not** be invariant — but `GL` is not admissible here: ADR-0032's
co-isometry constraint, the Frobenius band and the Euclidean disagreement norm each break it
to the orthogonal subgroup. A `GL` change of basis does not relabel this surface, it moves it
off the constraint set the learning problem is defined on. **So the identifiability statement
B24 could not find is: a learned sheaf here is identified up to `O(k_v) × O(m_e)`, and every
quantity #532 records is invariant under exactly that group.**

**Where it is fixed rather than free.** `O_v` is unavailable at a boundary cell (the world
writes its stalk in fixed coordinates) and at a cell whose whole incidence is pinned
(`pinned_incidence` — the exact gauge fixes those maps outright): **264 cells fixed, 150
free**. No relay cell on a rim-to-apex chain is among the fixed ones.

## 2. What that does to the composed object

A chain's hop is `F_out F_inᵀ` with **both maps held by the same relay cell**, so

    (Q_out F_out O_vᵀ)(O_v F_inᵀ Q_inᵀ) = Q_out (F_out F_inᵀ) Q_inᵀ

`O_v` cancels *inside every hop*, and along a chain each interior `Q_e` meets its own
transpose. The composed operator transforms by `Q_last (·) Q_firstᵀ` — an orthogonal
equivalence. Therefore **composed effective rank is an exact gauge invariant, and so are the
principal angles between a relay cell's two carried subspaces.**

Measured rather than asserted: `orbit_read` moves the surface along its own gauge orbit —
three draws, two respecting the fixed cells and one ignoring them — and re-reads everything.

| re-read after a random gauge | worst absolute change |
|---|---|
| composed ER median / p90 / max, all `cos θ`, leading & second cosine per hop, `σ₂/σ₁`, `σ_min/σ_max`, Gram peak, map norms | **9.1 × 10⁻⁷** (construction) · **6.3 × 10⁻⁷** (worst over all six checkpoints) |
| edge Dirichlet energy on fixed probe states (relative) | **5.3 × 10⁻⁹** · **1.0 × 10⁻⁸** |

That is float32 round-off. Dropping the fixed-cell restriction changes nothing, as the
algebra says it must not.

**So item 2 needs no restatement.** [B1](https://github.com/NGL321/patchworks/issues/537) rung
(d) and [B11](https://github.com/NGL321/patchworks/issues/555)'s 1–3% both compare
*distributions of composed ER*, which is already a gauge invariant. **Genericity is a
statement about the maps, not about an unfixed gauge.**

## 3. Searching for the structure

Because the composed spectrum and the principal angles cannot be moved by *any* change of
basis, a search over admissible gauges for a basis in which the maps look non-generic cannot
change a single ruling on this map. **That search is not inconclusive; it is provably empty.**

What the invariance argument does *not* settle is structure against a frame that is
**physical rather than coordinate**. Two were read, both gauge-invariant by construction.

**(a) Traffic alignment.** Per (cell, interior lane), `capture = tr(VᵀCV)/tr(C)` with `V` the
carried subspace and `C` the cell's **centred** state covariance over a 1000-tick window
([B18](https://github.com/NGL321/patchworks/issues/567) is why centred). Both transform as
`O_v(·)O_vᵀ`, so `capture` is invariant. Scored as a z against a Haar null drawn in the same
`k`-space **against the same `C`**, so the null carries `C`'s own anisotropy.

**(b) Sibling structure.** All pairs of a cell's interior lanes, leading principal-angle
cosine against a Haar null at the same `(m_a, m_b, k)` — item 3's *"shared leading directions
across a cell's edges"*, read per cell over its whole incidence.

Lanes with `m ≥ k_v` span the whole exposed block, so capture is 1 and so is every null draw;
they are excluded throughout (44 of 818 on `reserve_p16`, 0 on `shipped`).

| ticks | arm | traffic z | traffic pct | lanes > null p95 | sibling lead | sibling z | composed ER | world `std_max` | moving |
|---|---|---|---|---|---|---|---|---|---|
| 100 | `reserve_p16` | **−0.17** | 0.472 | 0.057 | 0.888 | **+0.00** | 2.969 | 9.42 | 0.344 |
| 300 | | +0.22 | 0.625 | 0.150 | 0.894 | +0.00 | 2.989 | 4.07 | 0.026 |
| 1 000 | | +0.27 | 0.641 | 0.140 | 0.899 | +0.00 | 2.947 | 2.22 | 0.007 |
| 3 000 | | +0.32 | 0.665 | 0.151 | 0.900 | +0.06 | 2.879 | 0.48 | 0.009 |
| 10 000 | | +0.31 | 0.674 | 0.160 | 0.908 | +0.28 | 2.877 | 0.23 | 0.002 |
| 20 000 | | +0.27 | 0.640 | 0.183 | 0.902 | +0.23 | 2.933 | 0.03 | 0.000 |
| 100 | `shipped` | **−0.35** | 0.444 | — | 0.478 | **+0.06** | 1.330 | — | — |
| 300 | | +0.04 | 0.629 | — | 0.495 | +0.22 | 1.278 | — | — |
| 1 000 | | +0.38 | 0.716 | — | 0.539 | +0.61 | 1.067 | — | — |
| 3 000 | | +0.63 | 0.790 | — | 0.596 | +1.15 | 1.010 | — | — |
| 10 000 | | +0.99 | 0.860 | 0.329 | 0.653 | +1.78 | 1.002 | — | — |
| 20 000 | | +0.81 | 0.830 | 0.304 | 0.675 | **+2.05** | 1.001 | — | — |

**Two things follow, and they pull against each other.**

1. **The literal genericity claim fails on the shipped arm, and increasingly so.** A cell's
   incident lanes end up **2.05 standard deviations** more mutually aligned than Haar frames
   of the same shape, leading cosine 0.478 → 0.675, rising monotonically at every checkpoint
   (1936 lane pairs, none degenerate). Traffic alignment moves the same way: **30.4%** of
   lanes sit above their own Haar null's p95, against the 5% the null predicts.
   *"Statistically indistinguishable from Haar frames of the same shape"* is not what this
   surface reads. On `reserve_p16` the same statistic barely moves (+0.00 → +0.23), which is
   consistent with its lanes being wide enough that there is little room to align further.

2. **What it aligns toward is the collapse, not discovered meaning.** Composed ER falls
   1.330 → 1.002 over exactly the checkpoints where sibling alignment climbs. That is the
   mechanism [B9](https://github.com/NGL321/patchworks/issues/551) already established from
   the other side — `c = 1` pushes a cell's incident maps *apart* and destroys composed rank,
   so pushing them *together* is what collapse looks like in this coordinate-free statistic.
   The structure training finds is **a cell's lanes agreeing to carry the same direction**,
   which is the thing that makes seven hops compose to one.

**And the honest caveat, which changes how much of (1) to believe.** The world stops moving:
boundary `std_max` runs 9.42 → 0.03 and the fraction of moving components 0.344 → 0.000 across
the ladder (#572's advisory, confirmed here independently). **At tick 100, the one checkpoint
where the world is demonstrably live, traffic alignment is at or below the Haar null on both
arms (z −0.17, −0.35), and sibling alignment is exactly at it (+0.00, +0.06).** Every point of
the rise happens after the world has gone quiet. So the alignment training accumulates is
alignment with **the graph's own endogenous relaxation**, not with world-driven traffic — which
is a much weaker thing than "the architecture discovered what two cells share", and is the
reading [B25](https://github.com/NGL321/patchworks/issues/574)'s sheaf-diffusion result
predicts.

## 4. The overfitting alternative

[B25](https://github.com/NGL321/patchworks/issues/574) named the test: *"does in-sample edge
agreement generalise to held-out episodes?"* The instrument is, per interior edge,

    rel = ‖F_u h_u − F_v h_v‖² / (‖F_u h_u‖² + ‖F_v h_v‖²)

scale-free and gauge-invariant (`Q_e` cancels). Since `var_d = var_u + var_v − 2·cov`, the
**centred** form satisfies `rel = 1 − 2·cov/(var_u+var_v)`, so `1 − rel` is exactly the
normalised covariance between the two ends' fluctuations — the **agreement coefficient**: 1 is
perfect agreement on what varies, 0 is none.

**Reading the centred form is not optional.**
[B18](https://github.com/NGL321/patchworks/issues/567) found the mean carries 99.6–99.9% of a
cell's stalk energy, and it dominates here too: uncentred, the trained surface beats its
untrained control by **2×** on a live world — and by a spurious **~18,000×** on a dead one.

### The in-process evaluation could not answer it, and why that is worth recording

`PlanarPushSandbox` motion collapses within a few hundred ticks, and **`reset()` does not
revive it** — the sampler is explicitly not allowed to change the arm's pose
(`sandbox/env.py`, `PLACEMENT_ATTEMPTS`), so a stalled arm stays stalled through a
rearrangement. On the 20k surface, boundary `std_max` read **7 × 10⁻⁸ with 0.0000 of
components moving** in *every* evaluation window, early and late, on all four rungs.
Disagreement measured there is a ratio of two numerical zeros. Those numbers are in
`577-generalise-*.json` and **none of them is evidence about transport.**

### The transplant, on a world that is moving

`b28_gauge.py` saves the trained maps (`577-maps-*.pt`), so they are transplanted into a
**freshly built agent per world arrangement** — body not yet stalled — and read over ticks
10–140, where the world is live. The control is the same fresh build with construction maps,
so the two differ in exactly one thing. `reserve_p16`, 20k, six arrangements per rung:

| arm | rung | agreement coefficient (trained) | (untrained) | uncentred `rel` trained / untrained | world `std_max` | moving |
|---|---|---|---|---|---|---|
| `reserve_p16` | `train_seen` (the exact worlds trained on) | **+0.00058** | −0.00050 | 0.496 / 1.020 | 1.66 | 0.235 |
| | `train` (same slice, unseen) | **+0.00054** | −0.00027 | 0.496 / 1.020 | 1.62 | 0.247 |
| | `heldout_pair` | **+0.00048** | −0.00005 | 0.497 / 1.020 | 1.67 | 0.237 |
| | `heldout_sector` | **+0.00034** | −0.00040 | 0.497 / 1.020 | 4.11 | 0.243 |
| `shipped` | `train_seen` | **+0.00981** | +0.00614 | 0.313 / 0.993 | 1.64 | 0.133 |
| | `train` | **+0.00993** | +0.00580 | 0.313 / 0.993 | 1.64 | 0.132 |
| | `heldout_pair` | **+0.00987** | +0.00654 | 0.313 / 0.993 | 1.64 | 0.131 |
| | `heldout_sector` | **+0.00999** | +0.00582 | 0.313 / 0.993 | 1.74 | 0.140 |

**The overfitting hypothesis is struck, and not because the lanes generalise.** After 20,000
ticks the two ends of an interior edge agree on what varies to a normalised covariance of
**0.0006** (`reserve_p16`) and **0.0098** (`shipped`) — six parts in ten thousand, and one
part in a hundred — against 0.0000 and 0.0061 for their untrained controls. Training moves
the shipped arm's agreement by **+0.0037 absolute**, and the reserve arm's by +0.0011.

**There is no in-sample/out-of-sample gap because there is no in-sample fit.** The four rungs
are indistinguishable from one another on both arms: `shipped` reads +0.00981 on the *exact
worlds it trained on* and +0.00999 on a held-out sector — the held-out number is, if
anything, the larger. B25's regime — *"near-perfect in sample and noise out of sample"* —
requires the in-sample half, and it is absent. The lanes are not overfit; they were never fit.

What training *does* buy is the uncentred half: `rel` 1.020 → 0.496 and 0.993 → 0.313, a
genuine 2–3× on the standing component. **The transport rule reduces disagreement on what a
cell holds steadily, and leaves what varies very nearly untouched.**

## 5. What it does to the record

**Nothing is retracted on the gauge horn.** Every reading at issue is stated in gauge
invariants:

| ruling | status |
|---|---|
| [B1](https://github.com/NGL321/patchworks/issues/537) rung (d) — composed ER is a function of `(m, k_v, hops)` | **stands**; the compared quantity is invariant |
| [B11](https://github.com/NGL321/patchworks/issues/555) — subspaces generic to 1–3% | **stands as a statement about composed ER**; see the qualification below |
| [B13](https://github.com/NGL321/patchworks/issues/560), [B14](https://github.com/NGL321/patchworks/issues/561), [B20](https://github.com/NGL321/patchworks/issues/569) — everything read off composed ER | **stands**; invariant |
| [B24](https://github.com/NGL321/patchworks/issues/573) caveat 5 | **discharged, negatively** — the admissible gauge is orthogonal, not `GL`, and the record's quantities are invariant under it |
| [B25](https://github.com/NGL321/patchworks/issues/574)'s overfitting alternative | **struck** — no in-sample fit exists to fail out of sample |

**Two things want qualifying, and one is a correction to how a claim is worded.**

1. **[B17](https://github.com/NGL321/patchworks/issues/565) should not lean on *"statistically
   indistinguishable from Haar frames"* as stated.** Its conclusion — that nothing in this
   architecture discovers what two cells share — is now supported *more directly* than before,
   by §4: the objective's agreement on what varies is one part in a hundred or less after
   20,000 ticks, on seen and unseen worlds alike. But the premise it actually cites is false
   on the shipped arm (sibling z **+2.05** at 20k; 30.4% of lanes above their traffic null's
   p95, against 5% predicted). The right sentence is **"the learned lanes depart
   from Haar only by aligning with each other, which is the collapse"**, not "they are
   indistinguishable from random".

2. **An instrument warning that generalises [B18](https://github.com/NGL321/patchworks/issues/567)'s
   ledger row from the state to the objective.** Uncentred edge disagreement is dominated by
   the shared standing component: it reports a 2× improvement as ~18,000× once the world stops
   moving. Any reading anywhere on #517 or #532 that quotes falling disagreement or Dirichlet
   energy as evidence that transport is learning needs the centred form and a motion stamp.
   Filed as a ledger candidate rather than acted on here.

## What was **not** measured

- **Only seed 42, and only 20,000 ticks**, on two arms. Two runs of the same arm at the same
  seed drift: this ticket's two independent `reserve_p16` runs read composed ER 2.964 and
  2.947 at 1k. The gauge deltas and the traffic/sibling z-scores reproduced across both runs;
  the composed ER medians did not, to better than ~0.02.
- **No `reserve` (`p = 8`) arm**, which is the surface B11's 1–3% was actually measured on.
  The two arms here bracket it.
- **The trained agent's own trajectory on a moving world.** Unavailable at any horizon on this
  rig, for the reason in §4; the transplant carries `sheaf.maps` only, and `K`, the biases and
  the stalks are the fresh agent's.
- **No search over gauges was run**, because §2 proves it cannot move any recorded quantity.
  The claim rests on the algebra plus the orbit measurement, not on an optimisation.
- **Why sibling alignment rises on `shipped` and not on `reserve_p16`** is conjectured
  (lane width leaves less room to align) and not tested.
- Composed rank, joint span, ripple distance, forced overlap — B20/B21/B22's, untouched here.
