# T4 (#537): the principal angles, and which lever moves composed rank

Reads [#537](https://github.com/NGL321/patchworks/issues/537) under
[the transport map](https://github.com/NGL321/patchworks/issues/532), on the
mechanism [#533](https://github.com/NGL321/patchworks/issues/533) established.

**Surface.** Full dome (`DEFAULT_SPEC`, forward normalisation, `interior_m = 3`,
`boundary_m = 4`), `map/cold-start`, seeds 42/43/44, 263 rim-to-apex chains of
**7 hops each**. The instrument is `hop_operator` / `composed_reads` from
`prototypes/cold-start/T2/run.py`, unmodified; `trained.py` reproduces T3's
composed-ER ladder to five decimals at every checkpoint it shares with it
(1.0204 @100, 1.0125 @200, 1.0089 @500, 1.0078 @1000), which is what says this
is the same surface #524 read.

## Three premises of the ticket that did not hold

**1. The surfaces are not on disk.** #537 says "nothing here needs building or
running — the surfaces are already on disk in `prototypes/cold-start/T3/`".
T3's npz holds the chart operators `K`, the per-cell reads and the per-chain
composed effective rank; it does **not** hold `agent.sheaf.maps`, and the angle
read is a function of exactly those. Construction is rebuildable in seconds, so
items 1–6 are all readable there; the trained arm needed the ticks re-run
(`trained.py`, ~2 hours per 100k arm).

**2. At construction the maps are not co-isometries.** #537 writes the hop as
`σ²·U_out(V_outᵀV_in)U_inᵀ` on the strength of ADR-0032's floor holding to
`σ_min/σ_max = 0.9999995`. But `project()` runs *after* a transport step, so on
the untrained surface the floor has never been applied: measured
`σ_min/σ_max` has **median 0.444** (p10 0.367, max 0.563) across all three
seeds. From tick 100 onward it is 0.999999 and the co-isometry form holds
exactly. The ledger's **1.025 at construction is a pre-floor reading**. It
barely matters numerically — applying the projection moves seed 42's median from
1.0255 to 1.0335 — but the algebra #537 reasons with is only true of the trained
surface.

**3. The ambient is the relay cell's mask, not the stalk.** All of a cell's
incident maps share one structural mask, a prefix of its node stalk
(`Dome.restriction_mask`; it is also why `_push_apart` can use one shared
right-transform per cell). So `V_in` and `V_out` are `m`-frames in the same
`k_v`-dimensional space, and the genericity that governs the angles is `m/k_v`,
**not** `m/n`. Measured over the 1578 hops: `m = 3` on 1315 hops and `4 → 3` on
the 263 rim hops; `k_v` runs 15–31 with a mode of 18 (1057 hops). #537's
parenthetical — "two 4-frames in ℝ³² being near-orthogonal generically" — is the
wrong pair of numbers; the real ratio is 3/18, not 4/32.

## 1. The angle distribution

Cosines of the principal angles between each relay cell's two carried subspaces,
over all 1578 hops × 3 angles, at construction:

| | min | p10 | median | p90 | max | mean |
|---|---|---|---|---|---|---|
| all `cos θ`, seed 42 | 0.000 | 0.067 | **0.312** | 0.633 | 0.834 | 0.340 |
| all `cos θ`, seed 43 | 0.001 | 0.049 | **0.319** | 0.626 | 0.818 | 0.333 |
| all `cos θ`, seed 44 | 0.000 | 0.059 | **0.326** | 0.627 | 0.859 | 0.338 |
| leading `cos θ` per hop (s42) | 0.474 | 0.507 | **0.568** | 0.639 | 0.688 | 0.572 |
| second `cos θ` per hop (s42) | 0.222 | 0.274 | **0.329** | 0.370 | 0.402 | 0.324 |

No hop is near-aligned and none is near-orthogonal: the leading cosine sits
tightly around 0.57 with a total spread of 0.21 across 1578 hops. That
tightness is itself the finding — the hops are interchangeable. Under training
the median rises slowly (0.332 @100, 0.353 @500, 0.376 @2000) while composed ER
*falls*, which is the first sign that the per-hop angles are not what carries
the composed reading.

## 2. Does the mechanism explain the numbers?

**As #537 states the test, no — but the test is unsound.** Singular values are
not multiplicative along a product, so an elementwise product of seven sorted
cosine spectra is neither an upper nor a lower bound on the composed spectrum,
and its failing falsifies nothing. It does fail: median product-ER 1.0013 /
1.0024 / 1.0020 against observed 1.0255 / 1.0079 / 1.0131, per-chain
correlation +0.16 / −0.05 / +0.14.

**The sound version confirms the mechanism, and more strongly than it claimed.**
`ablate.py` rebuilds each chain's composed operator from progressively less of
the real surface:

| rung | what is kept | median | mean | p90 | max | frac > 1.1 | corr with exact |
|---|---|---|---|---|---|---|---|
| (a) | exact — the truth | 1.0255 | 1.0981 | 1.3096 | 1.9886 | 0.243 | 1.000 |
| (b) | angles only, `U` dropped | 1.0520 | 1.1669 | 1.5686 | 1.9671 | 0.392 | −0.058 |
| (c) | real frames, random `U` | 1.0263 | 1.1153 | 1.3828 | 1.9971 | 0.274 | +0.192 |
| (d) | **random frames, same `(m, k_v)`** | **1.0216** | **1.0948** | **1.2577** | **1.9259** | **0.240** | +0.091 |

(seed 42; seeds 43 and 44 in `537-ablation.json` agree.) Rung (d) throws away
the entire constructed surface and keeps only each hop's dimension counts and
the chain length — and reproduces the real distribution, tail included. So

> **composed effective rank is a function of `(m, k_v)` and the hop count, and
> of nothing else.**

That is a stronger claim than #533's and it subsumes it. The mechanism is
principal-angle collapse, and the angles are generic.

## 3. The 1.641 chain

**The confirmation #537 called "the sharpest available" fails — and the reason
it fails is worth more than a pass would have been.**

The top-5% chains are not better-aligned. On seed 42 their mean second cosine is
0.3244 against the rest's 0.3244 — identical to four decimals — while their mean
ER is 1.676 against 1.068. Correlations of chain ER against every alignment
summary are near zero and do not hold a sign across seeds (`cos_mean` +0.017 /
−0.062 / +0.282; `cos_top_mean` −0.141 / −0.081 / +0.172). Every surrogate
rung's per-chain correlation with the exact ER is near zero too.

But rung (d) reproduces the *whole distribution including its tail*. Both facts
together say the tail is **generic**: a chain wins by the luck of seven draws,
not by having better-aligned relay cells. There is no structurally special
chain to find. The mechanism carries the variance in distribution, which is what
it claimed; it does not license a per-chain story, and #532 should not tell one.

*Provenance note.* #524's "1.641 max over chains" is the **mean of the three
winner-arm per-seed maxima** (1.5135, 1.9376, 1.4717); the baseline's 1.171 is
likewise the mean of (1.0649, 1.4352, 1.0122). It is not a single chain, so
"the 1.641 chain" has no referent.

## 4. Domination, not rank — as #533 predicted

The composite is numerically **full rank 3** (`= m` of the last hop) on the
median chain. Normalised median spectra:

| seed | σ₁ | σ₂/σ₁ | σ₃/σ₁ |
|---|---|---|---|
| 42 | 1.0 | 0.1128 | 0.00225 |
| 43 | 1.0 | 0.0629 | 0.00038 |
| 44 | 1.0 | 0.0810 | 0.00045 |

σ₂/σ₁ reaches p90 0.394 / 0.251 / 0.408 and max 0.926 / 0.975 / 0.908. So the
second direction is present on every chain, roughly one order down on the
median, and on the best chains nearly co-equal. **ER 1.000 is domination, not
annihilation** — the transported information is there, at about a tenth of the
leading direction's weight.

## 5. `c` alone: no, and not by a little

**`c` alone buys nothing, because the constraint it parameterises is slack.**

`c` reaches the surface only through `RestrictionMaps._push_apart`, which caps a
holding cell's summed Gram at `g_v²·c_v` and is an exact no-op at a cell already
inside the bound. So it can only move anything at a cell where
`λ_max(Σ_e F_evᵀF_ev)` actually reaches its target. Measured over the 414 cells
the cap can reach, at construction on all three seeds:

- ratio to target: **median 0.329, p90 0.388**
- cells at the cap: **exactly 4**, on every seed

and those 4 cells are **precisely the cells `c` does not govern**: three touch
cells (`deg = 1`, stalk 1) and the drive (`deg = 8`, stalk 1), all with wholly
pinned incidence, where [#228](https://github.com/NGL321/patchworks/issues/228)
sets `c_v = deg(v)` and the exact gauge makes `Σ_e ‖F‖²_F = deg(v)` an algebraic
identity. They sit at ratio 1.000 because they cannot sit anywhere else.

Of the **150 cells whose `c_v` does change** when `c` goes 2 → 12, the largest
Gram ratio anywhere is **0.229** — slack by a factor of 4.4.

Sweeping `c ∈ {1, 2, 3, 4, 6, 8, 12}` therefore moves composed ER by **nothing
at all**: 1.0335 / 1.0106 / 1.0368 flat to six decimals on seeds 42/43/44. Under
training the sweep stays flat in the relaxing direction; the only movement
anywhere is a 1.2 × 10⁻⁴ wobble at `c = 1` from 2000 ticks on, in the
*tightening* direction.

**On ADR-0010's ~1.05 floor.** #537 asks whether the floor moved under ADR-0032.
The question does not arise: that floor is on **per-map** effective rank, a
different quantity from the composed rank, and `c` is not near binding on either
side of it. ADR-0010's own words — *"a map that transmits one direction cannot
be made incoherent with anything"* — remain true and remain beside the point,
because nothing here is asking `c` to make anything more incoherent than the
random draw already is.

## 6. `m` alone: yes — and it is `m`, not the ratio

On Haar frames at the real chain length (`genericity.py`, 256 chains per cell),
median chain ER at the built `k_v = 18`, 7 hops:

| `m` | 1 | 2 | **3 (built)** | 4 | 5 | 6 | 7 | 8 | 9 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| ER median | 1.000 | 1.000 | **1.011** | 1.066 | 1.175 | 1.378 | 1.625 | 1.897 | 2.162 | 3.601 |
| frac > 1.1 | 0.00 | 0.04 | 0.15 | 0.42 | 0.67 | 0.88 | 0.98 | 1.00 | 1.00 | 1.00 |

Narrowing the mask instead — at fixed `m = 3`, which is the privacy invariant's
direction — barely moves it:

| `k_v` | 18 | 12 | 9 | 6 | 5 | 4 |
|---|---|---|---|---|---|---|
| ER median | 1.010 | 1.017 | 1.013 | 1.045 | 1.088 | 1.215 |

**The two are not one lever seen twice.** At equal ratio `m/k_v = 0.5` they
disagree by a factor of 25 in excess over one: `m = 9, k_v = 18` gives 2.162,
`m = 3, k_v = 6` gives 1.045. The ceiling on composed ER is `m` itself; mask
width only sets how fast that ceiling is approached. Collapsing the mask to
`k_v = 4` still leaves the median at 1.215 with `m = 3`.

Smallest `m` at `k_v = 18`, 7 hops, whose **median** chain clears a stated bar:
1.05 → `m = 4`; 1.1 → `m = 5`; 1.25 → `m = 6`; 1.5 → `m = 7`; 2.0 → `m = 9`.

## Advisory: depth, which #537 does not ask about

Since the law is a function of `(m, k_v, hops)`, the third argument came for
free. At the built `(3, 18)`, median chain ER against hop count:

| hops | 1 | 2 | 3 | 4 | 5 | 6 | **7 (built)** | 9 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| ER median | 1.730 | 1.333 | 1.153 | 1.071 | 1.045 | 1.025 | **1.011** | 1.002 | 1.001 |

This is the map's `degree` fog patch with a number on it. It belongs to
[#540](https://github.com/NGL321/patchworks/issues/540), not here, and it is
recorded rather than acted on.

## What this says about the lever

**`m` on its own is sufficient. `c` on its own is not — not second-order, but
exactly zero.** `c` cannot move composed rank on this surface at any value,
because the cells it governs are slack by 4.4× and the cells at the cap are the
ones it does not govern. And because rung (d) holds, *no* rearrangement of the
surface inside fixed `(m, k_v, hops)` can move composed rank either — which
rules out a whole class of levers at once, not just `c`.

Choosing the lever is [#540](https://github.com/NGL321/patchworks/issues/540)'s
job and is deliberately not done here. #483 is held (#533 §9) and this ticket
does not unhold it; nothing on the live surface was edited.

## Files

- `angles.py` — the angle read, the cap read, and #537's items 1–6 per seed →
  `537-construction.json`
- `ablate.py` — the four-rung ablation ladder → `537-ablation.json`
- `genericity.py` — the `(m, k_v, hops)` law on Haar frames → `537-genericity.json`
- `trained.py` — the same reads on the trained surface, T3's two arms
  → `537-<arm>-seed42-100000.json`
