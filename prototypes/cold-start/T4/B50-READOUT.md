# B50 (#618): the per-hop cosine spectra, read as spectra

[#618](https://github.com/NGL321/patchworks/issues/618), under
[the map](https://github.com/NGL321/patchworks/issues/532). Existing surface
only, as ruled — no constructed arm, no imposed blocks, no candidate scored.

**Surface.** `eedb781` (2026-09-09), full dome, `n = 32`, 263 rim chains,
6 interior hops per chain, 682 carried edges (`m_e > 0`), all of them carried.

**Instrument.** `b50_spectra.py` (the spectra) and `b50_forced.py` (the
structural control). Both take #537's `angles.py` object verbatim — the
principal-angle cosines between a relay cell's two carried subspaces — and
change only the statistic taken over them.

---

## 0. The premise the ticket inherited is stale: the surface moved under #548 and #562

#618 pre-registers *"#537's construction table already implies the answer is
interior — its max over 1578 × 3 is 0.834–0.859, so **no cosine reaches 0.95
anywhere at construction**. Confirm."*

**It does not confirm.** Running #537's *own* `read_surface`, unmodified, on
today's `main`:

| #537's `cos_all`, seed 42 construction | as filed (`0a22a1c`) | today (`eedb781`) |
|---|---|---|
| median | 0.312 | **0.932** |
| p90 | 0.633 | **1.0000** |
| max | **0.834** | **1.0000** |
| composed ER median | 1.025 | **1.412** |
| composed ER max | 1.989 | **2.869** |

Same instrument, same seed, same chains. Two commits that post-date #537's data
moved the surface:

- **[#548](https://github.com/NGL321/patchworks/issues/548)** (`24e6c5f`, wrote
  #540's ruling) deleted `DomeSpec.interior_m` and allocates interior lanes
  **per edge**. #537 read a uniform `m ∈ {3, 4}`; today's widths are
  `{4, 11, 12, 15, 16, 20}`.
- **[#562](https://github.com/NGL321/patchworks/issues/562)** (`5ecaf3e`, wrote
  #556's reserve mask) set `k_v = n − p`, so `k_v` is **20 at every interior
  hop**. #537 read a `k_v` histogram spanning 15–31.

This is the #455 rule firing exactly as written: a rig's data ages with `main`.
Two consequences, both load-bearing:

1. **The map's headline collapse figure is a fact about a surface that no longer
   exists.** *Composed rim-to-apex ER 1.025 at construction* — quoted on #532 as
   *"the strongest single statement of the problem"* — reads **1.412** on the
   same instrument today, with a chain max of 2.869. It was not re-read after
   #548. This readout does **not** rule on what that means; it reports it.
2. `t0.surface()` had been throwing `AttributeError` on `DEFAULT_SPEC.interior_m`
   since #548 — every T-rig entry point was dead on `main`. Repaired here.

---

## 1. Corner mass — and the whole of it is forced by dimension counting

Over the 395 **unique** interior hops the 263 chains induce (2730 cosines):

| seed | ≥ 0.99 | ≥ 0.95 | ≤ 0.05 | interior (0.05, 0.95) | max |
|---|---|---|---|---|---|
| 42 | 0.3055 | **0.3553** | 0.0198 | 0.6249 | 1.0000 |
| 43 | — | **0.3575** | 0.0150 | 0.6275 | 1.0000 |
| 44 | — | **0.3619** | 0.0158 | 0.6223 | 1.0000 |

Under #537's own instance weighting (all 1578 hop instances, duplicates kept —
the population its 1578 × 3 names), the corner mass is **larger**, not smaller:
**≥ 0.95 → 0.4838**, interior → 0.4904. The deduplicated figures above are the
conservative ones and are reported as primary.

So there *is* a real mass at the top corner and effectively none at the bottom.
But the mass at the top is **not earned**:

> Two subspaces of dimensions `m_in` and `m_out` inside one `k_v`-dimensional
> mask must intersect in at least `f = max(0, m_in + m_out − k_v)` dimensions,
> and every direction of that intersection is a principal angle of exactly
> zero — `cos θ = 1`.

**The identity `#{cos ≥ 1 − 1e-6} == f` holds at 395 of 395 hops, on all three
seeds, with no exceptions.** Not approximately — exactly, hop by hop.

| seed | forced mean `f` | measured `r@0.95` mean | **excess** | identity |
|---|---|---|---|---|
| 42 | 2.086 | 2.456 | 0.370 | 395/395 |
| 43 | 2.086 | 2.471 | 0.385 | 395/395 |
| 44 | 2.086 | 2.501 | 0.415 | 395/395 |

`f` is identical across seeds because it is a function of the widths and the
mask and of nothing else — no seed, no objective, no tick can move it. **30.18%
of every cosine on this surface sits at the corner before a map is drawn.**

---

## 2. The shape: a flat top of exactly the forced height, then an interior ramp

Each spectrum normalised by its own leading value, median over the hops of that
width. The forced height `f` is computed from the widths alone and is *not*
fitted:

| hop | `f` | median normalised spectrum |
|---|---|---|
| `4→12`, `4→11` (263 hops) | **0** | 1.0, 0.90, 0.76, 0.57 |
| `11→11` (20 hops) | **2** | **1.0, 1.0,** 0.99, 0.94, 0.87, 0.77, 0.66, 0.51, 0.38, 0.22, 0.06 |
| `12→12` (84 hops) | **4** | **1.0, 1.0, 1.0, 1.0,** 0.96, 0.89, 0.80, 0.69, 0.57, 0.38, 0.23, 0.06 |
| `15→15` (20 hops) | **10** | **1.0 × 10,** 0.84, 0.71, 0.50, 0.32 |
| `20→20` (4 hops) | **20** | **1.0 × 20** |

The count of exact ones is `f` in every row. **The spectrum is neither #618's
corner-pinned `(1, 1, 0)` nor its interior `(1, 0.6, 0.4)`. It is a pigeonhole
step of height `f` followed by a smooth interior ramp** — and where `f = 0`
(the 263 `4→12` and `4→11` hops, two-thirds of the population) it is a pure
interior ramp with no flat top at all, exactly as `f` predicts.

The four `20→20` hops are the limiting case: both lanes span the entire 20-dim
mask, so all 20 cosines are 1 and transport is exact in every direction. **The
flat bundle's condition `r = m` already occurs, unremarked, in the built dome.**

---

## 3. Direction of travel: training moves mass **toward** the corner, and narrows the ratio

Seed 42, both T3 arms, the checkpoint ladder to 20k. `f` is structural and
**cannot move**, so every increment below is unforced excess.

| ticks | baseline ≥0.95 | baseline `r@0.95` | winner ≥0.95 | winner `r@0.95` | winner aud-diff |
|---|---|---|---|---|---|
| 0 | 0.3553 | 2.456 | 0.3553 | 2.456 | 0.5307 |
| 500 | 0.3766 | 2.603 | 0.4117 | 2.846 | 0.5061 |
| 2000 | 0.3985 | 2.754 | 0.4333 | 2.995 | 0.4727 |
| 5000 | 0.4132 | 2.856 | 0.4491 | 3.104 | 0.4407 |
| 10000 | 0.4231 | 2.924 | 0.4524 | 3.127 | 0.4309 |
| 20000 | **0.4172** | **2.884** | **0.4337** | **2.997** | **0.4540** |

*Run-to-run spread, measured — both arms were run twice at seed 42, by accident.*
A session teardown orphaned the first run rather than killing it, so each arm has
an independent replicate:

- **baseline**: 0.4099 vs 0.4132 at 5k, 0.4161 vs 0.4231 at 10k — agree to 0.007.
- **winner**: 0.4524 vs 0.4502 at 10k, 0.4337 vs 0.4344 at 20k — agree to 0.002.

Identical seeds and arms, so this is the rig's own nondeterminism, not a
condition. The trajectory (+0.06 to +0.10 in corner mass) is an order of
magnitude larger than it and is quoted no finer. The tabled numbers are the runs
this session drove; the replicates are in the git history of the same files.

**Mass moves toward 1, monotonically, in both arms**, fastest in the winner
(which reaches by 500 ticks what the baseline needs 5000 for) and saturating by
10k, with a small retreat at 20k in the winner. The bottom corner never fills:
`≤ 0.05` stays at 0.011–0.020 throughout and *falls* slightly.

#618's item 3 asks to separate *widening the ratio* from *approaching the
corners*, which #537's statistic cannot do. Taking **#537's own statistic** —
median leading cosine over median second cosine, per hop:

| | #537, old surface | today, baseline | today, winner |
|---|---|---|---|
| construction | 1.728 | **1.089** | **1.089** |
| 20k | 2.021 (**widening**) | **1.079** (**narrowing**) | **1.058** (**narrowing**) |

So on today's surface the two motions come apart and point **opposite ways**:
the ratio narrows — the top of each spectrum flattens as more directions tie the
leader — while mass moves toward the corner. #537's widening is not a property
of the per-hop cosines on the current surface.

## 4. `r` as measured

Per hop, count of cosines above the threshold, chain population, seed 42:

| | construction | baseline 20k | winner 20k |
|---|---|---|---|
| `r@0.90` mean | 3.023 | 3.344 | 3.552 |
| `r@0.95` mean | 2.456 | 2.884 | 2.997 |
| `r@0.99` mean | 2.111 | 2.342 | 2.473 |
| `r@0.95` **median** | 1.0 | 1.0 | 1.0 |
| `r@0.95` **max** | 20 | 20 | 20 |
| **forced floor `f`** | **2.086** | **2.086** | **2.086** |

**`r ≈ 1` is true only at the median.** The mean is 2.46–3.00 and the
distribution is heavy-tailed to a max of 20 — four hops where `m_in = m_out =
k_v = 20`, both lanes spanning the entire mask, so all twenty cosines are 1 and
transport is exact in every direction.

## 5. Audience differentiation (#532's standing constraint from B42)

Mean over each interior cell's carried-edge pairs, `1 − Σcos²θ / min(m_in, m_out)`;
B42's flat bundle is 0.0000 by this measure at its stagger-0 row.

| | median | p10 | p90 | cells |
|---|---|---|---|---|
| construction s42 | **0.5307** | 0.2923 | 0.7484 | 142 |
| construction s43 / s44 | 0.5578 / 0.5453 | | | |
| baseline 20k | **0.4712** | | | |
| winner 20k | **0.4540** | | | |

Nowhere near the flat bundle, and **falling monotonically with training** —
0.5307 → 0.4309 at the winner's 10k trough. Training moves this surface *toward*
B42's failure mode on both instruments at once: more directions shared at gain
1, fewer distinct things said to distinct neighbours. It has a long way still to
go.

---

## Verdict against #618's branch table

**(C) fires, and (B)'s first half fires without its second.**

- **(A) Interior everywhere** — *does not fire*. Corner mass is 0.355 at
  construction and 0.43–0.45 trained, not ≈ 0, and training does move mass
  toward 1.
- **(B) Already near-corner** — *fires by half*. There is a real mass at
  `≥ 0.95`; there is **no** matching mass at `≤ 0.05` (0.011–0.020, and
  falling). (B) is a conjunction and its second clause fails.
- **(C) Training moves mass toward the corners** — **fires**, monotonically, in
  both arms, saturating by 10k.

So (C)'s consequent applies: *#616's load-bearing claim needs restating.* It
needs restating **more sharply than (C) anticipated**, because both of its
halves are wrong in the same measurement.

> #616 §3: *"So `r ≈ 1` is **not what the geometry forces. It is what the
> objective buys.**"*

1. **Geometry forces the bulk of it.** `f = max(0, m_in + m_out − k_v)` is
   2.086 of a measured 2.456 — 85% of `r` at construction, 30.18% of every
   cosine on the surface — and the identity is exact at 395/395 hops on three
   seeds. No seed, objective or tick can move `f`.
2. **The objective buys *more* corner, not less.** Training raises `r@0.95` from
   2.456 to 2.997 and lowers audience differentiation from 0.531 to 0.454. If
   the objective is buying anything here it is buying **isotropy of the shared
   part** — the flat bundle's direction — not the `r ≈ 1` concentration #616
   attributed to it.
3. **`r ≈ 1` is a median, not the distribution.** Mean 2.46–3.00, max 20.

**What this does to #616's candidate third move.** Anisotropy was proposed as an
unvisited region: `r` directions at gain 1, a different block per neighbour.
The measurement says **the region is already occupied, and the width allocation
put it there.** Since #548, `f` is a per-hop, route-specific, exactly-flat
sub-bundle whose dimension is set by `m_in + m_out − k_v` — which is precisely
#616 §3's *"flat part that is `r`-dimensional and route-specific"*, arrived at
by nobody choosing it. So §3's open question (*does the topology do work again
when the flat part is `r`-dimensional and route-specific?*) is **answerable on
the existing surface** rather than needing a construction, and `r` is a design
variable only in the sense that **the width allocation is one** — `r` is not a
new knob, it is `allocate_lane_widths` read in a different basis.

**Nothing is scored and no candidate is opened here**, per the ticket's scope
and #616's standing risk that both known escapes score well on every instrument
this map owns.

## What is outside the table entirely

The stale-surface finding in §0 is not a reading of the spectra and no branch
covers it. It goes to the surprise ledger ([#520](https://github.com/NGL321/patchworks/issues/520))
as a row amending **row 6**, whose *composed rank 1.025 at construction* is
quoted on #532 as the strongest single statement of the problem and reads
**1.412** today. Per the ledger's own rule no grilling is minted from it.
