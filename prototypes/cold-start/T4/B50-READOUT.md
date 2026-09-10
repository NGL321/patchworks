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
