# B66 (#642) — the alphabet ladder, and what the gate's teeth are actually made of

No runs. Everything here is read off `637-analysis.json` from
[B63 (#637)](https://github.com/NGL321/patchworks/issues/637), on branch
`worktree-b63-dependence-instrument-637`. B63's alphabet labels are **nested**,
so `k' = 2` and `k' = 4` were already recorded beside the headline `k = 8`.

Every number is `traditional` mutual information ([B64, #638](https://github.com/NGL321/patchworks/issues/638)),
on **node stalks** at the world-read boundary ([B49, #616](https://github.com/NGL321/patchworks/issues/616)).
Nothing here is comparable with a lane reading.

Scripts: `b66_ladder.py`, `b66_untrained_check.py`. Both take the path to
`637-analysis.json` and print the tables below.

---

## Why the ladder was run

The user's ruling on B66 is that the ticket's premise is wrong: the flat
bundle out-reading the trained surface is **not a defect**. `I(P; Δ)` is a
transmission reading, and [B42 (#605)](https://github.com/NGL321/patchworks/issues/605)
derived the flat bundle as the one arrangement where a message survives a loop
unchanged. A transmission reading is maximised by it. So nothing separates
them and nothing could.

The user's own account of the 2× is that the probe's task is **too small** for
specialisation to pay. That makes a prediction: the flat bundle's margin
should shrink as the discrimination gets harder. The alphabet is the only
difficulty knob available without enriching the sandbox — which is out of
scope on #532 — and it costs zero runs.

---

## 1. The ladder, on patch — the only stratum with headroom

Margin is `flat − trained` in bits. The ceiling is `log k` and moves with the
rung, so the margin in bits is **not** comparable across rungs; the share of
the ceiling is.

| reduction \| decoder | margin k=2 | k=4 | k=8 | share k=2 | k=4 | k=8 |
|---|---|---|---|---|---|---|
| peak \| centroid *(ADR-0021's own)* | +0.393 | +0.565 | +0.788 | 0.393 | 0.282 | **0.263** |
| peak \| nn1 | +0.461 | +0.430 | +0.709 | 0.461 | 0.215 | **0.236** |
| trace \| centroid | 0.000 | 0.000 | +0.058 | 0.000 | 0.000 | 0.019 |
| trace \| nn1 | 0.000 | 0.000 | **−0.296** | 0.000 | 0.000 | −0.099 |

**The result is not evident, and that is the finding.**

- In **bits** the margin *grows* with `k`. As a **share of the ceiling** it
  *falls* — 0.393 → 0.263 and 0.461 → 0.236. The sign points the user's way on
  the normalised reading and the other way on the raw one.
- The fall is almost entirely `k=2 → k=4`. From `k=4` to `k=8` it is flat on
  one decoder (0.282 → 0.263) and **rises** on the other (0.215 → 0.236).
- Three rungs, one seed, two decoders that disagree. This is not a trend.

## 2. The reduction that carries the content is saturated at every rung

B63 found the `trace` reduction recovers **2.810** bits on patch against the
peak's 0.857 — most of the content is in the window. On that reduction both
arms sit **exactly at the ceiling** at `k = 2` and `k = 4`, on every stratum.
Only `k = 8` on patch has any headroom left, and there **the sign flips with
the decoder**: flat leads by 0.058 under `centroid`, and the **trained surface
leads by 0.296** under `nn1`.

So the flat bundle's advantage is not stable across reductions. It is large and
consistent on `peak`, and absent or reversed on `trace`.

**The probe saturates at every alphabet available to it.** `k = 8` is
insufficient, as the user said.

## 3. Saturation is bijectivity, and it is now measured rather than argued

The trained arm reads the ceiling **exactly** on proprioceptive and touch at
`k = 2`, `k = 4` **and** `k = 8`:

| stratum | dim | at ceiling, k=2 / 4 / 8 |
|---|---|---|
| patch | 12288 | no / no / no |
| proprioceptive | 6 | **yes / yes / yes** |
| touch | 3 | **yes / yes / yes** |

A stratum that reads full marks at every ceiling offered has no headroom at any
alphabet. This settles the ticket's item 2 in the negative: the saturation is
**not** an artifact of choosing 8, and raising `k` cannot fix it. It is the
narrow strata being near-bijective to the terminus.

The one exception proves it. Touch's flat arm reads **2.414** at `k = 8` and
the ceiling at `k = 2, 4` — its 3-dimensional product space cannot hold eight
distinguishable patterns (B63 measured max pairwise `|cos|` **0.9881**), so the
crowding appears only at the rung where crowding starts.

## 4. The gate's teeth are narrower than the map states

Q1's ruling — dependence is a **gate**, not a ranking — rests on the untrained
surface failing it. B63 reported that it fails on every stratum. **That holds
on ADR-0021's own `peak | centroid` reduction and does not hold generally.**

Untrained arm, `k = 8`, against its own 400-permutation null:

| stratum | peak\|centroid | peak\|nn1 | trace\|centroid | trace\|nn1 |
|---|---|---|---|---|
| patch | 0.0698 | 0.5337 | 0.1297 | 0.5387 |
| proprioceptive | 0.0574 | **0.0025** | **0.0025** | **0.0025** |
| touch | 0.3042 | **0.0150** | 0.1920 | **0.0025** |

Bold clears at `p ≤ 0.05`. On the narrow strata the **untrained** construction
clears the null under three of four reductions, reading up to **1.561** bits
(proprioceptive, `trace|nn1`, excess 1.208 over its null).

Two consequences, and neither is small:

1. **The map's B63 entry needs a qualifier.** *"It does not clear on the
   untrained one on any stratum"* is true of the headline reduction only.
2. **B58's discharge is scoped to patch.** The map records B63 as discharging
   [B58 (#630)](https://github.com/NGL321/patchworks/issues/630)'s *a
   construction may supply the quantity free* in the architecture's favour,
   *"the first time on this map that it has been."* On the narrow strata, under
   the reduction that carries the content, **the construction does supply it
   free.** The discharge stands on **patch** and only there.

## 5. Patch is the only stratum that carries a verdict, twice over

Two independent lines converge on it:

- **Headroom** (§3): the narrow strata saturate at every alphabet, so no two
  arms can differ there.
- **Teeth** (§4): patch is the only stratum where the untrained arm fails
  under every reduction and both decoders — `p` never below 0.53 on `nn1`,
  never below 0.07 on `centroid`.

A dependence claim is quoted on **patch**. The other two strata are reported
as saturated and as carrying no verdict.

---

## What this does not show

**It does not test the user's bet.** The bet is that a parsed-up problem is
solved better by specialists whose communication is tailored per neighbour, and
that a flat sheaf constrains the cells themselves. The alphabet is a weak proxy
for task difficulty and `k = 8` is a small task by any measure. A margin that
does not move across three rungs of a saturated probe says the **probe** cannot
see the effect. It says nothing about whether the effect is there.

One measurement does point at the cells-are-constrained half, and it is not
this one: B42 read the flat bundle's rank-measured `k_v` at **14.8 of 32**
against the control's **20.0**. One reading, one surface — support, not proof.

**It does not reach the terminus.** Whether the read may widen past the 3-wide
`commanded` block is [B67 (#643)](https://github.com/NGL321/patchworks/issues/643)'s,
and untouched here. §3's saturation is the strongest evidence yet for B67's
item 1 — narrow strata near-bijective to the terminus saturate, the wide one
does not — and B67 should have it.

## Files

- `b66_ladder.py` — §1, §2, §3. Input `637-analysis.json`.
- `b66_untrained_check.py` — §4. Same input.
