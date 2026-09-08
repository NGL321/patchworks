# B13 (#560): what `p` is, and where the reserve arm's floor peaks

The instrument is `prototypes/cold-start/T6/`, brought onto current `main`
(`13c015b`, [PR #558](https://github.com/NGL321/patchworks/pull/558) landed) from
`worktree-cs-555`, where [PR #559](https://github.com/NGL321/patchworks/pull/559)
still sits unmerged against a base that has since merged.

New here: `sweep_p.py` (item 1), `curve_p.py` (item 2), and `reserve_p<N>` arms in
`arms.py::ARMS` so both the construction sweep and `trained_arms.py` take an
arbitrary `p` unchanged. `check_allocator_matches_shipped` is untouched and still
passes, so the rig has not drifted from `graph.allocate_lane_widths`.

[B11](https://github.com/NGL321/patchworks/issues/555)'s `p = 8` runs are
**reused, not re-run**: this rig's `reserve_p8` reproduces their construction
figures exactly — ER 1.3824568, generic 2.2652, floor 1200, cap binding 2 of 409
— so they are the same object under an older label, and re-running them would
have cost 2.3 hours to re-measure a number that already exists.

---

## 1. The construction sweep, and the curve that is not there

`p` = 0 … 28 at budget 63, seeds 42 and 43,
`560-sweep-p-construction.json`.

| `p` | `k_v` | floor | cap binds | generic ER | built ER (s42 / s43) |
|---|---|---|---|---|---|
| 0 | 32 | 0 | 0/409 | 2.0232 | 1.5884 / 1.4868 |
| 2 | 30 | 300 | 2/409 | 2.1444 | 1.5150 / 1.5237 |
| 4 | 28 | 600 | 2/409 | 2.1679 | 1.5713 / 1.4268 |
| 6 | 26 | 900 | 2/409 | 2.1154 | 1.5286 / 1.4781 |
| 8 | 24 | 1200 | 2/409 | 2.2652 | 1.3825 / 1.4680 |
| 10 | 22 | 1500 | 2/409 | 2.3624 | 1.5167 / 1.3295 |
| 12 | 20 | 1800 | 2/409 | 2.4305 | 1.4116 / 1.3949 |
| 14 | 18 | 2100 | 10/409 | 2.6128 | 1.3335 / 1.4162 |
| 16 | 16 | 2400 | 10/409 | 2.9283 | 1.3433 / 1.4000 |
| 18 | 14 | 2700 | 70/409 | 3.3277 | 1.3681 / 1.4982 |
| 20 | 12 | 3000 | 70/409 | **4.0000** | 1.2485 / 1.2165 |
| 22 | 10 | 3300 | 194/409 | **4.0000** | 1.2015 / 1.1978 |
| 24 | 8 | 3600 | 194/409 | **4.0000** | 1.0697 / 1.1595 |
| 26 | 6 | 3900 | 194/409 | **4.0000** | 1.0470 / 1.0403 |
| 28 | 4 | 4200 | 194/409 | **4.0000** | 1.0056 / 1.0108 |

**The two instruments disagree about the sign of the whole effect.**

The **generic** reading does what
[#556](https://github.com/NGL321/patchworks/issues/556) predicted — narrowing
`k_v` at fixed `m` raises `m / k_v` and composed rank climbs, 2.02 → 2.93 by
`p = 16`. Then it **saturates at exactly 4.0000** for `p ≥ 20`, and the exactness
is the tell: 4 is the modal chain's first-hop width, and a composed operator
reads ER exactly `m` only when its spectrum is flat — that is, when every
principal angle is zero. The median leading cosine confirms it, 0.9316 (`p = 0`)
→ 0.9784 (8) → 0.9994 (16) → **1.0000** (`p ≥ 18`). Past `p = 18` the lanes fill
the whole permitted block, the two carried subspaces at each relay cell
**coincide**, and the generic hop becomes orthogonal by construction.

The **built** surface does the opposite, and there is **no interior peak at
construction at all**. Composed ER is highest at `p = 0` (1.54 mean over seeds),
flat within seed spread across `p = 0…6` (~1.50–1.54, and the seeds differ by up
to 0.14, so nothing in that range is separable), and downhill from there — 1.43
at 8, 1.40 at 12, 1.37 at 16, 1.23 at 20, 1.01 at 28.

**Why they part is [B11](https://github.com/NGL321/patchworks/issues/555)'s
departure finding, read on a new axis.** B11 found the whole generic-to-built gap
was singular-value spread, not genericity — the generic instrument idealises the
hop to a partial isometry and the real one is not. Coincident carried subspaces
are exactly where that idealisation is most expensive: with the angles gone the
generic hop is orthogonal and reads full rank, while the real hop keeps its
spread and the composite is **dominated by its leading direction**, which is
[B1](https://github.com/NGL321/patchworks/issues/537)'s mechanism. So at
construction, raising `p` hands the composite to domination faster than it hands
it rank.

**This paragraph is where a construction-only ruling would have stopped, and it
would have been wrong.** Section 2 trains the same arms and the ordering
reverses completely: the singular-value spread that costs the built surface its
rank at construction is *itself* a training-time object — ADR-0032's band closes
`σ_min/σ_max` to 1.0000 within 100 ticks — so the very quantity that makes high
`p` look bad here is gone by the time it matters. Nothing in this section
survives as a ruling; it survives as the record of a trap.

The `n − p` cap is cheap up to `p = 12` (2 of 409 interior edges, as #556
measured at 8) and then bites: 10 edges at `p = 14–16`, 70 at 18–20, 194 at 22
and above, where the modal chain's widths start falling (12 → 10 → 8 → 6 → 4).
The `dim H⁰` floor is exactly linear, `150p`, because `p_v = p` is flat over 150
predicting cells.

**So item 1's question — where does composed ER peak, and is the peak flat or
sharp — has a construction answer that cannot be ruled on:** it peaks at `p = 0`,
where the floor is zero and the entire purpose of #556's unwelding is
surrendered. That is the answer B11 warned would be wrong, and item 2 is why.

---

## 2. The trained ladder, which reverses the ordering completely

Baseline condition, seed 42 unless noted, `560-reserve_p<N>-*.json`.

| `p` | floor | cos gap at construction | construction ER | **trained ER @20k** | erosion | seed 43 @20k | @100k |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0547 | 1.5884 | 1.0845 | ×6.97 | — | — |
| 4 | 600 | 0.0394 | 1.5713 | 1.1824 | ×3.13 | — | — |
| 8 ([B11](https://github.com/NGL321/patchworks/issues/555)) | 1200 | 0.0266 | 1.3825 | 1.4386 | ×0.87 | 1.5307 | 1.4237 |
| 12 | 1800 | 0.0163 | 1.4116 | **2.1309** | ×0.36 | **2.0328** | **2.2127** |
| 16 | 2400 | 0.0080 | 1.3433 | **2.9171** | ×0.18 | **2.8922** | **2.9408** |
| 20 | 3000 | 0.0000 | 1.2485 | 4.0000 | ×0.08 | — | — |

**The construction curve and the trained curve run in opposite directions.**
Construction falls monotonically in `p`; trained rises monotonically in `p`. This
is [B11](https://github.com/NGL321/patchworks/issues/555)'s *"a construction
reading is the wrong number to rule on"* in its sharpest possible form — not a
caveat about a couple of arms inverting, but a **sign flip across the entire
sweep**. Had this ticket ruled on item 1 alone it would have chosen `p = 0`, the
one value on the curve with **no privacy floor at all**, and reported it as the
peak.

**Two arms clear [B2](https://github.com/NGL321/patchworks/issues/540)'s bar of
2.0 on a trained surface, which nothing on this map had done.** The previous best
trained composed rank anywhere on the record was
[B6](https://github.com/NGL321/patchworks/issues/546)'s 1.186, then B11's 1.4237.
`p = 12` reads **2.2127 at 100k** and `p = 16` reads **2.9408**. Both are still at
or above their 20k values at the horizon, so this is a floor rather than a point
on a decay.

**The mechanism is [B6](https://github.com/NGL321/patchworks/issues/546)'s
headroom law, and `p` is its dial.** B6 found that training collapses the
composite by pulling the leading per-hop cosine off the rest, and can only do so
where construction left a gap. The construction gap and the erosion factor move
together across all six arms, monotonically and over two orders of magnitude:

    gap 0.0547 -> x6.97    gap 0.0163 -> x0.36
    gap 0.0394 -> x3.13    gap 0.0080 -> x0.18
    gap 0.0266 -> x0.87    gap 0.0000 -> x0.08

Raising `p` narrows `k_v`, which forces the incident carried subspaces together,
which closes the gap — so **`p` is a direct dial on how much headroom training
has to concentrate**. That is why the two curves have opposite signs: `p` costs
construction rank and buys back more than it costs by denying training the room
to erode.

### What `p` actually buys the rank with, and the price

`sweep_p.py`'s generic column saturating at *exactly* 4.0000 for `p ≥ 20` is not
a rounding artifact, and `forced_overlap.py` says what it is. At a relay cell two
lanes of widths `m_in`, `m_out` are carved from the same `k_v`-dimensional
readable block, so whatever the maps learn they must intersect in at least
`max(0, m_in + m_out − k_v)` dimensions. Over all 1,578 hops of the 256 chains:

| `p` | `k_v` | forced overlap, median hop | hops with any forced overlap | fully coincident hops |
|---|---|---|---|---|
| 0 | 32 | 0 (0%) | 11/1578 | 4 |
| 4 | 28 | 0 (0%) | 338/1578 | 4 |
| 8 | 24 | 0 (0%) | 564/1578 | 4 |
| 12 | 20 | 4 of 12 (33%) | 1322/1578 | 19 |
| 16 | 16 | 8 of 12 (67%) | 1322/1578 | 181 |
| 20 | 12 | 12 of 12 (100%) | 1578/1578 | **1450** |

**This is a real cure and not a measurement illusion, and it is worth being
precise about why.** The collapse
[#533](https://github.com/NGL321/patchworks/issues/533) diagnosed is seven
successive projections onto *differing* subspaces, which annihilate everything
but their common direction. Forcing the subspaces to coincide is the direct
negation of that: nothing is projected away, and the chain transports its four
dimensions faithfully. `p = 20`'s 4.0000 is genuine faithful transport, not a
number the instrument invented.

**The price is selectivity, not rank.** At `p = 20` every interior lane carries
the cell's entire readable block — `m = k_v = 12` — so an interior edge performs
no selection whatsoever, and `06-graph-topology.md`'s account of the cell as a
compressor stops being true of the interior. The reserve buys composed rank by
shrinking the ambient until neighbouring lanes are *the same lane*, and past some
point that is a different architecture rather than a better-tuned one. The
readable block is also what
[#548](https://github.com/NGL321/patchworks/issues/548)'s allocator spends: the
`n − p` cap starts lowering lane widths at `p = 14` (10 edges), 70 at `p = 18–20`,
194 at 22 and above.

---

## 3. Does the floor survive at every `p`, or only at 8?

B11's falsifier, in its own words: *if a reserve arm at another seed, another
`p`, or a longer horizon falls below its construction value, the resolution is
wrong and the erosion is universal after all.* This ticket runs the `p` half, and
the seed half for the two candidates.

**The falsifier fires, on the `p` axis, at `p = 0` and `p = 4`.** Both fall well
below their construction values — 1.0845 against 1.5884, and 1.1824 against
1.5713. So **"the reserve arm does not erode" is not a property of the reserve
mask**; it is a property of `p ≥ 8`, and B11 read the one value where it happened
to hold. The corrected statement is B6's law with `p` as its dial: erosion is
governed by the construction cosine gap, `p` sets that gap, and the crossover
from eroding to gaining sits between `p = 4` (×3.13) and `p = 8` (×0.87).

**What does not fall is everything at `p ≥ 8`**, on every axis the falsifier
names: another seed (`p = 12` reads 2.0328 at seed 43 against 2.1309 at seed 42,
`p = 16` reads 2.8922 against 2.9171, both gaining), another `p` (8, 12, 16, 20
all gain), and a longer horizon (`p = 12` rises 2.0713 → 2.2127 from 20k to 100k;
`p = 16` holds 2.9202 → 2.9408).

**The `dim H⁰` floor itself is not at risk on any of these axes, and it is worth
separating the two senses of "floor" the falsifier blurs.** The `dim H⁰` floor is
`150p` exactly, at every `p`, because `p_v = p` is flat at all 150 predicting
cells by construction — it is a property of the mask, not of a trajectory, and
training cannot move it. It is re-read on the trained arms at 100k and is
unchanged (1800 at `p = 12`, 2400 at `p = 16`). What B11's falsifier is actually
about is whether *composed rank* holds above its construction value, and that is
the question answered above.

---

## 4. `p`'s derivation

*(Written before the trained ladder landed, because it does not depend on it.)*

**The record has one candidate and it is not the one `p` was invented as.**

`DomeSpec.privacy_budget`'s `@flexibility` annotation carries the invariant's
reason, which [#548](https://github.com/NGL321/patchworks/issues/548) wrote down
and nothing before it had:

> a cell will likely need more features to compute its own dynamics than it holds
> an authoritative position on network-wide, so the budget reserves the
> difference.

Under the union mask that reason had no number attached — the reserved amount was
whatever `n − Σ_e m_e` happened to leave, which is why it read **zero at 104 of
150 cells** at budget 63 and why #556 unwelded it. Under the reserve mask the
reserved amount **is** `p`, flat and chosen, so the reason now demands a
magnitude: *how many features does a cell need to compute its own dynamics?*

**The record answers that question with a constant it already has.** `body.py`
defines `CHART_DIM = 12` as *"`k`, the chart dimension — the cell's private
low-dimensional coordinates, and the memory depth its operator advances."* The
chart is precisely the cell's own-dynamics state: `encode` fuses stalk evidence
into it, the cell's own learned `K` advances it and nothing else does, and
`decode` reads it back onto the stalk. So the count of directions a cell needs
for its own dynamics is `k = 12`, and `p = k` is a derivation of the reserve from
a constant already on the record rather than a new invention.

**Two things that derivation does not give, stated rather than papered over.**

1. **It derives a magnitude, not a subspace.** The private block is the trailing
   `p` coordinates of the stalk in the stalk's own basis; `decode`'s image is a
   `≤ k`-dimensional subspace of the stalk that is not aligned to those
   coordinates and is not held fixed under training. So `p = k` says *a chart's
   worth of stalk is withheld*, not *the chart's own directions are withheld*.
   Making the second true is a different change from setting a constant.
2. **`k` itself is `stipulated`.** Its own annotation says so, and puts it on
   rung 5 of the flex ladder. Deriving `p` from `k` inherits `k`'s standing; it
   does not manufacture a warrant neither has. What it does buy is that `p`
   stops being **a second free constant** — it becomes a restatement of one the
   architecture already committed to, which is the difference
   [B4](https://github.com/NGL321/patchworks/issues/539) found `m` lacked when it
   ruled `m` had *"neither a derivation nor a vindicated value"*.

**What is refused as a derivation.** The world's state dimension —
`06-graph-topology.md`'s *"a world whose state is about twenty numbers"*, and the
sandbox's `qpos (3,) + qvel (3,) + touch (3,)` — is a **global** quantity, while
`p_v` is per cell and flat over 150 of them. Sizing every cell's private block by
the whole world's state confuses a fleet-wide number for a local one, and
[#271](https://github.com/NGL321/patchworks/issues/271) already measured the
per-cell private-width gradient's retention claim null (correlation −0.107 …
+0.047), so there is no measured per-cell demand to size against either. The
piece's box dimension is likewise refused: it sizes a **lane** (what an edge
carries), which is `m`, and [B5](https://github.com/NGL321/patchworks/issues/542)
and [#474](https://github.com/NGL321/patchworks/issues/474) have that axis.


---

## The recommendation

**`p = 12`, and the reason is that it is the only value on the curve that is both
derived and measured.**

| | `p = 8` (B11's) | **`p = 12`** | `p = 16` |
|---|---|---|---|
| derivation | none — taken from #556 and never swept | **`p = k = CHART_DIM`** | none |
| trained ER @100k | 1.4237 | **2.2127** | 2.9408 |
| clears the bar of 2.0 | no | **yes, on both seeds** | yes, with margin |
| `dim H⁰` floor | 1200 | **1800** (today's is 914) | 2400 |
| forced lane overlap | 0% | **33%** | 67% |
| `n − p` cap binds | 2 of 409 | **2 of 409** | 10 of 409 |

`p = 12` clears [B2](https://github.com/NGL321/patchworks/issues/540)'s bar on a
trained surface at both seeds and at the horizon, nearly doubles the shipped
`dim H⁰` floor, costs the allocator the same 2 edges of 409 that `p = 8` already
costs, leaves two-thirds of every lane free rather than forced, and is the value
`CHART_DIM` hands you rather than one this ticket picked to win.

**Its margin over the bar is thin and that is stated, not buried.** The worst of
the three readings is seed 43 at 20k, **2.0328** — 0.03 above 2.0, against a
measured run-to-run drift of **0.060** on this arm. So `p = 12` clears the bar in
every reading taken, but at 20k it does not clear it by more than the noise. The
100k reading is the one with margin (2.2127), and it is also the one B11's
finding says to trust; a horizon run at seed 43 is the cheapest thing that would
settle it and was not run here.

**`p = 16` is the higher number and is not recommended.** It buys +0.73 composed
rank for double the forced overlap (67% of the median lane, 181 hops fully
coincident against 19), five times the cap binding, and it has no derivation — it
would be `p = 8`'s problem again one ticket later, a swept constant with a good
reading and no reason. If the bar rather than the derivation is what matters to
the user, the measured case for `p = 16` is strong and this readout does not hide
it; the ranking between them is a **judgement about what the reserve is for**, not
about which number is bigger, and it is the user's.

**`p ≥ 18` is refused.** At `p = 20` every hop is fully coincident and the
composite reads exactly its own width; an interior edge selects nothing. That is
a different transport architecture, not a setting of this one, and this map's
destination is a parameterisation change with its ADR cost named — not a silent
retirement of the interior lane.

---

## What this does not settle

- **The derivation is of a magnitude, not a subspace** (§4). Aligning the private
  block with `decode`'s image is a further change and nobody has priced it.
- **No horizon run at a second seed.** `p = 12`'s 100k reading is seed 42 only.
- **Only the baseline condition.** No pinned-drive arm was run at any `p`; #555's
  ladder had the same shape and the same limit.
- **The forced-overlap price has no bar attached.** 33% is reported because it is
  measurable, not because any document says what fraction is acceptable. If the
  architecture wants a limit on how much of a lane may be forced, that is a new
  invariant and it does not exist yet.
- **`Σ_e m_e ≤ B` is still misnamed.** [B12](https://github.com/NGL321/patchworks/issues/556)
  ruled it survives as a capacity budget rather than a privacy invariant, and
  `DomeSpec.privacy_budget` still carries the old name on `main`. That is
  [B15](https://github.com/NGL321/patchworks/issues/562)'s to write, along with
  everything above.
