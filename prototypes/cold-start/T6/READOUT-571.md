# B22 (#571): the largest region over which a direction stays consistent

*Instrument: `prototypes/cold-start/T6/b22_regions.py`, tables cut by
`b22_analyse.py`. Raw: `571-regions-construction.json` (8 arms, 16 level-stratified
seeds, 3 growth families, 2 row conventions) and
`571-regions-<arm>-baseline-seed42-20000.json` (3 arms trained to 20k). Surface:
branch `worktree-b22-coherent-571` off `main`; rig `T0`-`T6` taken from
`worktree-b20-joint-span-569`.*

## The answer in one line

**The earned part of `dim H⁰` is exactly zero coordination on every arm measured —
not small, exactly the number dimension-counting predicts, eight arms out of eight —
so the split this ticket asked for turns out to separate the trivial part from
*slack*, not from structure. But the regional question underneath it comes back
positive: on the arms where regions are finite at all, directions are held over
substantial regions that do cut across levels, and the user's sharper claim
survives a deliberately hostile test — a cross-level region carrying **8-9× the
constraints per cell** holds exactly as many directions as a within-level one of
the same size, up to `p = 12`. Above `p = 12` that stops being true, and at
`p = 16` — the arm scoring best on this map's composed-rank bar — the median
direction is consistent over **one cell**. None of it is learned: across three
arms and nine checkpoints, 20k ticks of training move every one of these numbers
by exactly zero, and `restriction.py`'s mask says why.**

## 0. What was built, and what it is checked against

`dim H⁰` on this map is `ker δ_P`, where `δ_P` is the coboundary on the predicting
cells' node-stalk components (`diagnostics.py:762-832`). The instrument rebuilds
that matrix and **reproduces `Diagnostics.whole_graph` to the integer** — `dim H⁰`
2323 and `dim H¹` 0 on `shipped`, 1200 and 269 on `reserve` — before anything is
concluded. That check is the point: what follows is a split of the map's own
number, not of a differently-rounded one. `b22_regions.py check` runs it.

**The split.** A cell `v`'s incident restriction maps, stacked, span `U_v` of
dimension `k_v`; its complement `U_v^⊥` is killed by every incident map, so
anything supported in `⊕_v U_v^⊥` is a global section for free.

    trivial = Σ_v (n − k_v)        earned = dim H⁰ − trivial

`trivial ⊆ ker δ` by construction, so the subtraction carries no tolerance and
`earned ≥ 0` is exact.

**Regions.** A region `R` is a set of predicting cells; `H⁰(R) = ker δ_R`. The
trace `P_c(R) = {s(c) : s ∈ H⁰(R)}` shrinks monotonically as `R` grows and always
contains `U_c^⊥`, so `P_c(R) = U_c^⊥ ⊕ E_c(R)` exactly and the **earned trace**
`E_c(R)` has `dim E_c(R) = dim P_c(R) − (n − k_c)`, starting at `k_c`. It is
computed from two ranks, `dim P_c(R) = n − rank δ_R + rank δ_R^{(c deleted)}`,
with no null space ever formed.

Because the traces are **nested**, `dim E_c(R_r)` *is* the number of directions at
`c` whose region reaches radius `r`, and the size distribution follows from the
dimensions alone — no direction is named or tracked. A direction alive at radius
`r` is held across every cell of `R_r`, so `R_r`'s level span is that direction's
level span. Item 3 is exact rather than inferred.

**Two row conventions, because they bracket the question.** `internal` counts only
edges with both ends in `R` — the sheaf-correct reading of a region's own
consistency. `sealed` also carries `R`'s edges to boundary cells homogeneously,
which is `diagnostics.py`'s own convention and the one that reproduces `dim H⁰`.
Both are reported throughout; they differ a great deal and neither is *the*
answer.

**Growth stops for a reason, and the reason is recorded.** A region that ran out
of graph is right-censored, not dead. `lateral` is 85-92% censored — a level has
only 8 to 70 cells — and pooling that with a direction that actually died would
be a lie about where directions end.

## 1. The earned part is exactly the counting prediction, on every arm

`earned` means nothing without the value a surface in general position would show.
Delete the private columns, which are null by construction, and a generic surface
has `earned = max(0, (columns − trivial) − rows)`. Coordination — sections that
exist because the maps *agree* — would have to appear **above** that line.

| arm | p | rows (Σm_e) | dim H⁰ | trivial | earned | generic | earned−generic | dim H¹ |
|---|---|---|---|---|---|---|---|---|
| shipped | – | 2477 | 2323 | 914 | 1409 | 1409 | **+0** | 0 |
| doubling | – | 3885 | 915 | 54 | 861 | 861 | **+0** | 0 |
| reserve | 0 | 3885 | 915 | 54 | 861 | 861 | **+0** | 0 |
| reserve | 4 | 3877 | 923 | 600 | 323 | 323 | **+0** | 0 |
| reserve | 8 | 3869 | 1200 | 1200 | 0 | 0 | **+0** | 269 |
| reserve | 12 | 3861 | 1800 | 1800 | 0 | 0 | **+0** | 861 |
| reserve | 16 | 3823 | 2400 | 2400 | 0 | 0 | **+0** | 1423 |
| reserve | 20 | 3591 | 3000 | 3000 | 0 | 0 | **+0** | 1791 |

**Eight arms, eight exact hits.** The `shipped` arm's 1409 earned sections are not
1409 coordinated directions; they are the slack of a system with 2477 constraint
rows against 3886 non-private columns. Widen the lanes — every `reserve` arm runs
3591-3885 rows — and the slack disappears, reaching exactly zero from `p = 8` on.

Two things follow, and both bear on rulings this map has already made.

**The `150p` floor is exact.** `trivial` reads 600, 1200, 1800, 2400, 3000 at
`p = 4, 8, 12, 16, 20` — precisely `150p`, on 150 predicting cells. The ticket
suspected this; it holds to the integer. Raising `p` inflates the uncoordinated
part and nothing else.

**The construction bound the record quotes is tight, not loose.** `Σ_v
private_dimensions` (`diagnostics.py:967`), the combinatorial `max(0, n − Σ_e m_e)`
count, equals the rank-measured `trivial` on every arm. So the record's practice of
comparing `dim H⁰` against that bound was comparing it against the whole of itself
wherever `earned` is zero — which, from `p = 8` up, is everywhere.

## 2. Region sizes: bimodal, and finite only where the slack is gone

Sixteen level-stratified seed cells per arm; every earned direction at each seed
contributes one region size.

| arm | p | k_v med | directions | frac size 1 | median | p90 | max | levels med |
|---|---|---|---|---|---|---|---|---|
| shipped | – | 25 | 369 | 0.09 | 150 | 150 | 150 | 7 |
| doubling | – | 32 | 446 | 0.08 | 150 | 150 | 150 | 7 |
| reserve | 0 | 32 | 446 | 0.08 | 150 | 150 | 150 | 7 |
| reserve | 4 | 28 | 392 | 0.09 | 150 | 150 | 150 | 7 |
| reserve | 8 | 24 | 336 | 0.20 | **27** | 44 | 87 | **5** |
| reserve | 12 | 20 | 280 | 0.30 | **7** | 12 | 25 | **3** |
| reserve | 16 | 16 | 224 | 0.57 | **1** | 7 | 7 | **1** |
| reserve | 20 | 12 | 168 | 0.67 | **1** | 7 | 7 | **1** |

*(`sealed` rows; 150 = the whole predicting subcomplex.)*

**The distribution is bimodal, not spread.** The size histograms are two spikes —
the whole graph, or a handful of cells — with very little between. On `shipped`:
335 directions over all 150 cells and 34 over exactly one, and nothing else. On
`reserve p = 16`: 127 over one cell, 68 over seven, 29 over five. What the user
hoped for — *many overlapping regions of substantial size, each covering a
different partly-shared slice* — is visible only at `p = 8`, where the sizes
genuinely spread (48 directions at 27 cells, 35 at 40, 24 at 22, 23 at 30, 22 at
34, and 66 at one).

**Under `internal` rows the same bimodality is sharper still**: at `p = 16` the
median region is 7 cells but the p90 is 150 — four of the fourteen seeds hold
directions consistent over the entire graph while the rest hold almost nothing.
Regions are not uniformly sized and no single number describes them, which is why
the ticket's insistence on a distribution rather than a number was right.

## 3. Large regions do cut across levels — and it costs nothing, up to p = 12

A BFS ball on this dome reaches three or four levels by radius 2, so **the ball
family cannot answer item 3**: size and level-span are not independent properties
of a ball. Two matched families separate them:

* `lateral` — BFS confined to the seed's own level, so span is pinned at 1.
* `ladder` — one cell per step, always to the highest-level neighbour available.

Compared at **equal cell count**, `ladder − lateral` is the earned directions
spanning levels costs.

| arm | p | ladder/lateral rows | matches | ladder−lateral med | at par | worse | min |
|---|---|---|---|---|---|---|---|
| shipped | – | 4.6× | 55 | +0.00 | 100% | 0% | 0 |
| doubling | – | 8.4× | 55 | +0.00 | 100% | 0% | 0 |
| reserve | 0 | 8.4× | 55 | +0.00 | 100% | 0% | 0 |
| reserve | 4 | 8.4× | 55 | +0.00 | 100% | 0% | 0 |
| reserve | 8 | 8.4× | 55 | +0.00 | 100% | 0% | 0 |
| reserve | 12 | 8.4× | 55 | +0.00 | 100% | 0% | 0 |
| reserve | 16 | 9.0× | 45 | +0.00 | 80% | 20% | −9 |
| reserve | 20 | 7.9× | 37 | **−1.00** | 38% | **62%** | −11 |

**The comparison is handicapped against the conclusion.** The dome's lateral edges
run at `m = 1` while its interior edges are several lanes wide, so a `ladder`
region carries **8-9× the constraint rows** of a `lateral` region with the same
number of cells. It is being asked to satisfy far more agreement conditions and it
ties anyway. That is the user's claim — *a simple, low-complexity thing should span
a region containing both very concrete and very abstract cells, because degrees of
abstraction are not a ladder* — surviving a test built to break it.

At `p = 8` the level-span histogram says the same thing directly: of 336
directions, 186 span 5 levels and 21 span 6 of the 7. The large regions are
cross-level regions.

**And the failure has a location.** At `p = 16` a fifth of matched pairs go
negative; at `p = 20` the median direction loses one, and 62% of pairs are worse
across levels than within one. Somewhere between `p = 12` and `p = 16`, abstraction
starts behaving like a ladder after all.

## 4. How the distribution moves with `p`

Every quantity moves the same way and they move together:

| p | trivial (`150p`) | earned | ball median | levels | frac size 1 | ladder−lateral |
|---|---|---|---|---|---|---|
| 0 | 54 | 861 | 150 | 7 | 0.08 | +0.00 |
| 4 | 600 | 323 | 150 | 7 | 0.09 | +0.00 |
| 8 | 1200 | 0 | 27 | 5 | 0.20 | +0.00 |
| 12 | 1800 | 0 | 7 | 3 | 0.30 | +0.00 |
| 16 | 2400 | 0 | 1 | 1 | 0.57 | +0.00 (80% at par) |
| 20 | 3000 | 0 | 1 | 1 | 0.67 | −1.00 (38% at par) |

**The ticket's hypothesis is confirmed: raising `p` fragments the regions while
inflating the floor.** The trivial part grows exactly `150p`; the median region
falls 150 → 27 → 7 → 1; the level span falls 7 → 5 → 3 → 1; and the share of
directions consistent nowhere but at home rises from 8% to 67%.

This is a cost of `p` the map has not priced. [B17](#565) found that `p` is a
geometric substitute for a missing mechanism — it shrinks the readable block until
subspaces indistinguishable from random have no room not to overlap. This reading
adds what that costs: **the same dial that raises composed rank destroys regional
coherence, and it destroys it fastest exactly where the rank reading looks best.**
At `p = 16`, which [B14](#561)/[B16](#564) read at ratio ≥ 0.988 and which
[B20](#569) found delivering 4.3 distinct directions at the apex, the median
direction on this reading is consistent over a single cell and spans one level.

## 5. Training moves none of it — and `restriction.py` says why

[B11](#555) is right that a construction reading is the wrong number to rule on, so
three arms were trained to 20k on `baseline`, seed 42, with the split taken at every
checkpoint and the full region filtration at 1000, 5000 and 20000.

**Every number is invariant. Not approximately — exactly, at every checkpoint.**

| arm | ticks | dim H⁰ | trivial | earned | dim H¹ | ball med (sealed) | levels | ladder−lateral |
|---|---|---|---|---|---|---|---|---|
| shipped | 0 → 20000 | 2323 | 914 | 1409 | 0 | 150 | 7 | +0.00 |
| reserve p=8 | 0 → 20000 | 1200 | 1200 | 0 | 269 | 30 | 5 | +0.00 |
| reserve p=16 | 0 → 20000 | 2400 | 2400 | 0 | 1423 | 1 | 1 | +0.00 |

Nine checkpoints per arm — 100, 200, 500, 1000, 2000, 5000, 10000, 20000 and
construction — and the split does not move by one dimension on any of the three.
The region distributions at 1000, 5000 and 20000 are identical to construction in
both row conventions, down to the censoring fractions and the at-par percentages.

**The body stalls by about 2,000 ticks on this rig**, so the later checkpoints are
taken against a motionless world and cannot be read as 20k ticks of *experience*.
That caveat does not reach this result: the invariance is already complete at 100,
200, 500 and 1000 ticks, all of them before the stall.

**This is not the instrument failing to look.** On the *identical* arm, condition
and seed, [B20](#569) reads per-chain composed ER moving 1.149 → 1.330 → 1.002 and
joint `d_eff` 1.655 → 2.202 → 1.029 across the same 20k ticks. The maps move a
great deal. What does not move is anything rank-derived.

**And the architecture predicts exactly this.** `restriction.py`'s docstring, quoted
in [B17](#565): the structural mask naming which node-stalk directions may
participate on an edge *"is set at construction, it closes and never re-opens"*,
and `project()` re-applies it after every transport step **specifically so a
learning rule cannot re-open it**. Every quantity in this reading — `k_v`, the
trivial block, `rank δ_R`, every trace dimension — is a property of *which*
directions each map may touch, not of what it does with them. The mask fixes all of
them at construction, and learning is confined to rotating inside it.

So the coherent structure of this graph is decided before a tick is run, and the
only dial that changes it is `p`. That is the same shape as [T3](#524)'s finding
that the full dome *"never had the rank to begin with"*, arriving from a completely
different direction, and it means a ruling on regional coherence can be taken at
construction without waiting for a trained surface — the one methodological point
where this reading disagrees with [B11](#555)'s caution, and it disagrees with a
mechanism rather than with a preference.

## What this reading does **not** measure

- **It reads the restriction maps only.** `Surface` is built from `sheaf.maps` and
  never touches `agent.sheaf.stalks` — [B17](#565)'s finding applies to this
  instrument exactly as it does to `composed_reads`, and is **not** repaired by it.
  Nothing here says whether traffic travels the directions a region retains. That
  is [B19](#568)'s object, not this one.
- **Maximal, never maximum.** Every region reported is one no single cell can be
  added to; none is the largest region that exists. `greedy` grows irregular
  regions as a check and is order-dependent, and the three families are three
  families, not the lattice of all regions.
- **`ladder` is capped at 40 cells** and `lateral` is bounded by its level's size
  (8-70 cells), so both are right-censored on most arms. Censoring is reported per
  arm and never pooled away, but it does mean the *death points* of those two
  families are lower bounds. The matched-size statistic is unaffected: it compares
  measured dimensions at equal size, not death points.
- **One seed (42), one condition (`baseline`), 16 seed cells per arm** (12 on the
  trained arms). No seed spread, and no second condition. Three arms trained, not
  eight — `p = 4`, `p = 12` and `p = 20` are construction-only, so the claim that
  training moves nothing is tested at `p ∈ {shipped, 8, 16}` and *argued* from
  `restriction.py`'s mask elsewhere.
- **The trained horizon is longer than the world.** The body stalls by ~2,000 ticks
  on this rig, so checkpoints at 5000, 10000 and 20000 run against a motionless
  world. The invariance is visible well before that and does not depend on them,
  but no claim here should be read as "20,000 ticks of experience change nothing".
- **Nothing about which directions.** The reading is entirely dimensional. Two
  cells whose regions are the same size may hold completely different directions,
  and whether regions *overlap in content* — the "partly-shared slice" half of the
  user's picture — is not measured here at all.
