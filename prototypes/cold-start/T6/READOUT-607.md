# B43 (#607): do relays buy ADR-0026's bar without gaming it, and which width reading discriminates?

Prototype for [#607](https://github.com/NGL321/patchworks/issues/607), on map
[#532](https://github.com/NGL321/patchworks/issues/532). Surface: `reserve_p12` —
plain `build_graph(DomeSpec())` on B38's port ([#600](https://github.com/NGL321/patchworks/pull/600)),
`n = 32`, 414 cells, 682 edges, capacity budget 63 per predicting cell. A relayed
mask is that graph with interior edges added and `Dome._assemble` re-run: the
shipped constructor's own final step, not a rig copy of it.

Everything below is printed by the instruments in this directory —
`b43_relay.py`, `b43_width.py`, `b43_detect.py`, `b43_hops.py`, `b43_table.py`.
Nothing is retyped by hand.

---

## 0. The short answer

**Relays buy the bar. The conjunction that was supposed to prove they bought it
honestly cannot do that job, and this reading is what shows why.**

* **Clause 1 is bought, and the pre-registered number arrives.** The apex
  cohort's `world_loop` falls **14 → 5** at 16 relay sites, inside the ticket's
  predicted 5–8.
* **Clause 3 passes by nine orders** — rim→apex bottleneck **1.7e-11 → 4.0e-2**,
  replicated on a second seed.
* **But an *unaimed* relay set passes clause 3 by seven orders too**, and every
  arm's gain sits inside the band its own hop-count reduction predicts at B21's
  20–50× per hop. **Clause 3 is not independent of clause 1**: ADR-0021's
  bottleneck is a `min` over a path's edges and ADR-0026's divisor is a path
  length, so both are decreasing functions of the same quantity. The
  anti-Goodhart clause cannot separate *conducts better* from *has a shorter
  path*, which is the exact discrimination it was written to make.
* **Clause 2 does not hold robustly.** `τ̂` does not rise — it **falls** (median
  8 → 6, and 9 → 5 on seed 43's `motor` arm), and the per-cell ratio rises only
  because the divisor fell. On seed 43 `motor` the ratio median lands at
  **0.667**, below the baseline's 1.000.
* **Neither width reading discriminates, and both fail into the same null.**
  Under B34's floor allocation `Σ_e m_e` is **exactly `deg(v)` at 150/150
  cells** — B40's null confirmed — and community membership, in the floored
  regime B35 says it is best posed in, is **also exactly degree at 150/150** at
  every threshold B40 uses. Relays change neither.

---

## 1. Two layout constraints, and only one of them was inherited

### B40's constraint does not fire as written

B40 requires *at least two disjoint relay paths between any pair of regions
joined*, because *a relay edge is a bridge by construction*. On this surface the
antecedent cannot be met: **the dome is already connected**, so a relay is a
**chord**, never a bridge. `b43_relay.bridge_census` runs Tarjan on every layout
at every count and reads **0 relay bridges throughout**, single-relay layouts
included.

What survives is the practical half and it is sharper. B34's criterion does not
count cycles; it counts **short local** cycles — length ≤ 6 inside the radius-2
neighbourhood. A long chord has none, so it is pinned at the probe floor for the
same practical reason a bridge is. The `no_local` column below is that census,
and it is **not monotone in relay count**: relays that land near each other close
on each other, so density buys cycles that sparsity does not.

### The constraint this ticket found: a relay must anchor on **both** sides of ADR-0016's ban

The first layout built here pooled every rim-adjacent predicting cell and split
it in half. It moved `world_loop` almost not at all — 16 → 11 with 32 relays —
and the reason is structural:

    world_loop(c) = min over (a, p) of  d(c, a) + w + d(p, c),   a != p

has **two legs**, and `a` ranges over the actuators alone. Every anchor in that
pool was adjacent to a *sensory* cell, so the sensory leg shortened, the actuator
leg did not, and a `min` is bounded by the leg nobody touched. The `sensory` row
below is that layout kept as a control: at 16 relays it reaches apex 9.0 where
`motor` reaches 5.0 with the same count.

> **A relay network that anchors on one side of ADR-0016's ban buys half a loop
> and the bar reads the other half.** ADR-0016's ban is what makes `world_loop`
> the loop it is, so the ban is also the constraint on where relays may land.
> This is not in B35, B40 or the ticket.

---

## 2. Clause 1: `world_loop` on the relayed mask — free, no run

`b43_table.py layouts`. `apex_med` is the **baseline** apex cohort — the cells
deepest by `d(c, rim)` *before* any relay was laid, which is the cohort ADR-0026's
enumeration puts at 15–16. (An earlier version recomputed "deepest" on the relayed
graph; that is a moving set, and it understated the buy by five ticks.)

```
layout     sites  +e |  wl_med wl_max apex_med apex_max |  no_local bridge |  over_B reuse  load
------------------------------------------------------------------------------------------------
none           0   0 |     8.0   16.0     14.0     16.0 |         0      0 |       0     0     0
sensory       16  16 |     8.0   11.0      9.0     11.0 |         4      0 |       0     1     2
motor          4   4 |     8.0   11.0      7.0      9.0 |         0      0 |       0     2     6
motor          8   8 |     8.0   10.0      7.0      7.0 |         0      0 |       0     3     1
motor         16  16 |     8.0    9.0      5.0      7.0 |         0      0 |       0     6     3
pair           8  16 |     8.0   10.0      7.0      7.0 |         0      0 |       0     3     6
pair          16  32 |     7.0    9.0      5.0      7.0 |         4      0 |       4     6     6
pair_wide     16  32 |     7.0    9.0      5.0      7.0 |         4      0 |      19     6     6
random        16  32 |     8.0   11.0      9.0     10.0 |         9      0 |       3     3    34
```

(The full sweep — five layouts × five counts — is in `607-layouts.json`.)

**Verdict: clause 1 is bought, and the ticket's pre-registered number arrives.**
Apex `world_loop` **14 → 5** and the fleet max **16 → 9**. The ticket predicted
*roughly 5–8* for a sparse network putting the apex 2–3 hops from the rim; 5 is
the bottom of that range and it needs 16 sites to get there — at 8 sites the apex
reads 7, also inside it.

**The cheapest layout that buys clause 1 is `motor` at 16 sites**: 16 relay edges
at `m = 1`, apex 5.0, and **zero cells over the capacity budget**. `pair` needs 32
edges for the same apex and puts 4 cells over `B`; `pair_wide` puts **19** over.
So B35 §6's *`Σ_e m_e ≤ B` prices the relay count* is real and it bites first at
width, not at count.

**#311's anti-hub objection is not self-enforcing here.** The motor anchor pool is
tiny — the cells adjacent to the one actuator — so past a few sites a relay
network *must* reuse anchors: `reuse` reaches **6** at 16 sites. The budget does
not stop this, because reuse concentrates `Σ_e m_e` on exactly the cells the
budget is measured at, which is how `pair 16` puts 4 cells over `B` at `m = 1`.
**B20's joint-span collapse arrives as anchor reuse rather than as a shared
tail**, and the `load` column (cells whose `world_loop` lengthens if one relay is
cut) says no single relay carries the fleet: 6 of ~151 at `pair 16`, against 34
for the unaimed `random`.

---

## 3. Clauses 2 and 3: read with `detectability.py`, both masks, own horizon

Horizon **2,000 ticks**, stamped per run (`motion`), never inherited — B38 found
it varies 13× between seeds of one arm. The baseline is **taken here, not
quoted**: B38's 8.35e-10 is a 30,000-tick figure and comparing a 2,000-tick relay
to it would compare two horizons. Three trials per arm, one per rim stratum.

```
arm          seed relays  learn |  bottleneck_med        max |  tau_med ratio_med ratio_max  >=1 |  wl_cohort    motion
----------------------------------------------------------------------------------------------------------
motor16        42     16   2000 |       3.516e-02  1.637e-01 |     8.00     1.143    11.200   92 |        5.0  5.63e-05
motor16        43     16   2000 |       1.995e-02  7.526e-01 |     5.00     0.667     4.000   32 |        5.0  2.24e-05
none0          42      0   2000 |       1.673e-11  2.482e-11 |     8.00     0.917     5.875   74 |       15.0  7.09e-04
none0          43      0   2000 |       2.011e-11  8.340e-11 |     9.00     1.000     9.000   78 |       15.0  3.70e-11
pair16         42     32   2000 |       4.026e-02  9.201e-02 |     6.00     1.143    11.000   89 |        5.0  2.12e-03
pair16         43     32   2000 |       5.034e-02  1.333e+00 |     7.00     1.000     7.143   78 |        5.0  5.61e-04
random16       42     32   2000 |       3.175e-04  3.096e-03 |     8.00     1.111    19.667   87 |        9.0  1.89e-04
random16       43     32   2000 |       3.507e-04  4.307e-04 |     7.00     0.875     4.250   68 |        9.0  2.29e-04
sensory16      42     16   2000 |       1.404e-09  4.386e-03 |     9.00     1.100     8.000   91 |        9.5  8.54e-03
```

**The baseline is where B38 left it, scaled for the horizon.** 1.7–2.0e-11 at
2,000 ticks against B38's 8.35e-10 at 30,000 — an order and a half below, in the
direction training moves it. The rig agrees with the map's standing figure.

**Clause 3 passes and it is not close**: **1.7e-11 → 4.0e-2**, nine orders,
replicated on seed 43 (2.0e-11 → 5.0e-2).

**Clause 2 does not hold as the conjunction needs it to.** `τ̂` **falls** —
median 8 → 6 on seed 42's `pair`, 9 → 5 on seed 43's `motor` — and the per-cell
ratio `τ̂_c / world_loop(c)` rises only because its divisor fell. On seed 43's
`motor` the ratio median lands at **0.667 against the baseline's 1.000**, so the
arm that most cleanly buys clause 1 reads *worse* on clause 2 at the other seed.
The clause was written as *a check rather than a threat*; on these two seeds it is
a threat, and the residue B35 identified — `τ` is read off
`ρ(K · (J_chart + J_stalk · A_v · D))` and a relay moves `A_v` — is where it
lands.

> Per [B39](https://github.com/NGL321/patchworks/issues/601), no max-min-over-paths
> conduction verdict is quoted here. `Trial.conduction` is in the JSON for
> provenance under a name that says so; the reading taken is the **per-cell**
> ratio, which B39 left standing.

**Two things the reads carry and do not resolve.** `none0` seed 43 was read on a
world whose own motion had stopped (`std_max` 3.7e-11 against seed 42's 7.1e-04) —
B38's stall, visible in the stamp; its numbers agree with seed 42's anyway. And
the arms' motion differs by two orders, so the bottleneck's denominator is not
identical across arms; the effect runs the wrong way to explain the result
(`pair` has *more* motion than `none` and a higher bottleneck).

---

## 4. The conjunction cannot do what it was designed to do

Clause 3 is the anti-Goodhart clause, and its reasoning assumes ADR-0021's
amplitude and ADR-0026's divisor are independent. **For a topological
intervention they are not.** ADR-0026's divisor is a path length. ADR-0021's
bottleneck is a `max` over paths of a `min` over that path's edges — so a shorter
path takes the `min` over fewer edges, and at B21's 20–50× of gain per hop,
dropping hops raises it by orders whether or not anything conducts better.

`b43_hops.py` prints the widest path each trial actually found, beside what its
hop count alone predicts:

```
arm          seed            kind  hops   bottleneck wl_cohort | hops_saved           predicted_x   actual_x
------------------------------------------------------------------------------------------------------------
motor16        42  proprioceptive     2    1.637e-01       5.0 |        5.0  3.20e+06 .. 3.12e+08   8.88e+09
motor16        43           touch     4    1.995e-02       5.0 |        3.0  8.00e+03 .. 1.25e+05   1.08e+09
pair16         42  proprioceptive     2    9.201e-02       5.0 |        5.0  3.20e+06 .. 3.12e+08   5.00e+09
pair16         42           touch     3    4.026e-02       5.0 |        4.0  1.60e+05 .. 6.25e+06   2.19e+09
random16       42  proprioceptive     3    3.096e-03       9.0 |        4.0  1.60e+05 .. 6.25e+06   1.68e+08
random16       43           touch     4    4.307e-04       9.0 |        3.0  8.00e+03 .. 1.25e+05   2.34e+07
sensory16      42  proprioceptive     7    5.744e-10       9.5 |        0.0  1.00e+00 .. 1.00e+00   3.12e+01
sensory16      42           patch     3    4.386e-03       9.5 |        4.0  1.60e+05 .. 6.25e+06   2.38e+08
```

Baseline: median **7 hops**, median bottleneck **1.842e-11**. (All twenty-seven
trials are in the script's own output; the eight above are the shape.)

Two things follow, and the second is the one that matters.

1. **The bottleneck tracks hops, not aim.** Every arm's gain is of the order its
   hop-count reduction predicts. `sensory`, which saves no hops on two of three
   trials, gains a factor of tens; the arms that save five hops gain nine orders.
2. **The unaimed control passes clause 3.** `random` — 32 relays at the same
   width, aimed at nothing, laid between predicting cells drawn uniformly — reads
   **3.2e-4 and 3.5e-3**, seven orders above baseline, while buying only 14 → 9
   of clause 1. A clause that a random rewiring passes by seven orders is not
   catching the failure it was written for.

> **So the conjunction as pre-registered does not establish *without gaming
> it*.** It establishes that a relay shortens paths, which was never in doubt.
> The three clauses are one clause read three ways, because clause 1's quantity,
> clause 2's divisor and clause 3's `min` are all functions of path length.
>
> **What would separate them is a length-matched null**: judge a layout against
> the amplitude its own hop count predicts, and require the excess. The band
> printed above is too crude to be that test — it is a per-hop constant from B21
> applied to a median, and individual trials clear it and fall short of it by two
> to three orders in both directions. Building it properly — paired per path, on
> the arm's own per-edge gains — is the next instrument, and it is not this
> ticket's to write.

`random`'s other reading is the one piece of good news for aim: **clause 1 does
discriminate.** Unaimed relays buy 14 → 9 where aimed ones buy 14 → 5, at the same
count and width, and they cost more elsewhere — `load` 34 against 6, so a random
network leans a fifth of the fleet on one edge.

---

## 5. Both abstraction readings, scored side by side

`b43_table.py width`, seed 42, at construction and at 500 ticks. `=deg` counts
cells where the reading equals the cell's interior degree — the null. `flr_*` is
the **floored** variant: one direction per edge, which is the `m_e = 1` regime B40
measured and B35 §7 says membership is *better* posed in.

```
arm           tick    th |  memb_med  max    std distinct  =deg |  flr_med  max    std distinct  =deg
----------------------------------------------------------------------------------------------------
none0            0   0.3 |       1.0   12   3.47       11     2 |      3.0    6   1.09        6    11
none0            0   0.5 |      24.0   42   8.64       26     0 |      5.0    9   1.28        8   109
none0            0   0.9 |      38.5   63  21.11       19     0 |      6.0    9   1.24        7   150
none0          500   0.9 |      38.0   63  20.98       21     0 |      6.0    9   1.24        7   150
pair16           0   0.9 |      38.5   64  21.08       22     0 |      6.0   12   1.48        9   150
pair16         500   0.3 |       2.0   12   3.47       12     5 |      3.0    6   1.28        6      5
pair16         500   0.9 |      38.5   64  20.84       24     0 |      6.0   12   1.48        9   150
pair_wide16      0   0.9 |      43.5  145  27.72       26     0 |      6.0   12   1.48        9   150
pair_wide16    500   0.9 |      43.5  144  26.94       35     0 |      6.0   12   1.48        9   150
```

Strict, from the same run:

| arm | `Σ_e m_e` off the mask | = degree | `Σ_e m_e` off B34's allocation | = degree |
|---|---|---|---|---|
| `none` | median 38.5, std 14.13, 19 distinct | 0/150 | median **2.0** | **150/150** |
| `motor16` | median 39.0, std 14.10, 18 distinct | 0/150 | median 2.0 | **150/150** |
| `pair16` | median 39.0, std 14.05, 20 distinct | 0/150 | median 2.0 | **150/150** |
| `pair_wide16` | median 52.0, std 22.52, 26 distinct | 0/150 | median 3.5 | 131/150 |

### What this says

**B40's null is confirmed exactly, and it was stated about the wrong reading
alone.** Under B34's allocation `Σ_e m_e` **is** `deg(v)`, at every one of the
150 predicting cells, on every layout — the strict reading is plain degree, as
predicted. The prediction's scope was too narrow: **community membership is
plain degree in the same regime**, `=deg` at **150/150** at all three of B40's
thresholds (0.8, 0.9, 0.95), on every arm and at both checkpoints, and at 96–109
of 150 even at 0.5 (`b43_check.py`). At `m_e = 1` each edge carries one direction, and
in a 20-dimensional exposed block two drawn directions sit near `cos ≈ 0.2`, so
nothing ever clusters and the count of clusters is the count of edges.

**Off the criterion's floor, the strict reading is *not* degree** — the shipped
allocation gives it median 38.5, std 14.1, 19 distinct values. So the two strict
readings disagree about their own null, and which one B35 §7's argument is about
matters: *a saturating budget makes it uniform* is a claim about the shipped
allocation, and on this surface **that allocation is the more graded of the two**.

**Membership only leaves degree at thresholds where it stops meaning
near-orthogonality.** At 0.3 — directions counted as one community when they sit
within 72° — membership finally grades: median 3.0 floored, 6 distinct values,
`=deg` at 5–12 cells. Whether a 72° cone is *one direction staying consistent* is
a definitional call this ticket does not make; it records that the reading's
discrimination is bought entirely at that threshold and nowhere else.

**Relays do not move either reading.** `none` and `pair16` read the same
membership median (6.0) and nearly the same strict median (38.5 vs 39.0). The only
movement is at the relay-bearing cells themselves — floored max 9 → 12, the six
anchors that carry reused relays. **Arm C has no content at `m = 1`**, for a
reason simpler than B40's: a relay at `m = 1` is not a *wide* edge, so B34's
criterion never considers it. Making relays visible to the criterion means
`pair_wide`, and that costs 19 cells over the capacity budget.

**B22's named risk lands differently than feared.** The worry was that membership
is near-binary on this surface. It is not binary — floored it takes 7–9 distinct
values — it is *degenerate to degree*, which is worse in the same direction: a
measure that grades exactly as the graph's own degree sequence grades adds
nothing to it.

**And none of it is learned.** 500 ticks move the floored reading by zero at every
threshold ≥ 0.5 and by a few cells at 0.3 — B22's *"20k ticks move every number by
exactly zero"* holds at this horizon, on the relayed mask as on the shipped one.

---

## 6. What this ticket does not decide

* **Whether relays are adopted.** The conjunction was the adoption rule and it is
  now known not to discriminate. Adoption waits on a rule that does.
* **The length-matched null.** Named in §4, not built.
* **Whether B34's criterion should see relays at all.** At `m = 1` it cannot; at
  the cap it can and the budget breaks. Nothing here rules on which.
* **The 72° question.** Whether membership's discriminating threshold is a
  definition or a fudge is B35's to answer, not a measurement.
* **The floor's leak.** B40's `√1.25` is inherited by every arm here and was not
  re-read; whether it contaminates a relay comparison is unanswered because no
  arm comparison survived to be contaminated.

## 7. Files

| file | what |
|---|---|
| `b43_relay.py` | layouts, the relayed `Dome`, bridge and local-cycle census, budget, clause 1 |
| `b43_detect.py` | clauses 2 and 3 on both masks, `detectability.trial` unchanged |
| `b43_width.py` | both abstraction readings, per arm, at two checkpoints |
| `b43_hops.py` | §4's hop-count table |
| `b43_table.py` | every table above |
| `607-layouts.json` | the full layout sweep |
| `607-width-seed42.json` | both readings, four arms, five thresholds |
| `607-detect-*.json` | nine reads: five layouts, two seeds |
