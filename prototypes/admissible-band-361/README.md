# The admissible band: `|loop(c)|` from the mask, paired with dwell across seeds (#361)

[#226](https://github.com/NGL321/patchworks/issues/226) composed the chain

> `|loop(c)| <= tau_c < dwell_c`

and read its consequence — **`dwell_c > |loop(c)|`**, two graph quantities with no `tau` in
them — by pairing [#206](https://github.com/NGL321/patchworks/issues/206)'s
`final_cumulative_dwell` (seed 42, 100,000 ticks) against
[#274](https://github.com/NGL321/patchworks/issues/274)'s per-cell `tau_full` from **nine
different runs**, six of them 2,000-tick reads. On that pairing the median `dwell/tau` ran
0.861 to 3.468 and **seed 46 failed the bar outright**.

#226 labelled that a *sensitivity statement, not a measurement*, and opened this ticket to
close it. A verdict resting on one seed's dwell is the exact defect
[#208](https://github.com/NGL321/patchworks/issues/208) §3 exists to kill: *"an unstated
estimator let one measurement circulate as three different numbers across three tickets."*

This rig reads **dwell and `tau` off the same run**, at a horizon held fixed across seeds,
and enumerates `|loop(c)|` **per cell from the mask** rather than re-keying ADR-0026's
table by level.

## The three files

- **`loops.py`** — `|loop(c)|` per predicting cell, enumerated from the built graph.
  **The round trip already has a home and this is not it:**
  [#351](https://github.com/NGL321/patchworks/issues/351) built
  `benchmarks/loop_length.py`, which computes `2 · d(c, rim)` with the same rim and
  carries #343's cutoff hook. It is unmerged — it rides
  [PR #365](https://github.com/NGL321/patchworks/pull/365) — so `loops.py` **prefers that
  rig when it is importable** and falls back to an equivalent sweep until then;
  `check_against_loop_length` compares them whenever both exist. Read against the branch,
  the two agree **cell for cell** on `DEFAULT_SPEC`.

  What `loops.py` adds is the reading #351's rig deliberately declines — its own words,
  *"leaves the other to the ADR that checked it"* — the **vertex-disjoint genuine cycle**,
  which ADR-0026 checked at the apex alone and this reads at all 150 cells.
- **`read.py`** — one run per seed. Builds the `real` dome through
  `benchmarks/untrained_fixed_point.build`, runs `teaching` with both rules on, and at
  #206's checkpoint ladder reads `FoldRead.dwell` (the cumulative estimator #208 fixed)
  **and** both radii per cell, off the same live tick.
- **`summarise.py`** — the cross-seed roll-up. `--ticks` narrows to one horizon, because
  dwell is cumulative and mixing horizons reproduces #226's defect in a new place.

**No new instrument.** `spectra`, `tau_of` and both of #274's checks are imported from
`prototypes/driven-rho-274/read.py` by path; dwell is read off `FoldRead` unchanged. What
is new is the **join** and the **sweep**.

## Reproducing

    PYTHONPATH=src python prototypes/admissible-band-361/loops.py
    pwsh prototypes/admissible-band-361/sweep.ps1 -Ticks 30000
    PYTHONPATH=src python prototypes/admissible-band-361/summarise.py --ticks 30000

Nine seeds (42–50, #274's own, so the two readings sit beside each other) at **30,000
ticks each**, one torch thread per process, about 50 minutes wall clock for the nine in
parallel. Seeds 42–44 additionally to 100,000.

## 1. `|loop(c)|`, enumerated

`loops.py` reproduces **ADR-0026's published ladder exactly**, and agrees with #351's rig
cell for cell — 414 cells, 682 edges, a sensorimotor rim of 263 (256 patch, 3
proprioceptive, 3 touch, 1 actuator), the drive cell at its own distance 8, and
`|loop(c)| = 2 · d(c, rim)` running 2 at L1 to **14 at the apex**. #242's inherited 14
survives enumeration against the real mask, now from code rather than from a round number.

Two things the enumeration adds to what ADR-0026 checked:

- **`d(c, rim)` is exact within a level at every level**, not only at the apex.
- **ADR-0026 checked the vertex-disjoint genuine cycle at the apex alone.** Read at all 150
  cells, it is **longer** than the round trip at **17 cells** (3 at L1, 2 at L2, 12 at L3,
  by one tick) and **shorter at none**. A shorter genuine cycle would put a cell's true loop
  below what `2 · d(c, rim)` claims and *weaken* the bar; none exists, so the round trip is
  the conservative reading at every cell and the predicate does not turn on which is meant.

## 2. The pairing #226 made, reproduced exactly

Before replacing it, the old pairing is reproduced bit for bit from the artifacts on disk,
so the swap is like-for-like and the disagreement below is the sweep's and not a rig
difference:

| #226's pairing (seed-42 dwell at 100k vs #274's `tau`) | published | here |
|---|---|---|
| median `dwell/tau`, chart-only | 9.49 | **9.487** |
| clears `dwell > tau`, chart-only | 147/150 | **147/150** |
| median `dwell/tau`, full loop | 1.997 | **1.997** |
| clears `dwell > tau`, full loop | 93/150 | **93/150** |

On that same pairing the empty band is **38/150**.

## 3. The measurement: nine seeds, 30,000 ticks, both quantities from the same run

### The empty band — `dwell_c <= |loop(c)|`

| seed | median dwell | median `dwell/|loop|` | empty band |
|---|---|---|---|
| 42 | 8.06 | 2.055 | **39**/150 |
| 43 | 23.77 | 6.472 | 15/150 |
| 44 | 17.59 | 4.368 | 14/150 |
| 45 | 11.75 | 3.249 | 21/150 |
| 46 | 9.67 | 2.364 | 34/150 |
| 47 | 24.08 | 6.533 | **2**/150 |
| 48 | 12.22 | 3.475 | 27/150 |
| 49 | 20.62 | 5.443 | 13/150 |
| 50 | 25.53 | 6.411 | 10/150 |

**The empty-band count runs 2 to 39 of 150.** The per-cell reading is the one that matters:

> **0 of 150 cells have an empty band on every seed. 77 have one on at least one seed. 73
> on none.**

**No cell's band is structurally empty.** An empty band is a property of the *run*, not of
the wiring — which is what makes it a thing learning can move rather than a thing the
taper forecloses.

### The licence — `dwell_c > tau_c`

| seed | median `dwell/tau` | p05–p95 | licensed |
|---|---|---|---|
| 42 | 2.165 | 0.073–16.087 | 104/150 |
| 43 | 9.704 | 2.208–51.364 | 149/150 |
| 44 | 7.218 | 0.869–34.335 | 141/150 |
| 45 | 3.641 | 0.138–24.341 | 121/150 |
| 46 | **2.459** | 0.047–14.092 | 104/150 |
| 47 | 8.535 | 1.022–29.524 | 143/150 |
| 48 | 4.239 | 0.395–26.592 | 132/150 |
| 49 | 6.699 | 0.451–27.802 | 136/150 |
| 50 | 10.984 | 0.630–55.645 | 140/150 |

**Median `dwell/tau` runs 2.165 to 10.984, and every seed clears 1.** #226's sensitivity
statement gave 0.861 to 3.468 with **seed 46 failing outright**; read same-run, seed 46
gives **2.459**. That failure was an artifact of the mismatched pairing.

**Why the old pairing ran low, and it was not bad luck.** Seed 42's dwell is the *lowest of
the nine* (8.06 against 9.67–25.53). #226 held dwell at that seed's while varying `tau`, so
the numerator was pinned at the worst seed's value against every seed's denominator.

### Conduction — `tau_c >= |loop(c)|`, ADR-0026's ratio

Cells clearing it run **40 to 85 of 150**; the median `tau/|loop|` runs **0.642 to 1.155**.

**This is not a PASS on ADR-0026's predicate, and the count must not be read as one.** The
predicate is `max` over paths of `min` over the cells of a path, and every rim-to-apex path
contains an apex cell, where 0 to 1 of 8 clear on any seed (apex median `tau_full` 1.66 to
5.83 against `|loop| = 14`). The per-cell count is high only because `|loop(c)| = 2` at L1,
where 33–56 of 70 clear. **The bar is still short**, and it is short where the record has
always said it is short.

### What the operator correction does, in both directions

| | chart-only | full loop |
|---|---|---|
| licensed (`dwell > tau`) | 146–150 / 150 | 104–149 / 150 |
| conducts (`tau >= |loop|`) | **0–1** / 150 | 40–85 / 150 |

#274's correction makes the **licence harder** and **conduction easier**, because it raises
`tau`. Both readings move, in opposite directions, off one change of operator. Recorded
here because reporting only the half that tightens would misstate the correction.

### The whole chain held at once

`|loop(c)| <= tau_c < dwell_c` at both ends holds at **32 to 56 of 150 cells** across the
nine seeds — roughly a third, on every seed.

## 4. The same reading at #226's own horizon — seeds 42–44 to 100,000 ticks

30,000 is a horizon this rig chose. #226's numbers are at 100,000, so three seeds were run
there too, and **the character does not change — it strengthens**:

| seed | median dwell | empty band | median `dwell/tau` | licensed | conducts |
|---|---|---|---|---|---|
| 42 | 12.92 | 20/150 | 3.923 | 131/150 | 70/150 |
| 43 | 28.82 | 7/150 | 13.315 | 149/150 | 24/150 |
| 44 | 21.07 | 11/150 | 9.876 | 140/150 | 32/150 |

Empty band **7 to 20 of 150**, median `dwell/tau` **3.923 to 13.315**, and again **0 of 150
cells have an empty band on every seed** (28 on at least one, 122 on none).

**The like-for-like comparison with #226 is the point of this table.** Same seed, same
horizon, same dome and split — the only change is that both quantities now come from one
run:

| seed 42 at 100,000 ticks | #226's pairing | same-run |
|---|---|---|
| median `dwell/tau` | 1.997 | **3.923** |
| clears `dwell > tau` | 93/150 | **131/150** |
| empty band | 38/150 | **20/150** |

The mismatched pairing **understated the ratio by about 2x and overstated the empty band by
about 2x**, in the pessimistic direction on both.

## 5. The caveats that bind anything quoted from this

- **Every number here is a range over seeds**, never a median from one run. #274's binding
  caveat carries: `tau = -1/ln rho` is violently sensitive near `rho = 1`, so the spread is
  the honest object.
- **`tau = -1/ln rho` is not ADR-0026's `tau_hat`.** ADR-0026's quantity is the e-fold decay
  of a paired counterfactual deviation in private features; this is the cheap spectral
  stand-in, and #226's whole point is that the stand-in is licensed only where
  `dwell > tau`. The corrected number is a better stand-in and is still a stand-in.
  [#99](https://github.com/NGL321/patchworks/issues/99) owes the real reading.
- **A fixed seed does not fix the run, and the spread grows with the horizon.** Two
  independent runs at seed 42 agree exactly to tick 5,000 and diverge after. At 30,000 they
  give median `dwell/tau` of **2.165 and 1.958**, empty band **39 and 39**, licensed 104 and
  103, conducting cells 85 and 79 — ~10% on the ratio, zero on the count. At 100,000 the
  gap is wider: this rig reads seed 42's median dwell at **12.92** where #206's own run of
  the same seed, split, dome and horizon left **9.71** in `206-per-tick.npz`, a ~33%
  spread. So a single run is not a point, and it is least a point where the record quotes
  it. Consistent with #195's four runs at one seed giving four different binding cells.
- **The horizon is a choice, and the empty band moves with it.** Dwell is cumulative, so
  the empty-band count falls on every seed as the run lengthens (seed 47: 59 → 2 between
  ticks 100 and 30,000; seed 42: 129 → 39 → 20 at 100, 30,000 and 100,000). A count quoted
  without its horizon says nothing. **It does not fall without limit**, though: dwell is
  `ticks / (1 + crossings)` and the crossing rate settles, so the median flattens rather
  than growing with the run — seed 42's goes 3.7 at 5,000 to 12.9 at 100,000, a 3.5x rise
  over a 20x longer run. `dwell > |loop|` is a bar a long run can fail, not one any run
  eventually passes.
- **Per cell, never per level.** The by-level rows in the roll-up are a reporting axis;
  nothing is indexed on them (#181).

## 6. What this does not do

It reports whether bands are empty. **It does not nominate a remedy** — not
[#317](https://github.com/NGL321/patchworks/issues/317), not anything else. The map's
standing caution is that a measurement arriving with its mechanism already chosen
forecloses emergent properties nobody has predicted, and it applies here.

It does not re-open the operator question, settle the fidelity spread the map holds in
*Not yet specified*, or read ADR-0026's `tau_hat`.
