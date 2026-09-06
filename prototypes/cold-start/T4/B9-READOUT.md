# B9 (#551): `c = 1` in circuit — projection *and* gain, ungated, to horizon

Resolves [#551](https://github.com/NGL321/patchworks/issues/551), under
[the map](https://github.com/NGL321/patchworks/issues/532).

**Answer: no. `c = 1` does not hold its advantage in circuit — it inverts it.**
The advantage was an artifact of the instrument that produced it. In circuit,
`c = 1` collapses composed rim-to-apex rank harder than `c = 2` does, and it
does so by a mechanism that makes the sign of the effect structural rather than
incidental. The surface stays admissible in the ordinary sense; it is the rank
that dies.

## The arm

`c = 1` for all three of its readers — the projection cap
(`RestrictionMaps._push_apart`, via the `overlap_target` buffer),
`tick.reconciliation_gain`, and `bias_selection`'s fold-margin nomination —
applied before the agent is built so the buffer is registered under it too.
All three reach `c` through one call, `gain_denominators`, taking it from a
**keyword default**; `gauge_c.py::put_c_in_circuit` patches that default.
Rebinding `restriction.GAUGE_C` would *not* have worked — the defaults bind at
definition time, so the constant would have moved and every reader stayed on 2.

Verified rather than asserted, and the script raises if it fails to land:

| | `c = 2` | `c = 1` |
|---|---|---|
| `overlap_counts` histogram | `{1: 262, 2: 150, 3: 1, 8: 1}` | `{1: 412, 3: 1, 8: 1}` |
| gain ratio vs `c = 2` | — | **2.0 at 150 cells, 1.0 at 264** |

Both of `overlap_counts`' clamps hold — the pigeonhole floor and
[#228](https://github.com/NGL321/patchworks/issues/228)'s `c_v = deg(v)` on a
wholly-pinned incidence — so the actuator stays at 3 and the drive at 8. They
live inside `overlap_counts`, below the patch, so they were structurally out of
reach of this arm.

Ungated to 100k on both of T3's conditions, seed 42, on `f78377d` — the same
surface [#547](https://github.com/NGL321/patchworks/issues/547) read (its two
arm commits differ by no `src/` change). No instability check anywhere in the
loop, ruled by the user on B7 Q3.

**Surface stamps, honestly** (#455): the rig was committed while the runs were in
flight, so the baseline arm stamps `f78377d` and the winner arm `f099685`.
`git diff f78377d f099685 -- src/` is **empty** — the commit added only
`prototypes/cold-start/T4/` files — so both arms and both comparators ran the
same library. Neither run was dirty.

## The reading

Excess over one — the quantity the map is arguing about — at 100k:

| arm | `c = 1` **in circuit** | `c = 2` in circuit | `c = 1` **re-projected** |
|---|---:|---:|---:|
| baseline | **1.31e-09** | 2.60e-06 | 3.84e-03 |
| winner | **5.20e-10** | 1.50e-05 | 7.93e-03 |

- `c = 1` in circuit is **~1,990x (baseline) and ~28,800x (winner) worse** than
  `c = 2` — where the re-projection said it was ~1,474x and ~530x *better*.
- The re-projection overstated `c = 1` by **~2.9e6x** and **~1.5e7x**.
- **The tail is gone, not merely reduced.** #547 read p90 **1.111** and max
  chain **1.914**. In circuit, p90 and *max* both read **1.0000** (max
  1.000020 / 1.000026), against `c = 2`'s max chain 1.0209 and 1.6132.
- **The recovery shape is gone.** #547's `c = 1` fell to a floor at 10k and then
  climbed. In circuit the baseline decays monotonically and the winner does too;
  neither recovers.
- Training's erosion of the excess is **1.66e7x / 2.97e7x** under `c = 1`
  against `c = 2`'s 7.84e3x / 1.4e3x — roughly 2,000–20,000x worse.

## Why the sign is structural, not incidental

`c` is the **incoherence** cap: `_push_apart` pushes a cell's incident maps
*apart* whenever their summed Gram spectrum reaches `g_v² · c_v`. But
[#533](https://github.com/NGL321/patchworks/issues/533)'s mechanism makes a
hop's spectrum the **cosines of the principal angles between that relay cell's
two carried subspaces** — so alignment between incident maps is precisely the
material composed rank is made of. Pushing them apart destroys it. Measured at
100k:

| | `c = 1` | `c = 2` |
|---|---:|---:|
| leading cosine, median | **0.353 / 0.370** | 0.818 / 0.794 |
| all cosines, median | **0.082 / 0.074** | 0.348 / 0.362 |
| s₂/s₁, median | **2.5e-05 / 1.6e-05** | 1.1e-03 / 2.7e-03 |
| cells at cap | 28 / 29 | 21 / 7 |

So **tightening `c` can only ever lower composed rank**, and loosening it does
nothing because [B1](https://github.com/NGL321/patchworks/issues/537) already
found the constraint slack at the median (0.2500 at every checkpoint of both
arms, here too). `c` is not a lever in either direction: slack when loosened,
actively harmful when tightened. B1's "`c` is not a lever at all" survives this
ticket — and is now shown on a surface where `c` was actually in circuit, which
is the one way it had not been shown.

## What else moved

The doubled gain reaches the tick, so the arm read more than one number.

- **Stability: admissible.** Zero non-finite state, error or maps in either arm
  at every checkpoint. Map Frobenius norms stay in band `[1.0, 2.0]`, per-edge
  map effective rank sits at full width (3.0000 interior, 4.0000 motor rim), and
  `h` norms are comparable to `c = 2`. The ungated arm was worth running and it
  did not destabilise: **`c = 1` is admissible and useless**, not inadmissible.

- **Disagreement energy falls, prediction error rises.** #547's arms never
  recorded edge reads, so `c = 2` was re-run on the same rig at 5k to have
  something to difference against (`551-cap-breach-seed42-5000.json`). The
  doubled gain reconciles harder: per-stratum disagreement energy drops to
  **0.30–0.99x** on most strata in both arms. It is paid for twice — the
  deepest stratum *spikes* **4.0x** (baseline L6-L6) and **4.1x** (winner
  L7-L7), and median prediction error roughly doubles, 0.0077 → 0.0135
  (baseline) and 0.0136 → 0.0228 (winner).

- **The `c` cap and #502's pin re-derivation interact badly — but ADR-0010's
  bound is not broken.** The winner arm reads a cap ratio of **1.107–1.120**
  from 5k ticks on, and the baseline never does. Named rather than inferred: the
  offending cells are the **drive-side apex cells** (406, 410, 411, 412; deg 5,
  drive + interior incidence), and the cause is that
  `T1/run.py::pin_drive_edges` re-derives `overlap_target = g_v²·c_v −
  pinned_count`, per #502. That subtracts an **absolute** count from a budget
  that `c` **scales**, so halving `c` takes the residual budget for the held
  maps from `8 − 1 = 7` to `4 − 1 = 3` — a 2.33x tightening at those cells,
  where `_push_apart` can no longer fit the held subset.

  **This is not an admissibility failure.** The reconciliation gain divides by
  `gain_denominators` = `g_v²·c_v` = 4, and the measured `λ_max` there is
  **3.14–3.34**, i.e. `peak / (g_v²·c_v)` = **0.774–0.836**. The bound the gain
  assumes holds with 16–23% slack, and #220's pairing is intact. What is
  breached is #502's stricter internal bookkeeping, not the ADR. The remaining
  rows at exactly 1.0000 are cells sitting *at* the cap, not over it.

  It is still worth recording as a **latent scaling bug in #502's
  re-derivation**, invisible at `c = 2` and only exposed by a small `c`: an
  absolute subtraction from a scaled budget has no guard against going
  non-positive, and a pinned map on a `c_v = 1` boundary cell would take it
  to zero.

## What this settles for the map

- **`c` is finished business, in both directions.**
  [B1](https://github.com/NGL321/patchworks/issues/537) found it slack when
  loosened; this ticket finds it *harmful* when tightened — in circuit, at
  horizon, on both arms. `GAUGE_C = 2` stands and
  [B10](https://github.com/NGL321/patchworks/issues/552)'s mark is the last word
  on `c`. Nothing further should be spent on it.
- **The `c = 1` signal on the map's index is withdrawn.** #547's 1.0038 / 1.0079
  were the re-projection, wrong by ~2.9e6x and ~1.5e7x. B7's Decisions-so-far
  line should not be read as leaving a live lever.
- **The counterexample probe fails, so
  [B6](https://github.com/NGL321/patchworks/issues/546)'s fog is not relieved.**
  The map recorded B9 as a probe on whether training's erosion is universal —
  `c = 1` being the one setting whose excess recovered with horizon. In circuit
  it does not recover, and its erosion is **1.66e7x / 2.97e7x** against
  `c = 2`'s 7.84e3x / 1.4e3x. So the erosion is universal on everything measured
  so far, and *more* incoherence makes it worse. Whether it is a fixed fraction
  or a fixed amount stays B6's question, unrelieved.
- **A sign constraint for B6 and B2, free.** If composed rank is the product of
  principal-angle cosines, any lever that pushes a cell's incident maps apart
  lowers it. That is a constraint on the whole remaining stack: the per-edge
  `m_e` / lateral / invariant levers must buy their rank *without* raising
  incoherence, and it is worth checking they do not.

## Files

- `gauge_c.py` — the arm.
- `551-c1-baseline-seed42-100000.json`, `551-c1-winner-seed42-100000.json` — the runs.
- `b9_table.py` — the three-column table.
- `b9_cap_breach.py`, `551-cap-breach-seed42-5000.json` — the cap cells and the
  matched `c = 2` edge read.
- `537-baseline-seed42-100000.json`, `537-winner-seed42-100000.json` — #547's
  comparators, committed here for the first time.
