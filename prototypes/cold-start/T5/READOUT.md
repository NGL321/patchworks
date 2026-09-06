# T5 (#546): does training's erosion of composed rank scale with lane width?

Reading for [#546](https://github.com/NGL321/patchworks/issues/546), under
[the map](https://github.com/NGL321/patchworks/issues/532). Surface: `cs-537` at
`f78377d`, which is `main` `2bce07d` plus [B1](https://github.com/NGL321/patchworks/issues/537)'s
T4 rig. Baseline arm, full dome, seed 42 throughout, seed 43 where stated.

## The answer

**Neither hypothesis #546 pre-registered. The erosion is not a fixed fraction of
the excess over one, not a fixed amount, and not a floor — it weakens sharply
with lane width and then stops.**

Composed rim-to-apex effective rank in excess of one, at construction and at the
horizon each arm reached, on domes genuinely **rebuilt** at each width:

| `interior_m` | construction | horizon | ticks | erosion | excess retained |
|---|---|---|---|---|---|
| 3 | 2.545e-2 | 2.602e-6 | 100k | **×9783** | 0.010% |
| 3 | 2.545e-2 | 5.826e-5 | 20k | ×437 | 0.23% |
| 6 | 2.027e-1 | 7.531e-4 | 20k | ×269 | 0.37% |
| 10 | 4.115e-1 | 2.285e-2 | 20k | ×18.0 | 5.6% |
| 14 | 6.158e-1 | 1.490e-1 | 20k | ×4.1 | 24.2% |
| 14 | 6.158e-1 | 1.379e-1 | 100k | **×4.5** | 22.4% |

The m = 3 rows are #537's arms, not new ones, and the 20k row's ×437 is the
**~440× #546 quotes in its own statement of the problem** — which is the check
that this rig is reading the same quantity the ticket is asking about.

Each hypothesis is falsified on its own terms:

- **Multiplicative** predicts one erosion factor at every width. The factor moves
  **×437 → ×4.1** across the four widths at a common 20k horizon — a 107-fold
  spread on the quantity that was supposed to be constant.
- **A fixed amount** predicts one value for `excess(0) − excess(T)`. It reads
  0.0254, 0.2019, 0.3886, 0.4779 — it simply tracks the construction excess, to
  three significant figures in every case.
- **A floor** predicts one endpoint. The endpoints span 2.6e-6 to 1.4e-1, a
  factor of 53,000.

## The confound this could have died on, and why it did not

At 20k the m = 3 arm was still falling steeply, so a wide arm that merely eroded
*slower* would have produced this table and then caught up. That is the reading
#546 item 2 asks for and it was taken: **the m = 14 arm was carried to the same
100k horizon as m = 3.** It does not catch up. It does not move at all:

| ticks | 20k | 30k | 100k |
|---|---|---|---|
| m = 3 excess | 3.344e-5 | 2.634e-5 | 2.602e-6 |
| m = 14 excess | 1.555e-1 | 1.714e-1 | 1.379e-1 |

Both rows are read from the 100k arm at that width, so each is a within-run
comparison and none of the cross-arm spread below enters it. Over the span in
which the narrow arm loses a further **12.8×**, the wide arm moves by 1.13× and
not downward throughout — 1.555e-1, up to 1.714e-1, back to 1.379e-1. The wide
lane erodes **less**, not slower.

**Replication, and the spread it exposes.** Seed 43 was run at both widest widths
and reproduces the effect: ×21.4 at m = 10 (against seed 42's ×18.0) and ×2.6 at
m = 14 (against ×4.1).

The run-to-run spread is **not** uniformly small, and the table above is quoted
so as not to hide it. Two arms of the same width and seed, differing only in
where they stop, disagree at their shared 20k checkpoint: at m = 14 by 4%
(1.490e-1 standalone against 1.555e-1 in the 100k arm), but at m = 3 by **1.7×**
(5.826e-5 against 3.344e-5). The narrow arm is the noisier one precisely because
it is still falling steeply at 20k, which is the same fact the horizon check
below turns on.

Every claim here is therefore made on margins far larger than that spread: the
erosion factor moves by 107× across the widths, against a worst-case
reproducibility wobble of 1.7×.

## The mechanism, and it is visible (#546 item 3)

[B1](https://github.com/NGL321/patchworks/issues/537) established that composed
rank 1.000 is **domination, not annihilation**. So what training must do to
collapse the composite is pull the leading per-hop principal-angle cosine away
from the rest — and how much room it has to do that in is fixed at construction.
That room is what moves with width:

| `interior_m` | lead−second gap at construction | at horizon | opened | erosion |
|---|---|---|---|---|
| 3 | 0.2394 | 0.4663 | **+0.2269** | ×9783 |
| 6 | 0.1360 | 0.2743 | +0.1383 | ×269 |
| 10 | 0.0900 | 0.1217 | +0.0317 | ×18.0 |
| 14 | 0.0407 | 0.0445 | **+0.0038** | ×4.5 |

The gap training manages to open falls monotonically with width and tracks the
erosion factor across three orders of magnitude. The whole cosine distribution
shifts by a near-constant `+0.034 … +0.041` at every width, so this is not
training doing less work at wide lanes — it is training doing the same work with
no headroom to concentrate it. At m = 14 the leading cosine is already 0.954 at
construction; it reaches 0.977 and the composite survives.

This is the same event [B1](https://github.com/NGL321/patchworks/issues/537) and
[B3](https://github.com/NGL321/patchworks/issues/538) saw through two instruments
— training raising alignment while rank falls — now with the knob that governs it
identified.

## Two things the arms say that the ticket did not ask

**Uniform `interior_m` saturates well below the bar, and below #540's number.**
Rebuilt construction ER reads 1.0255 (m=3), 1.0988 (4), 1.1491 (5), 1.2027 (6),
1.2650 (8), 1.4115 (10), 1.4565 (12), 1.6158 (14), 1.5832 (16), 1.7772 (20). It
does not reach [#540](https://github.com/NGL321/patchworks/issues/540)'s 2.193,
because `k_v = min(n, sum_e m_e)` caps at `n = 32` while the chain stays seven
hops — widening one global constant widens the ambient along with the lane. So
these arms read the **trend** and cannot stand in for #540's per-edge stack,
whose own construction value has to be built to be trained.

**`c = 1`'s advantage does not scale with width (#546 item 5) — and it was never
real.** [B9](https://github.com/NGL321/patchworks/issues/551) resolved while these
arms were running and supersedes this reading: every `c = 1` figure below comes
from `angles.py::sweep_c`, the **post-hoc re-projection** B9 was opened to test,
and B9 found it overstates `c = 1` by ~2.9e6× (baseline) and ~1.5e7× (winner). In
circuit `c = 1` is ~1,990× and ~28,800× *worse* than `c = 2`, with no tail and no
recovery shape. So what the width series below measures is **how the artifact's
size varies with width**, not how `c` does. It is kept because it points the same
way B9 does and because the mechanism agrees — B9 finds in-circuit `c = 1` drives
the leading per-hop cosine to 0.353/0.370 against `c = 2`'s 0.818/0.794, which is
the same quantity the mechanism section above finds governs the erosion.
`GAUGE_C = 2` stands; `c` is finished business. The rest of this paragraph is the
pre-B9 reading, left for the record:
Against `c ≥ 2` at 100k it is **1474×** at m = 3 (3.835e-3 against 2.602e-6) and
**1.32×** at m = 14 (1.819e-1 against 1.379e-1). At 20k it is 6.4× at m = 3 and
1.00× at m = 14. It remains the only setting whose advantage grows with horizon
at *both* widths; but it is a narrow-lane effect that width supersedes rather
than a rule-level term that resists the erosion. On #546's own framing, `c = 1`
starts from a different place rather than resisting — and after B9, it does
neither, because the quantity was an instrument artifact.

## What this does to the map

The premise on which [#540](https://github.com/NGL321/patchworks/issues/540)'s
whole stack was made provisional — "if the erosion is a fraction, every
parameterisation lever on #532 fails and the transport **rule** is what must
change" — **is false.** The erosion is not a fraction. Widening the lanes buys
excess that training does not take back, and the fog entry proposing R-A's
residual `h ← (I+T)h` as the forced move is not forced by this evidence.

What the arms do *not* establish is that width alone reaches the bar. The best
trained composed ER anywhere on the record is now **1.186** (m = 14, seed 43,
20k), against a bar of generic median ≥ 2.0, and uniform width saturates before
it gets there. #540's per-edge stack is the live candidate, and what it needs is
its own construction surface built and trained, not an extrapolation from these.

## Reproducing

```
PYTHONPATH=src python prototypes/cold-start/T5/width.py --widths 3 4 5 6 8 10
PYTHONPATH=src python prototypes/cold-start/T5/trained_width.py --interior-m 6 10 14 --ticks 20000
PYTHONPATH=src python prototypes/cold-start/T5/trained_width.py --interior-m 14 --ticks 100000
python prototypes/cold-start/T5/analyse.py
python prototypes/cold-start/T5/mechanism.py
```

Each arm is about 25 minutes to 20k and 2h15 to 100k on this box, written at
every checkpoint.
