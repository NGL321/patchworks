# T6 (#555): #540's per-edge stack, built and trained

The rig for [B11](https://github.com/NGL321/patchworks/issues/555) on the map
[*Map: transport whose composed rank exceeds one*](https://github.com/NGL321/patchworks/issues/532).

[B2](https://github.com/NGL321/patchworks/issues/540) ruled a three-part stack —
per-edge `m_e`, lateral lanes at `m = 1`, and the privacy invariant doubled to
`Σ_e m_e ≤ 2n − 1` — and priced it at composed rim-to-apex ER **2.193** against a
bar of **generic median ≥ 2.0**. That figure came from generic chains at the
widths the stack was *expected* to realise. **No dome had ever been built that
way.** [B6](https://github.com/NGL321/patchworks/issues/546) could only vary
`interior_m`, one global constant, which **saturates** short of the stack
(rebuilt construction ER 1.4115 at m=10, 1.6158 at 14, 1.7772 at 20) because
`k_v = min(n, Σ_e m_e)` caps at `n = 32` while the chain stays seven hops.

This rig builds it, reads it, and trains it.

## Files

| file | what it does |
|---|---|
| `stack.py` | the build. `allocate` is the per-edge rule; `build_stack` rebuilds the dome with it; `generic_at` runs #540's own instrument at the *realised* widths |
| `departure.py` | attributes the gap between the built reading and #540's generic prediction |
| `trained_stack.py` | T5's `trained_width.py` on the stack, with `sigma_read` and `cap_read` at every checkpoint and `sweep_c` **dropped** |
| `analyse.py` | the readout tables |
| `check.py` | re-derives every figure asserted below; run it before quoting one |

Records are `555-*.json`. The rig sits on top of `worktree-cs-546`, which is
**not yet on `main`**.

## The allocation rule, because "largest both endpoints can afford" is under-determined

Two edges at one cell compete for one budget, so "largest" depends on which is
served first. `allocate` implements **max-min fairness** — every free edge starts
at 1 and is raised a unit at a time, round-robin, while both endpoints have slack
— which is the only order-independent reading. Boundary (`boundary_m = 4`) and
drive (`drive_m = 1`) lanes are held fixed: #540's ruled stack does not move
`boundary_m`, and the drive edge carries
[T7](https://github.com/NGL321/patchworks/issues/535)'s standing flag that
ADR-0009's pre-registered width re-read is due and unread.

It yields **17 distinct chain-width profiles** over the 263 rim-to-apex chains,
modal `[4, 12, 12, 13, 12, 13, 12]` (17% of chains), where #540 hand-priced
`[4, 10, 10, 14, 12, 12, 12]`.

## 1. The instrument is calibrated before anything is read off it

The `today` control — the same code path, per-edge allocation off — reproduces
the record exactly:

| | built ER median |
|---|---|
| `today` control, rebuilt here | **1.0255** |
| the ledger's construction reading (T3, row 6 on [#520](https://github.com/NGL321/patchworks/issues/520)) | **1.025** |
| [B1](https://github.com/NGL321/patchworks/issues/537)'s generic prediction | 1.0216 |

## 2. The rungs, rebuilt

Seed 42, 263 chains, at construction.

| rung | built ER | p90 | max | generic at realised widths | private dim total | cells at private dim 0 |
|---|---|---|---|---|---|---|
| today | 1.0255 | 1.3096 | 1.9886 | 1.0273 | 1278 | 0 / 150 |
| (d) per-edge `m_e` | 1.0991 | 1.5882 | 2.1347 | 1.2282 | 334 | 0 / 150 |
| (c1) + laterals at 1 | 1.1669 | 1.7192 | 2.2658 | 1.3553 | 776 | 0 / 150 |
| **(b) + invariant `2n − 1`** | **1.5933** | 2.0521 | 2.6825 | **2.0637** | **35** | **122 / 150** |

Zero invariant violations at every rung.

## 3. #540's 2.193 does not reproduce — and the reason exonerates #540's law

#540 named its own falsifier: *"if a surface is found whose composed ER departs
from the generic prediction at its own per-hop widths, §2's whole table is wrong
and the lever ranking must be re-derived."* The built stack reads **1.5933**
where the generic instrument at its own realised widths reads **2.0637**. The gap
attributes cleanly, and **the genericity claim is not what fails**:

| predictor | median |
|---|---|
| Haar `V`, unit `σ` — #540's instrument | 2.0637 |
| **the surface's own `V`**, unit `σ` | **2.0756** |
| the surface's own `V` **and** `σ` | 1.5933 |
| truth (`composed_er`) | 1.5933 |

The carried subspaces **are** generic. The whole gap is **singular-value spread**,
which #540's instrument set to 1 — `σ_min/σ_max` on a map's active block reads
0.628 at today's widths and **0.320** at the stack's, so the idealisation gets
worse exactly where the lever operates.

**And it is a construction artifact.** ADR-0032's band is a *training-time*
projection, not a construction one: at **100 ticks** `σ_min/σ_max` reads
**1.0000** and composed ER reads **1.9537**. So the construction number
understates the stack, and the trained arm is the only reading that decides it.

## 4. B9's sign constraint is satisfied

[B9](https://github.com/NGL321/patchworks/issues/551) showed `c` is not a lever in
either direction because pushing a cell's incident maps apart destroys the
principal-angle alignment composed rank is made of. Per-edge `m_e` changes what
each cell holds, so #555 asked whether the stack buys its rank by raising
incoherence. **It does not.**

| rung | Gram cap ratio median | p90 | cells at cap | leading per-hop cosine |
|---|---|---|---|---|
| today | 0.3288 | 0.3882 | 4 / 414 | 0.5680 |
| (d) | 0.3271 | 0.3878 | 4 / 414 | 0.6580 |
| (c1) | 0.3252 | 0.3907 | 4 / 414 | 0.7338 |
| (b) | 0.3294 | 0.3900 | 4 / 414 | **0.9372** |

The cap does not move; alignment rises sharply. Width raises alignment where
tightening `c` destroyed it. `GAUGE_C = 2` is untouched by the stack.

## 5. The trained arm — the stack fails the bar, and buys nothing over the width knob

Baseline, seed 42, dome rebuilt to the full stack.

| ticks | ER median | p90 | max | erosion | cap p90 | cells at cap | `σ_min/σ_max` |
|---|---|---|---|---|---|---|---|
| construction | 1.5933 | 2.0521 | 2.6825 | — | 0.3900 | 4 | 0.3203 |
| 100 | 1.9537 | 2.5625 | 3.1515 | ×0.6 | 0.2500 | 4 | **1.0000** |
| 1,000 | 1.3023 | 2.1574 | 2.8973 | ×2.0 | 0.3150 | 4 | 1.0000 |
| 5,000 | 1.1090 | 1.6041 | 2.5667 | ×5.4 | 0.5258 | 4 | 1.0000 |
| 10,000 | 1.0771 | 1.5280 | 2.7327 | ×7.7 | 0.6581 | 4 | 1.0000 |
| **20,000** | **1.1099** | 1.6013 | 2.8150 | **×5.4** | **0.9157** | **11** | 1.0000 |

`σ_min/σ_max` closing to 1.0000 by 100 ticks is ADR-0032's band projection, and
it is why the reading *rises* above construction before it erodes.

**Against B6's uniform `interior_m = 14`, the nearest arm on the record:**

| | construction | 20k | 100k | private dim total | cells at private dim 0 |
|---|---|---|---|---|---|
| B6 uniform `m = 14` | 1.6158 | 1.1555 | 1.1379 | — | — |
| **#540's full stack** | 1.5933 | **1.1099** | *(running)* | **35** | **122 / 150** |

The stack trains to *slightly less* than a single global constant B6 could already
set, while spending 97% of the graph's private dimension. **Its whole advantage is
at construction, and training removes it.** The best trained composed rank
anywhere on the record remains B6's **1.186**, which the stack does not reach.

**The one qualification to §4.** At construction the stack does not raise
incoherence. Under training it presses the cap harder than any surface on the
record — p90 **0.9157** and **11** cells at cap by 20k, against `m = 14`'s 0.3827
at 20k and 0.5241 at 100k. The median stays pinned at 0.2500 throughout, exactly
as [B7](https://github.com/NGL321/patchworks/issues/547) and
[B10](https://github.com/NGL321/patchworks/issues/552) found, so nothing here
re-opens `c` — but B9 showed a binding cap destroys the alignment composed rank is
made of, and whether that is what caps the stack is the 100k arm's to say.

## 6. What the doubled invariant costs

`docs/spec/06-graph-topology.md`, *Private dimension is a gradient*, states the
invariant's content in its own words:

> **It is nowhere zero, and that is [#474](https://github.com/NGL321/patchworks/issues/474)'s
> doing.** This table used to read `0` for every vision row, and 82 of the 150
> predicting cells had no private width at all. […] derived from `Σ_e m_e ≤ n − 1`
> — **the invariant whose whole content is that this column is never zero.** The
> floor is **`p_v ≥ 1`**.

Private dimension is `max(0, n − Σ_e m_e)` and `n = 32`, so it is zero for any
cell whose allocation reaches 32. The cap `2n − 1 = 63` sits **above** `n`, and a
fair allocator spends to the cap. As built, the doubled invariant takes **122 of
150** predicting cells to private dimension **zero** — worse than the 82 of 150
that #474 was raised to fix — and `dim H⁰ ≥ Σ_v max(0, n − Σ_e m_e)` falls
**1278 → 35**.

So `2n − 1` does not relax the invariant. **It negates its entire content.** The
general statement: *any cap above `n − 1` that is actually spent to is abolition
for every cell that spends to it* — `2n − 1` and "no budget" differ only in how
wide a lane may get, not in whether private dimension survives.

Rungs (d) and (c1) are clean on this axis: both hold `Σ_e m_e ≤ 31` with zero
violations and keep every cell non-zero, at 334 and 776 total private dimension.
