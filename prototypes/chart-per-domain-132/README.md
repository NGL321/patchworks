# #132 — must the chart be a per-domain number?

Instrument for [#132](https://github.com/NGL321/patchworks/issues/132): *is `k_piece`
materially different between the dome and a character stream, and if so what does that
cost [#128](https://github.com/NGL321/patchworks/issues/128)'s one frozen dictionary?*

## What this rig had to get past first

#132 had been re-specified three times and arrived carrying two blockers of its own
making.

**§1 — the instrument is not on `main`.** `prototypes/chart-double-duty-166/read.py`
lives only on `origin/worktree-chart-double-duty-166`.

**§2 — the read must be taken post-floor, and the floor is not on `main` either.**
[#432](https://github.com/NGL321/patchworks/issues/432) is *closed* but its PR
[#434](https://github.com/NGL321/patchworks/issues/434) is **open**, so
`RestrictionMaps.project()` on `main` still runs mask → band → `_push_apart` → re-cap
with no spectral floor. The ticket's blocker cleared as a *decision* while the *code*
stayed unmerged — [#240](https://github.com/NGL321/patchworks/issues/240)'s exact
class, arriving one level up.

So the runs here are taken on a stack of `main` + PR #434 + the #166 rig, and the base
is stated rather than assumed.

**§5 — the language term "cannot be taken by anyone today", because nothing builds a
language graph.** That is true of the *operator* half and **false of the `k_piece`
half**, which is what this rig turns on. ADR-0004 defines `k` as *the dimension of the
piece*, and its criterion is explicitly read **forwards**: *"known before anything runs
— an embedding is generic once the coordinate count exceeds twice the piece's
box-counting dimension — which makes self-intersection predictable at construction."*
A quantity knowable before anything runs does not need a builder. The piece is a fact
about the world; the graph is what charts it.

## The three files

| file | what it answers |
|---|---|
| `piece_dimension.py` | `k_piece` in both domains, by Grassberger-Procaccia on a configuration sweep |
| `bus_widths.py` | the **communication bus** `Σ_e m_e` per cell, both domains, counted the same way |
| `run.py` | the #166 operator read on the **post-floor** channel at 100k, with a pre-floor control |

## What `piece_dimension.py` found, and why it is a theorem rather than a sample

**The language piece is discrete, so it has no box-counting dimension to be
per-domain about.** An L1 wedge cell owns `fan = 4` buffer slots and heard is a 97-way
one-hot, so two windows differing in `j` slots sit at distance exactly `√(2j)`. There
are **four** possible non-zero distances, full stop:

| `j` slots differ | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| distance | `√2` | `2` | `√6` | `√8` |
| measured, ÷ median | 0.5000 | 0.7071 | 0.8660 | 1.0000 |

The rig measures exactly those four atoms, with **79.9%** of pairs on `j = 4`. `C(r)`
is therefore a **staircase**, not a power law: flat across three decades, then a jump
to 1 within a factor of two. The 11.05 the fitter reports is a fit across a two-point
riser and is **not a dimension** — `scaling_spread` comes back `None`, which is the
rig declining to certify it.

**This makes the result independent of the character stream**, which is the useful
part: the atoms follow from one-hot geometry, not from the statistics of any corpus.
The stream is the repo's own prose only because *some* stream is needed to count
masses; no other stream can change the four atoms, and none of
[#1](https://github.com/NGL321/patchworks/issues/1)'s corpus-pretraining exclusion is
touched by using one.

**The lanes do not rescue it, and the rig checks rather than asserts that.** The
obvious objection is that an L1 cell never sees the raw stalk — what arrives is the
image under four `m = 8` restriction maps, 32 numbers. Measured through exactly that:

| | raw one-hot | through 4 × `m = 8` lanes |
|---|---|---|
| ambient | 388 | 32 |
| **distinct points** | **3,959** | **3,959** |
| distinct pair distances | 4 | 260,248 |
| mass on the heavy atoms | 1.000 | 0.000 |
| certified scaling region | none | none |

The projection re-spreads the *distances* and changes nothing about the *set*: the
distinct-point count is identical, because a linear map cannot turn a finite set into
an infinite one. **Cardinality is the claim; the atoms are only its shape in the
coordinates the world happens to write.** Neither reading yields a certified scaling
region, so no box-counting dimension can be quoted for language at all — and that is
the answer rather than a failure of the rig.

**The dome's piece is continuous and low-dimensional.** Over the 52 of 64 L1 vision
cells whose aperture varies at all, and the 25 with a certifiable scaling region:

| | value |
|---|---|
| distinct pair distances | 51 to 191,346 (median 2,073) |
| cells reading discrete | **0 of 52** |
| cells with a certified scaling region | 25 |
| `d_corr`, quartiles over those 25 | **1.26 / 1.43 / 1.53** |
| coincident-pair fraction, median | 0.872 |

ADR-0004's genericity bar is `coordinates > 2 · d_box ≈ 2.9`. **`k = 12` clears it by
about fourfold and `n = 32` by elevenfold.** Read per cell, never as a graph-wide
average — #127's standing rule, #181's per-edge form.

### This is the demand `docs/research/032` argued for and never measured

`032`'s capacity table is a **ceiling on what a stalk could chart, not a demand the
piece makes on it** — a distinction it states in those words, because
[#166](https://github.com/NGL321/patchworks/issues/166) read the ceiling back as the
demand and got a worked example that "looks forced". `032` then puts the demand at
*"far below 6, in a world of about twenty numbers"* — **by argument, never measured**.
1.43 is that number, measured:

| width | capacity, `032` | measured demand | margin |
|---|---|---|---|
| chart `k = 12` | box-dim < 6 | 1.43 | ~4x |
| node stalk `n = 32` | < 16 | 1.43 | ~11x |
| boundary lane `m = 8` | < 4 | 1.43 | ~2.8x |
| **interior lane `m = 4`** | **< 2** | 1.43 | **~1.4x** |

**The thinnest margin is the lane, not the chart** — which is what `032` predicted
(*"`m = 4` is the number with the least headroom, and it is the one no source
vindicates"*) and what [#14](https://github.com/NGL321/patchworks/issues/14)'s ladder
already permits flexing first. It is also the same place `bus_widths.py` lands from
the other direction.

One care with that row: `m` carries the **overlap** between two cells, not a cell's
whole piece, and an overlap's dimension is bounded above by the piece's. So 1.43 is an
upper bound on what the lane must carry and the true margin is no worse than ~1.4x.
The chart's fourfold headroom does not depend on that subtlety; the lane's does, and
nothing here rules on `m` — that is not this ticket's question.

The high coincidence rate is aperture sparsity, not an artifact: most sweep
configurations leave a given 8×8 block showing arena floor. It is excluded from the
power-law fit and reported separately, because it is itself a reading — it is
ADR-0004's **self-intersection**, measured in the piece rather than in a stalk.

## What `bus_widths.py` found — §4's suspicion runs the other way

§4 reads `11-the-language-graph.md`'s table as showing the structural zero arriving
**harder** at the language rim than at the dome's. Counted the same way on both sides,
it arrives **softer**:

| L1 predicting cell | `Σ_e m_e` | as `× n` | overflow past `n = 32` |
|---|---|---|---|
| **dome**, vision | **44 – 52** | 1.38 – 1.62 | 12 – 20 |
| **dome**, somatomotor | 32 – 40 | 1.00 – 1.25 | 0 – 8 |
| **language**, L1 | **40 – 44** | 1.25 – 1.38 | 8 – 12 |

The dome's vision rim's **minimum** equals language's **maximum**. The counting
reproduces [#385](https://github.com/NGL321/patchworks/issues/385)'s **82 of 150**
predicting cells at the structural zero exactly, which is what licenses the
cross-domain comparison: the same rule, run twice, hits a number the record already
owns.

## What `run.py` found — the floor does not move this statistic, but `main` already has

§2's objection is that a `K` spectrum read through today's near-rank-1 channel is *"a
read of the collapse, not of the domain"*. It is testable rather than assumable, so
the read is taken twice. No-oping `_flatten` reproduces `main`'s `project()` exactly —
the only behavioural change PR #434 makes to `project()` is that call, the rest being
`__init__` bookkeeping and a new reporting method.

Seed 42 at 20,000 ticks:

| | stable rank | effective rank | ρ(K) | Σt | non-normality |
|---|---|---|---|---|---|
| post-floor | 10.253 | 11.890 | 0.958 | 214.8 | 0.0138 |
| pre-floor | 10.295 | 11.893 | 0.960 | 222.4 | 0.0139 |

**The floor moves the chart read by ~0.4%**, against a post-floor seed-to-seed spread
of 10.25–10.43 in stable rank and 0.0133–0.0147 in non-normality. Whatever the floor
does to the *maps* — [#435](https://github.com/NGL321/patchworks/issues/435)'s
question — it does not change what `K` looks like.

**What has moved is `main` itself.** Same seed, same horizon, same rig, against #166's
recorded run:

| seed 42 @ 30k | #166, pre-ADR-0031 | current `main` |
|---|---|---|
| stable rank | 8.777 | **10.215** |
| effective rank | 11.658 | 11.876 |
| ρ(K) | 0.894 | 0.956 |
| summed τ | 84.45 | **199.69** |
| non-normality | **0.0504** | **0.0158** |

[ADR-0031](../../docs/adr/0031-the-sparsity-pressure-is-deleted.md) deleted the
sparsity pressure from the transport rule between the two. #166's *conclusion*
survives and strengthens — there is now more width unspent, not less — but its
numbers, including the two #127's *Decisions so far* quotes for it, are stale. The
non-normality figure matters most: it is **3.2x smaller** than the 0.05 that
[#357](https://github.com/NGL321/patchworks/issues/357) was spun out against.

## Running it

```
PYTHONPATH=src python prototypes/chart-per-domain-132/piece_dimension.py --samples 6000
PYTHONPATH=src python prototypes/chart-per-domain-132/bus_widths.py
PYTHONPATH=src python prototypes/chart-per-domain-132/run.py --ticks 100000 --seeds 42 --arms postfloor
```

## Caveats

- **The heard stream is what is measured.** `11`'s spoken stalk carries an uptake flag
  and a coherence scalar, which are not one-hot, so the spoken half of the L1 rim is
  not strictly discrete. It is not measured here because its distribution is the
  interlocutor's and there is no interlocutor; the heard half is decisive on its own,
  because the destination's claim is a **universal** over predicting cells.
- **The bus comparison is spec against build.** The dome's side is read off
  `build_graph(DEFAULT_SPEC)`; the wedge's side is arithmetic from `11`, since the
  wedge has no builder. Both apply `dim H⁰ ≥ max(0, n − Σ_e m_e)` and the dome side
  reproduces #385's figure, but they are not the same kind of artifact.
- **The dome's sample is thin per cell, and the thinness is physical.** A given 8×8
  block shows arena floor in most configurations, so a cell sees a median of 381
  distinct appearances in a 6,000-configuration sweep. The direction of the
  comparison does not turn on it — 1.43 against a bar of 12 has four-fold headroom —
  but a `d_corr` quoted to two decimals would be over-claiming, which is why the
  quartiles are given and the certified-fit count with them.
- **`k_piece` is not the whole of `k`.** Under [#145](https://github.com/NGL321/patchworks/issues/145)
  the chart carries the piece's coordinates *and* the memory of the history that wrote
  them. This rig reads the first. The second is `run.py`'s, and per
  [#144](https://github.com/NGL321/patchworks/issues/144) it is first a claim about
  the field — [#375](https://github.com/NGL321/patchworks/issues/375) — before it is
  one about width.
