# The holonomy read (#453)

The run behind [#453](https://github.com/NGL321/patchworks/issues/453)'s reading,
kept because the ticket is [#454](https://github.com/NGL321/patchworks/issues/454)'s
cutoff and a cutoff that fires on a number nobody can re-open is a cutoff that
fired on folklore.

- `read.log` — the rig's whole output, one process: the free arms, the two
  trained trajectories, their rewired nulls, and the report.
- `read.json` — every one of the 260 cycles' columns for every arm, written as
  each checkpoint landed rather than at exit (`holonomy_read.read`'s `sink`, and
  `spectral_floor_read`'s reason: this run is hours long on a shared box).
The run also writes `maps-{floored,unfloored}-{30000,100000}.pt` beside these —
the four surfaces the readings were taken on, so a later session can compute a
control this rig did not think of without paying 1.6 h to reach the checkpoint
again. **They are not committed**: 2.7 MB each against a 439 KB largest binary
anywhere in `prototypes/`, which is a weight this directory has no precedent for
and nothing here needs. Re-run to regenerate them; the reading itself is
`read.json`, per cycle, which is the record.

Reproduce with:

    python benchmarks/holonomy_read.py read

## What it read

Medians over the 260 independent interior cycles, seed 42.

| median over 260 cycles | chance null | floored 30k | floored 100k | unfloored 100k |
|---|---|---|---|---|
| **channel return `\|<u₁,v₁>\|`** | 0.399 | 0.9815 | **0.9881** | **0.9941** |
| cycles above 0.9 | 4.0% | 90.0% | 91.2% | 99.2% |
| its own rewired null | — | 0.397 | 0.457 | 0.433 |
| identification, whole operator | 0.997 | 0.895 | 0.888 | 0.896 |
| flatness of `H` | 3.3e-4 | 1.7e-5 | 1.5e-5 | 8.5e-10 |
| `sigma_max` of `H` | 7.4e-6 | 0.187 | 0.500 | 1.15 |

**The transport rule moves cross-edge alignment toward the identity, on the one
direction it transmits.** The channel return sits at 0.99 against a chance null
of 0.40, at every cycle length from 3 to 14, at both horizons, with the floor and
without it. On the unfloored surface at 100k **not one of the 260 cycles** falls
below 0.876.

**The rewired null is what makes that a statement about alignment.** The same
trained maps, permuted among endpoints of the same block shape and composed
around the same cycles, read 0.457 — chance. So the channel return is not
arithmetic about rank-1 factors; it is which map sits next to which.

**Off the channel, nothing moves.** Identification departure on the whole
operator goes 0.997 → 0.888, and flatness around a cycle stays five to nine
orders of magnitude below 1 in both arms. The floor buys four orders of that
flatness (1.5e-5 against 8.5e-10) and does not buy the channel return, which is
marginally *better* without it.

## The limits, stated rather than discovered

- **One seed (42) on the trained arms**, eight draws on the free ones. A 100k
  trained trajectory is ~1.6 h on this box and the read needs two of them.
- **The rewired null destroys all incidence structure at once**, so it says *the
  wiring matters* and does not isolate which part of the wiring.
- **`F^T` is the adjoint, not the pseudoinverse.** They coincide on a
  co-isometry, which is what the floor makes each map; on the unfloored arm they
  do not, so that arm's `sigma_max` column carries the maps' scale. The
  identification and channel columns are invariant to per-map positive scale and
  are unaffected.
- **A fundamental cycle basis is not canonical.** BFS is chosen because it gives
  the shortest cycles, which is the choice that flatters the hypothesis under
  test.

**The surface.** Every arm was trained in this process, on `main` at the commit
this directory lands on, and nothing is differenced against stored JSON — the
map's 2026-09-04 standing rule, which exists because ADR-0031 deleted the
sparsity pressure and made every `lambda = 0.4` number a number about a build
that no longer exists.
