# ADR-0006: Boundary cell node stalks are world-shaped, not `n`-shaped

**Status:** accepted

## Context

`01-cell-and-sheaf.md` makes the node stalk dimension `n` a **global constant** — every cell's public
face has the same dimension. `04-action-and-the-boundary.md` puts the world's seam at the node stalk:
a boundary cell is one whose node stalk the environment writes (sensory) or reads (motor), and it
bans, narrowly, any out-of-graph module that compresses across slices on the way in.

Those two commitments collide at the sensory tiling. The world writes a patch of the render into a
boundary cell's node stalk directly, with no compressor in between, so **the patch must fit in the
stalk, raw**. The render is `(64, 64, 3)` and colour is load-bearing — puck identity *is* colour — so
a `p × p` patch is `3p²` values:

| patch | values | required `n` | sensory cells |
|---|---|---|---|
| 2×2 px | 12 | ≥ 12 | 1024 |
| 4×4 px | 48 | ≥ 48 | 256 |
| 8×8 px | 192 | ≥ 192 | 64 |

With `n` global, an 8×8 patch imposes a 192-dimensional stalk on **every cell in the graph**. Via
`dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)` that hands each interior cell some 170 private dimensions in a
world whose actual state is roughly twenty numbers. Choosing a smaller patch to protect `n` makes the
tiling granularity a hostage to a dimension it has nothing to do with.

## Decision

**Boundary cells are exempt from `n`. A boundary cell's node stalk has the dimension the world gives
it.** For the sandbox: 48 for a sensory patch cell, 2 for a proprioceptive cell, 6 for the actuator
cell (three commanded, three efference).

`n` is henceforth **the node stalk dimension of every predicting cell**.

**Amended by [#36](https://github.com/NGL321/patchworks/issues/36): world-shaped is the *reason*, not
the rule.** A **drive boundary cell** is written from outside the sheaf but not by the world, so
nothing out there gives it a dimension. The rule is that a boundary cell is **exempt from `n`**; what
sizes its stalk is whatever writes or reads it. For the world that is the world's shape. For the drive
it is what the drive asserts, which is one number
(`04-action-and-the-boundary.md`, *Valence, not specification*), and its edges are `m_e = 1` to match —
a wider communication lane cannot raise the rank of a map out of a one-dimensional stalk.

**Every lane in the graph remains ordinary.** The world touches node stalks only; it is not a
cell, holds no restriction map, and there is no edge between the world and the graph. A patch cell
reaches its neighbours by an ordinary linear masked restriction map, 48 → `m`.

## Consequences

- **The single-GPU argument is untouched.** `n`'s globality exists so the shared frozen body can run
  batched ([ADR-0001](./0001-continual-learning-applies-to-the-adapting-surface.md)). Boundary cells
  run no body — `CONTEXT.md` already has them performing no inference — so they were never in the
  batch. The exemption costs nothing it was protecting.
- **The compression ban is not holed.** The 48 → `m` squeeze is performed by a cell, inside the graph,
  linearly, under the mask, costing a tick. That is a cell doing a cell's job, which is the opposite
  of what `04-action-and-the-boundary.md` bans.
- **Tiling granularity is freed** from `n` and can be chosen on vision grounds. It was chosen at 4×4,
  so that no cell ever sees a whole puck and recomposition is forced (`06-graph-topology.md`).
- **`χ` and `dim H⁰` must be computed over predicting cells only.** Boundary terms swamp the
  diagnostic — 256 cells × 48 dimensions of nominally private state that the world overwrites every
  tick and no cell holds. This corrects the diagnostic as recorded in `01-cell-and-sheaf.md`.
- **Boundary cells become more clearly a distinct kind of object** than "degenerate cell obeying the
  contract" suggested. They hold no chart, run no body, have no `k`, and now no `n` either. What
  remains of the contract for them is the stalk-and-restriction-map interface.

## Alternatives considered

- **Keep `n` global and shrink the patch to 2×2.** Satisfies `n = 32` easily but yields 1024 sensory
  boundary cells against ~150 predicting cells, and makes a dimension internal to the graph dictate
  how the world is sliced.
- **Keep `n` global and raise it to 192.** Rejected on the junk-capacity argument above, and it would
  make every lane, restriction map, and body forward pass in the graph carry a dimension only
  the rim needs.
- **A learned compressor between the render and the stalk.** This is exactly the banned act: it
  compresses across a slice on the way in, outside the graph, which is what cells exist to do.
- **Let the world write into a lane instead.** Would require the world to hold a restriction
  map, making it a cell. The seam is at the node stalk and stays there.
