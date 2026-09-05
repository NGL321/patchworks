# The constant-evidence collapse, as a twelve-line toy

Companion to [`docs/audit/collapse-mechanism-note.md`](../../docs/audit/collapse-mechanism-note.md).
Pure Python, no dependencies, one cell, about a minute. Run from the repository root:

```bash
python3 prototypes/audit-collapse-toy/collapse_toy.py            # 10,000 steps
python3 prototypes/audit-collapse-toy/collapse_toy.py --steps 20000
```

## What it is

The prediction rule's update to `K` is an outer product, `dL/dK = (Dᵀe) hᵀ` (`learning.py`,
`prediction_error`), and the used operator is `K / max(1, σ_max(K))` (`body.py`, before and after
#466). The toy keeps exactly those two facts and nothing else, then asks what happens to the used
operator's spectrum when the evidence direction `h` is constant and the error `e` is persistent.

The answer is the shape [#477](https://github.com/NGL321/patchworks/issues/477) measured on the real
graph: `σ_max` pinned at 1, `ρ` falling well beneath it, stable rank toward 1, non-normality toward
its rank-1 ceiling of `√2`. The controls (a varying evidence direction, or a bias absorbing the
persistent error) do not collapse.

## What it is not

Not the architecture. There is no `encode`, no reconciliation, no neighbour, no chart round trip.
It cannot say how fast the real collapse runs, only which way it runs and why. The note says what
reads on the committed 100k checkpoints would confirm or refute the mechanism on the real cells.
