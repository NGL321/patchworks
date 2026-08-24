# PROTOTYPE — route geometry of the sandbox

**Throwaway.** Built to answer [issue #25](https://github.com/NGL321/patchworks/issues/25): does the
sandbox actually generate tasks with distinct route homotopy classes, or was
`03-the-sandbox.md`'s "routing around it is a real subproblem" asserted rather than measured?

It was asserted. These two scripts measure it.

## Run

```bash
../../.venv-proto/bin/python homotopy_check.py
../../.venv-proto/bin/python standoff_check.py
```

Both import the sandbox prototype's `sandbox_env` and use its real sampler — no re-derivation of
the geometry, so a change to the sandbox changes these numbers.

## What they found

`homotopy_check.py` — the **puck's** route. For each sampled task, does the straight goal-puck →
goal-zone segment clip the pedestal inflated by the puck's radius, and if so how do the two
wrapping routes (tangent-arc-tangent) compare in length?

| | train | heldout |
|---|---|---|
| straight line clips the pedestal | 37.2% | 41.0% |
| detour cost, median / max | 1.039× / 1.16× | 1.038× / 1.17× |
| two wraps within 15% of each other | 15.0% of all tasks | 17.5% |

`standoff_check.py` — the **paddle's** standing position, which is what actually broke during the
sandbox prototype (its README, defect 4). To push the goal puck at the goal zone the paddle must sit
at `p − u·(puck_r + 0.03)`; is that point inside the pedestal + paddle disk?

| | train | heldout |
|---|---|---|
| direct push impossible | 2.0% | 1.0% |
| including marginal (within 4 cm) | 5.7% | 3.7% |

**Conclusion.** The pedestal is topologically a real obstacle and metrically a graze — a median 4%
detour, and the unrecoverable-standoff failure it was added to fix now fires on 1–2% of tasks. It
worked well enough to remove most of the occasions to route around it.

The thing that *is* load-bearing needed no sampling to see, and neither script measures it: because
links 1 and 2 collide with the pedestal (`arena.xml`), the paddle's reachable set is an **annulus**
(inner 0.11, outer 0.49), not a disk. `π₁` is `ℤ`, so every repositioning of the paddle across the
arena is a choice of swing direction — several times per task, and following from the arm being
anchored at the centre rather than from the pedestal being an obstacle. That is what
`04-action-and-the-boundary.md` (*Route selection*) tests.
