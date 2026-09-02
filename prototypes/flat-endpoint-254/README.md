# #254: why #229's two checks fail, measured

Both failures reproduce **only in the container** (ADR-0012's supported target).
On an out-of-spec host — Python 3.14 / torch 2.11 / numpy 2.x — the
flat-endpoint test *passes*, so a local green is not evidence here. Every
number below is from `ghcr.io/ngl321/patchworks:latest`
(Python 3.12.14, torch 2.2.2+cpu, numpy 1.26.4, `MUJOCO_GL=osmesa`), which
matches `ci.yml` exactly.

Baseline: on `main` both tests pass; on `main` merged with #229 the suite is
**2 failed, 1347 passed** — exactly the two the ticket names.

## 1. The flat-endpoint invariant is not falsified, and the projection is not over-reaching

Neither of the ticket's two hypotheses holds. Measured:

* **The projection never reaches a map here.** After the fixture's three ticks
  the whole map tensor is **bit-identical** between `main` and #229
  (`sha256[:32] = 053fc5f752c37e18f96d9bce48e0c569` on both). The fixture ticks
  but never runs a learning step, and each drive map has a single open weight
  gauge-pinned to exactly `1`. `project()` is not implicated at all.
* **The objective is exactly as flat under #229 as under `main`.** At every
  drive endpoint whose ends are collinear-and-opposed the relative disagreement
  is `1.000000` and the disagreement gradient is exactly `0`; the `p = 1`
  sparsity term is identically flat (`h ≡ 1`), so it contributes exactly `0` too.
* **What #229 moves is the state, through the gain.** Its `tick.py` swaps the
  reconciliation gain's denominator, which moves the node stalks and `incoming`,
  which changes *which* drive endpoints sit opposed: `main` has one (ep104),
  #229 has two (ep102 and ep106). The invariant holds at **more** endpoints
  under #229, not fewer.

### So why does the test fail?

It reads `unmoved` by **bit-identical** comparison of the update before and
after the perturbation. At ep102 and ep106 the update is exactly `0.0` before
and `+2.980232e-08` / `-4.647432e-08` after — one float32 ulp (`2^-25`) of
rounding in a quantity that is analytically identically zero. The residue lands
on the map's single **open** weight (`support = 1`), not on a masked or padded
one, so there is no gradient-hygiene defect. In float64 the same quantities are
exactly zero.

`main` passes by the luck of the rounding: its one opposed endpoint's perturbed
update happens to round to exactly `0.0`.

So the test's claim is true and its *operationalisation* is too sharp: an
analytically-zero float32 gradient is zero to within an ulp, not to the bit,
and which side of that it lands on is a property of the arithmetic rather than
of the objective.

`endpoints.py`, `terms.py` (splits the update into its disagreement and
sparsity parts), `residue.py` (locates the residue inside the map).

## 2. The render failure is a real, pre-existing defect in the sensory path

Not downstream of a moved trajectory, and not a lossy record:

* The re-render from the record is **bit-exact** against the env's own renderer
  once `mj_forward` runs — `max = 0` at every tick, on both trees.
* `PlanarPushSandbox.step` renders the observation straight after `mj_step`
  without an intervening `mj_forward`, so MuJoCo's derived kinematics are one
  integration behind the `qpos`/`qvel` returned in the *same* observation dict.
  The agent is shown where the arm was at the start of the tick while
  proprioception reports where it is now.
* The error scales with arm speed. On `main`: `|qvel| 0.16 → 1` level,
  `0.49 → 64`, `0.84 → 64`. Under #229 the gain raises `|qvel|` at the sampled
  tick to `1.59`, giving `54`.

**The assertion already fails on `main`.** Run at each capture tick there:

| capture tick | max | ≤1? | differing | <0.01? | |
|---|---|---|---|---|---|
| 4 | 1 | yes | 0.0032 | yes | PASS |
| 9 | 64 | no | 0.0085 | yes | FAIL |
| 14 | 64 | no | 0.0137 | no | FAIL |
| 19 | 99 | no | 0.0164 | no | FAIL |

The test samples only tick 4, where `main` happens to be nearly static. #229 is
not the regression; it moves the first capture tick past the threshold.

`stale_kinematics.py`, `render_per_tick.py`.

### What the one-line fix costs

Calling `mj_forward` before rendering the observation leaves the **qpos
trajectory bit-identical over 20 ticks** (`max abs difference 0.000e+00`,
never differs) while the image changes by up to 99 levels on 1.64% of pixels.
So the world does not move; only what the agent is shown does. Worth
re-checking over a longer horizon when the edit lands.

`forward_cost.py`.

## Running these

```
docker run --rm -v "$PWD:/app" -w /app --entrypoint python \
  ghcr.io/ngl321/patchworks:latest prototypes/flat-endpoint-254/terms.py
```

Each script assumes the tree it is mounted over, so point the mount at `main`
or at the #229 merge to get that tree's numbers.
