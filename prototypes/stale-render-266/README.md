# The observation's image lagged its own proprioception (#266)

[#254](https://github.com/NGL321/patchworks/issues/254) ruled that
`tests/test_surface.py::TestOneRenderer::test_the_scene_is_re_rendered_from_the_state` was failing
because `PlanarPushSandbox.step` rendered the observation straight after `mujoco.mj_step`, with no
`mj_forward` between. `mj_step` advances `qpos`/`qvel` and does not refresh what is *derived* from
them, so the image showed the arm where it was at the **start** of the tick beside a `qpos` and
`qvel` saying where it was at the end.
[#266](https://github.com/NGL321/patchworks/issues/266) carries out the fix; these are its
measurements.

```
PYTHONPATH=src python prototypes/stale-render-266/stale_channels.py
PYTHONPATH=src python prototypes/stale-render-266/world_unmoved.py
PYTHONPATH=src python prototypes/stale-render-266/reading_split.py
PYTHONPATH=src python prototypes/stale-render-266/forward_cost.py
```

## `touch` was stale too, not only the image

`stale_channels.py` compares every channel of the observation `step` returns against the same
observation re-read after an `mj_forward`. #266 described the defect as the image lagging; it is
wider than that. `sensordata` is derived the same way and lagged the same tick — the image by up to
**64 levels** on ~0.5–1% of pixels, and `touch` by up to **1.80**. `qpos` and `qvel` are untouched by
a forward, being the integrated state itself. Two of the four channels described one instant and two
described another.

## The world does not move, and the forward is 9 µs

`world_unmoved.py` drives two envs with an identical action stream, one taking `step` as it now
stands and one replaying `step` as it was, and compares bit-exactly. #254 saw `qpos` bit-identical
over 20 ticks; #266 asked for a longer horizon. Over **2,000 ticks × 5 seeds**, with 83–169
contact-bearing ticks per seed, `qpos` **and** `qvel` are bit-identical. `mj_step` computes forward
dynamics itself, so nothing about the physics depends on the call.

`forward_cost.py` prices it: `mj_forward` alone is **0.009 ms**, against a camera-drawn step of
0.731 ms (**1.2%**) and a camera-blanked one of 0.147 ms (**5.9%**).

## Which readings move: rendering is not the discriminator

#266 split the readings by *whether the reading renders at all*. `reading_split.py` shows the split
is narrower. `reset()` already ends in `_rederive_from_state`, which forwards, so an observation
taken from a reset was **never** stale — run against the pre-fix tree, its reset rows read zero while
its step rows read 54–64. The discriminator is **does the reading render after a step**.

That spares two readings the ticket expected to be caught. `benchmarks/alignment_read.py` and
`benchmarks/construction_grading.py` route through `untrained_fixed_point.build`, and so do render —
but each takes one observation from `env.reset` and then holds it still through `detectability.hold_still`,
which ticks the sheaf and never touches the world. Neither moves.

## A pre-existing replay flake, found and ruled out — not #266's

`tests/test_sandbox_snapshot.py::test_a_replayed_hundred_tick_tail_shows_zero_divergence` failed once
in a full-suite run with the edit in the tree, and did not reproduce in three further runs. It sits
close enough to this edit to be worth settling rather than dismissing, and `mj_forward` re-solves and
rewrites `qacc_warmstart`, so the suspicion had a mechanism behind it.

`replay_determinism.py` runs the test's own body over forty mid-flight seeds. **The edit is not the
cause, and slightly reduces the rate**: 3 of 40 with it, **5 of 40 without**. Every divergence is on
`image` alone — the whole `mjSTATE_INTEGRATION` matches at that tick and at every tick before it, so
the physics replays exactly. Two further checks place it:

* `render_repeatability.py` — the renderer redraws one unchanged state 200 times, **0 differ**. The
  renderer is not simply non-reproducible.
* the divergence **does not repeat for a given seed**. Seed 0 diverges in one 40-seed run, not in the
  next, and not at all when run alone; consecutive runs of the identical program fail at different
  seeds. So it is not a function of the state.

What is left is the process history: forty envs, each building and closing its own `mujoco.Renderer`,
against this machine's hardware GL. CI renders through `MUJOCO_GL=osmesa` in software, which is why it
does not see this. Out of scope here — recorded so the next reader of that test has the measurement
rather than the suspicion.
