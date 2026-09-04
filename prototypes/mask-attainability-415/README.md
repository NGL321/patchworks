# Mask attainability, for #415 / ADR-0032

Pre-registered by [#411](https://github.com/NGL321/patchworks/issues/411) §8 and taken on
[#415](https://github.com/NGL321/patchworks/issues/415), because the ADR may not claim its constraint
is attainable without it.

**The question.** ADR-0010's *Alternatives considered* rejected orthogonality on two grounds. #411
struck the first — a co-isometry constraint touches only `Σ` and leaves the basis alone — and the
second survives as the real caveat: **a mask may not contain a co-isometry.** For each structural mask
on `build_graph(DEFAULT_SPEC)`: does it, and how far from flat is the projection's fixed point where it
does not?

**Arithmetic, not a run.** Masks are construction-fixed prefixes of the node stalk, so attainability is
decided at build time and needs no training. `k` is the permitted column count; a scaled co-isometry
`FFᵀ = σ²I_m` exists iff `k ≥ m`. The spectra are read in float64 on each map's active `m × k` block,
so structural zeros and the padded tensor's out-of-range rows never enter an SVD. Seed 42 at the draw;
the attainability half does not depend on the draw at all.

## Reading

```
python prototypes/mask-attainability-415/read.py     # output in reading.txt
```

| population | endpoints | masks containing a scaled co-isometry |
|---|---|---|
| **banded** (a predicting cell's own maps) | 1091 | **1091** |
| **pinned** (a boundary cell's own maps) | 273 | 264 |

**Every banded mask is attainable, with room to spare.** `(m, k)` runs `(4, 32)`, `(4, 24)`,
`(8, 32)`, `(4, 28)`, `(4, 20)`, `(4, 17)` and `(1, 17)` — a minimum margin of `k − m = 13`, and never
fewer than **4x** as many permitted directions as the lane is wide. On those masks the projection lands
exactly flat: max relative departure `1.3e-15` after one projection, and `‖F‖_F` moves by `1.1e-15`, so
**ADR-0010's gauge and ADR-0032's floor do not interact on the maps the projection reaches.**

**The nine that are not attainable are all pinned**, at the small-stalk end of the sensorimotor rim:

| cells | `m` | `k` | best attainable `σ_min/flat` | `‖F‖_F` the projection would take |
|---|---|---|---|---|
| touch (3) | 8 | 1 | 0.354 | −65% |
| proprioceptive (3) | 8 | 2 | 0.500 | −50% |
| actuator (3 maps, one cell) | 8 | 6 | 0.866 | −13% |

`rank(F) ≤ k < m` by construction there, so `σ_min = 0` whatever the projection does — and because the
projection would also shrink `‖F‖_F` by `√(k/m)`, it would fight the exact gauge rather than sit beside
it. **This is the same population ADR-0010 already puts out of the projection's reach**, for the same
reason: a pinned map has no scale freedom to spend. ADR-0032 states the floor on the banded maps and
names these nine as an exclusion rather than leaving them to be discovered.

At `m = 1` — the drive's eight edges — the floor is vacuous: one singular value is `‖F‖_F/√1`
identically.

**A read asserts nothing.** No cutoff filed, no register written, nothing promoted to `benchmarks/`.
