# #166 — the chart's double duty, read on the driven rig

Instrument for [#166](https://github.com/NGL321/patchworks/issues/166): *can twelve
dimensions both name the cell's piece and hold the cell's memory?*

## Why this rig exists at all

#166 was opened blocked, on the ground that the measurement *"needs a **predicting**
cell's chart trajectory"* and that
[#155](https://github.com/NGL321/patchworks/issues/155) was *"the edit that makes a
graph transmit"*. **That gate is void.** #155 closed with the per-hop species and no
open edit delivers a transmitting graph;
[#236](https://github.com/NGL321/patchworks/issues/236) read it, and #127 re-gated
#166 onto stage 2's live step. Meanwhile
[#274](https://github.com/NGL321/patchworks/issues/274) built and ran exactly the
thing #166 said could not be had — per-cell chart-loop trajectories on a **driven**
`real` dome, both rules, nine seeds. This rig is #274's harness with the operator
statistics #166 needs read off it, at #274's own checkpoint ladder, so a number here
and a number there are the same moment.

## What it reports, and why not what the ticket asked for

#166 pre-registered *"numerical rank of each learned `K` across the population. If
they saturate at 12, the width is binding."*

**The pre-registered statistic cannot fire, and the rig proves that rather than
asserting it.** `CellOperators.__init__` builds `K = a·I` — rank 12 at construction —
and `project()` restores the band by `K.mul_(target / norms)`, a *rescaling* that
moves every singular value by one common factor and can send none to zero. Numerical
rank 12 is therefore the **null construction hands out for free**, not the signal that
the width is binding. The rank read is still taken and still published, so the vacuity
is on the record as a measurement.

The statistics that carry the question:

| statistic | what it answers | at construction |
|---|---|---|
| `numerical_rank` | the pre-registered read | 12 (the null) |
| `stable_rank` = `‖K‖_F²/σ_max²` | how many directions `K` **uses** | 12 (the max) |
| `effective_rank` = `exp(H(σ̄))` | the same, weighting the tail differently | 12 |
| `rho_K`, `modes_retaining`, `summed_tau` | how the twelve modes split between **holding** and **forgetting** | all 12 retaining |
| `nonnormality` = `‖KᵀK − KKᵀ‖_F/‖K‖_F²` | whether the memory ceiling is **reachable at all** | 0 (normal) |

The last row is the **tier-0 instrument
[#167](https://github.com/NGL321/patchworks/issues/167) prescribed** and the ticket did
not plan — *"measure the non-normality of learned `K` … alongside the numerical rank
the ticket already plans to take. Rank saturation and non-normality answer different
halves of the same question."* Ganguli, Huh & Sompolinsky (PNAS 2008) put the Fisher
memory capacity of a **normal** matrix at *exactly 1*, whatever its dimension. `a·I` is
normal, so a cell begins with one tap rather than twelve, and any memory it acquires it
must acquire by becoming non-normal. Non-normality is normalised by `‖K‖_F²` so the
band's rescaling cannot move it: it reads shape, never size.

**Construction is the maximum on every count**, which is what makes these statistics
informative where the rank is not: the read is *how far below twelve does driving push
it*, and every one of them can only be spent down.

## Running it

```
PYTHONPATH=src python prototypes/chart-double-duty-166/read.py --ticks 30000 --seeds 42 43 44
```

Writes `166-<dome>-<split>-seed<seed>-<ticks>.json` per seed: the ladder, per-cell
arrays, and a by-level breakdown. Defaults match #274 — `real` dome, `split=train`,
both rules.

## Caveats

- **`p_v` is best-effort.** The private-width column is read defensively and falls back
  to zeros if the maps API does not expose per-edge widths; it is context, never the
  measurement. #271 and #274 are the authorities on `p_v`.
- **`rho_K` is `K` alone, not the loop.** #274's `rho_full` — `K ∘ (J_chart + J_stalk
  A_v D)` — is the authoritative retention number and includes the relay this does not.
  `rho_K` is reported because the *allocation across the twelve modes* is what this
  ticket is about, and that is a property of `K` itself.
- Seeds and horizon are smaller than #274's nine. Claims should be read as a range
  across the seeds run, per #274's own ruling that near `ρ = 1` a point value is the
  error itself.
