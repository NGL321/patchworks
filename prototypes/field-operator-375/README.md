# The field-level operator, and whether it is non-normal (#375 §2)

[#375](https://github.com/NGL321/patchworks/issues/375) §2 orders one read before any
coherent-structure hunt and **gates the hunt on it**. This is that read.

The argument it is answering, from [#374](https://github.com/NGL321/patchworks/issues/374)'s pass:
every source that explains how a linearly-stable driven medium holds structure locates the capacity
in **non-normality**, and in every one of them non-normality is a property of the **coupling
structure** rather than of a node's own operator. [#166](https://github.com/NGL321/patchworks/issues/166)
measured the cell operator `K` at **0.0504** on ADR-0023's instrument
`‖KᵀK − KKᵀ‖_F / ‖K‖_F²`, which leaves the cell overwhelmingly normal. Whether the *field* — the
restriction maps composed with the cell operators along the graph — differs has never been measured.
If it does not, there is nothing for a sustained structure to be made of, and the expensive read
#375 would otherwise run is looking for something that cannot be there.

## The operator

One tick is `inference_phase` then `message_passing_phase` (`src/patchworks/tick.py:754-853`). Write
`z` for the persisted charts and `s` for the predicting cells' node stalks. Linearised in the
activation region the run is in:

    z(t+1)  =  K (J_chart z(t) + J_stalk s(t))
    s(t+1)  =  A_v D z(t+1)  +  g_v Σ_{p∋v} F_pᵀ F_p̄ D z_{u(p̄)}(t)

with `A_v = I − g_v Σ_e F_evᵀ F_ev`, which is exactly [#274](https://github.com/NGL321/patchworks/issues/274)'s
relay. The first line is inference. The second is reconciliation, and its **second term is the whole
of the coupling** — one restriction out of the neighbour `u` and one back into `v`, which is the
composed object #375 §2 says nobody has measured.

**`(z, s)` is closed, and that is a fact about the unit delay rather than a modelling choice.** What
`v` reconciles against is `broadcast(t−1)`, and `broadcast(t−1) = F (D z(t) + b)` — because the
broadcast was formed from the prediction that *this* tick's chart decoded to. So a neighbour's chart
reaches `v` with no extra state carried, and the operator is `6600 × 6600` on the real dome
(150 predicting cells × (`k` = 12 chart + `n` = 32 evidence)) rather than the `17512` the raw buffers
would suggest.

**The comparator is the same operator with the coupling deleted.** Non-normality is not
basis-free, so a field number is worth only its comparison. The block-diagonal part of `M` — drop the
`F_pᵀ F_p̄` term, change nothing else — is the same graph with every cell talking to nobody, in the
same coordinates and at the same size. It contains #274's per-cell loop
`K(J_chart + J_stalk A D)` as its own diagonal. Comparing the field number to `K`'s `0.05` directly
would be wrong: the aggregate instrument falls like `1/√N` over `N` similar blocks (reproduced here —
the per-cell median `0.51` over 150 cells reads `0.51/√150 = 0.042` in aggregate), so **only the
matched-size comparison means anything**, which is why the comparator exists.

## Two checks, in the rig rather than the write-up

- **`check_broadcast_is_last_prediction`** — the identity the whole reduction rests on. It is
  **exact (0.0)** at construction. On a driven run it drifts to a median ~7% because the transport
  rule moves `F` and the prediction rule moves the biases between the tick that wrote the broadcast
  and the tick that reads it; that drift is the rules' own step size, the same shape as #274's relay
  identity going from `2e-7` at tick 1 to `~3e-3` after.
- **`check_one_tick`** — the assembled operator finite-differenced against an actual tick, both rules
  off. In float64 (`--float64`) it agrees to **6.1e-12**, falling monotonically as the perturbation
  shrinks, which is round-off and not a missing term. In the float32 the run is normally in it floors
  at **3.2e-4** at its best scale, for the ordinary reason that a float32 difference of order-one
  numbers has about that much left. An earlier version of this rig read **0.85** here, because it
  linearised at `prior_charts`/`prior_evidence` — the pair the *last* inference phase ran on, which
  is right for #274's backward-looking report and wrong for an operator being checked against the
  *next* tick. With a region-dependent `encode` that is not a rounding difference, and the check is
  what caught it.

## What it read

`real` dome, `split=train`, seed 42, both rules on, #206's ladder to 2000 ticks
(`375-real-train-seed42-2000.json`, 1042 s). `nn` is ADR-0023's instrument; `henrici` is
`departure / ‖M‖_F`; `coup%` is the coupling's share of the operator's Frobenius norm.

| tick | nn field | nn uncoupled | `K` (#166) | henrici field | henrici uncoupled | ρ field | ρ uncoupled | coup% |
|-----:|---------:|-------------:|-----------:|--------------:|------------------:|--------:|------------:|------:|
| 0    | 0.04624  | 0.04627      | 0.00000    | 0.9373        | 0.9375            | 1.5352  | 1.5352      | 2.82  |
| 100  | 0.04600  | 0.04603      | 0.00305    | 0.9361        | 0.9364            | 1.3686  | 1.3686      | 3.06  |
| 200  | 0.04601  | 0.04605      | 0.00327    | 0.9353        | 0.9356            | 1.5633  | 1.5632      | 3.22  |
| 500  | 0.04616  | 0.04622      | 0.00434    | 0.9360        | 0.9363            | 1.3368  | 1.3368      | 3.93  |
| 1000 | 0.04624  | 0.04634      | 0.00525    | 0.9354        | 0.9359            | 1.3394  | 1.3394      | 5.03  |
| 2000 | 0.04604  | 0.04624      | 0.00723    | 0.9346        | 0.9354            | 1.3634  | 1.3632      | 7.12  |

**The instrument is anchored, not just asserted.** `cell_K_166` is #166's own object measured by this
rig: **exactly 0** at construction (`K = a·I` is normal) and **0.00305** at tick 100, against #166's
reported **0.0031** at the first rung on seed 42. Same quantity, same ladder, same numbers.

**The finding: the coupling contributes no non-normality.** Field and uncoupled agree to three or
four significant figures on *every* instrument at *every* checkpoint — while the coupling itself
grows from 2.82% to **7.12%** of the operator's norm. Even the extreme eigenvalue is a single cell's:
ρ agrees to four figures, and a block-diagonal operator's ρ is just `max_v ρ(B_v)`.

**The transient amplification is real, modest, and already in the record.** `‖M^t‖₂ / ρ^t` peaks at
**2.66** (construction), **2.93** (100), **3.50** (1000), **3.94** (2000) — and the uncoupled
comparator peaks at 2.66 / 2.93 / 3.50 / 3.94. At `t = 1` this ratio is `σ_max/ρ`, which at
construction reads **2.63** against [#27](https://github.com/NGL321/patchworks/issues/27)'s measured
**2.62** — the number `DEFAULT_SAFETY_FACTOR = 2.6` is booked from
(`docs/registers/rig.md`). **The field buys nothing over what the record already books per cell.**

The curve *plateaus* rather than decaying, because ρ > 1 here: the field operator is expansive even
though #274 found the median cell contracting. Both are true — ρ of a block-diagonal operator is the
**max** over cells, so a handful of expansive cells set it.

**And the medium is dissipative, which is what makes the negative mean anything.** Per-cell block
radii, never as a graph-wide average (#181):

| tick | min | p25 | median | p75 | max | expansive |
|-----:|----:|----:|-------:|----:|----:|----------:|
| 0    | 0.524 | 0.817 | 0.929 | 1.064 | 1.535 | 52/150 |
| 500  | 0.474 | 0.819 | 0.901 | 0.970 | 1.337 | 27/150 |
| 2000 | 0.324 | 0.812 | 0.900 | 0.974 | 1.363 | 29/150 |

Median **0.900** — contracting, and consistent with #274's ~0.88 on the chart loop alone — with an
expansive minority that learning *shrinks*, 52/150 → 29/150. So this is a driven **dissipative**
medium with hot spots, not a broadly unstable one, and #374's literature is addressing the right
object. It also means the run leaves the region this operator was linearised in: **§2 rules out a
mechanism, not a phenomenon.**

**Scope.** One seed, one dome, to 2000 ticks. That is enough for a question about the *operator* —
which is what §2 gates on — and is not a long-horizon claim. The `broadcast` identity drifts to a
median 3–7% on a driven run (the rules' step size, as #274's relay identity does); the operator uses
the live maps either way, which is the record's standing convention.

## Reproducing

    PYTHONPATH=src python prototypes/field-operator-375/read.py --ticks 2000
    PYTHONPATH=src python prototypes/field-operator-375/read.py --dome small --ticks 100 --float64

`--float64` is the *verification* mode: it puts the run's own buffers in double so the
finite-difference check is a real one. Reported numbers come from the float32 run the rest of the
record was read on.

## Re-taken on the floored surface, and the negative replicates

Everything above was read on **2026-09-04 at 00:58Z**. [#434](https://github.com/NGL321/patchworks/pull/434)
merged the spectral floor into `RestrictionMaps.project()` at **15:52Z the same day**, so the reading
above is a reading of the **near-rank-1 surface** — the collapse
[ADR-0032](https://github.com/NGL321/patchworks/blob/main/docs/adr/0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md)
exists to end. #375's own reshape pre-registered that as a confound in this exact read, and asked for
it again afterwards: *"§2's field-level non-normality read is retained, still first, still the gate —
and taken after the floor lands it is worth more, not less, because it then reads a property of the
coupling structure rather than of the collapse."*

**This is that re-take.** Same rig, same seed, same dome, same ladder, same `--eigs --horizon 20`;
the only difference is the surface. `375-real-train-seed42-2000-floored.json`, 942 s. `compare.py`
prints the two side by side.

The confound was real. The coupling block this rig assembles is `g_v (F_pᵀ F_p̄)` — a one-hop
composed pair — and [#436](https://github.com/NGL321/patchworks/issues/436) measured exactly that
object moving under the floor: per-hop effective rank **1.108 → 2.153**, off-channel energy share
**0.051 → 0.381**, over 7,122 directed hops. The coupling's directional structure is precisely what
changed between these two runs.

| | | *before* | | | | *floored* | | |
|-----:|---------:|-------------:|----------:|------:|-|---------:|-------------:|----------:|------:|
| tick | nn field | nn uncoupled | gap | coup% | | nn field | nn uncoupled | gap | coup% |
| 0    | 0.04624 | 0.04627 | −3.05e-05 | 2.82 | | 0.04624 | 0.04627 | −3.05e-05 | 2.82 |
| 100  | 0.04600 | 0.04603 | −3.58e-05 | 3.06 | | 0.04592 | 0.04596 | −3.38e-05 | 2.97 |
| 200  | 0.04601 | 0.04605 | −4.00e-05 | 3.22 | | 0.04596 | 0.04600 | −3.73e-05 | 3.11 |
| 500  | 0.04616 | 0.04622 | −6.04e-05 | 3.93 | | 0.04610 | 0.04615 | −4.76e-05 | 3.50 |
| 1000 | 0.04624 | 0.04634 | −1.01e-04 | 5.03 | | 0.04602 | 0.04609 | −6.14e-05 | 3.96 |
| 2000 | 0.04604 | 0.04624 | −2.04e-04 | 7.11 | | 0.04632 | 0.04641 | −8.71e-05 | 4.67 |

**The finding survives, and it sharpens.** Field and uncoupled still agree to three or four
significant figures at every checkpoint. Henrici at 2000 reads **0.93473** field against **0.93537**
uncoupled; `ρ` reads **1.4254** against **1.4255**, four figures, which is again a block-diagonal
operator's `max_v ρ(B_v)`. Peak `‖M^t‖₂/ρ^t` is **2.798** field against **2.797** uncoupled.

**And the gap the coupling does open is smaller than before, not larger** — `−8.71e-05` at 2000
against `−2.04e-04` — while the coupling's share of the operator's norm grows to **4.67%** rather
than 7.11%. The sign is unchanged and worth stating plainly: the coupling makes the field operator
*very slightly less* non-normal than the same graph with every cell talking to nobody.

A smaller share is what a flattened pair predicts — `F_pᵀ F_p̄` for two near-rank-1 maps is
`≈ ‖F_p‖‖F_p̄‖·|cos∠|`, and spreading each map's energy over `m` directions shrinks the product
unless the carried subspaces are aligned, which [#437](https://github.com/NGL321/patchworks/issues/437)
records the transport rule driving *away* from. That is offered as consistent-with and not as a
measurement: this rig reports the share, not its decomposition.

**Construction is bit-identical across the two runs** (field `0.04623822224043648` on both), which is
the control it looks like: at tick 0 no learning has run, so `project()` has never been called and
the maps are the raw draw on either side of #434. It also anchors `DEFAULT_SAFETY_FACTOR`'s warrant —
peak amplification at construction is **2.659** in both runs, so #27's 2.62 behind the shipped 2.6 is
untouched by the floor.

**What this does and does not change.** It does not change §2's answer, and it removes the one
caveat that could have been raised against it — that the negative was an artifact of the collapse.
So it does not reopen the *constrain the instruments, do not close the ticket* ruling; it removes
that ruling's remaining exposure. **§2's own half-two stands untouched**: this is a local
linearisation, the run leaves the region it is read in, and it therefore forecloses a mechanism
rather than a phenomenon. The next action on #375 is still **instrument III, the drive-quench test**.

**Scope.** One seed, one dome, to 2000 ticks, exactly as above — enough for a question about the
operator and not a long-horizon claim. The floored run's `broadcast` identity drifts to a median
1.9–6.2%, the same band as before (3.4–6.3%), with the same occasional large max on a pair whose
denominator is small (24.6 here, 14.1 before).

    python prototypes/field-operator-375/compare.py
