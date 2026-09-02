# ADR-0027: The demo's depth criterion is a conduction time, not an amplitude

**Status:** accepted

## Context

Settled in [#241](https://github.com/NGL321/patchworks/issues/241) and written here by
[#270](https://github.com/NGL321/patchworks/issues/270).

[`08-the-acceptance-demo.md`](../spec/08-the-acceptance-demo.md) measured depth as
[`05-timescales.md`](../spec/05-timescales.md)'s private-component readout: `‖Δ(private component)‖`
per cell against hop distance from the sensorimotor rim, passing when the trace **falls** with depth.

**That clause cannot fail.** The graph's own attenuation supplies a falling trace for free, and
[#214](https://github.com/NGL321/patchworks/issues/214) measured how much of it there is — a
bottleneck ratio of **8.7e-10** rim→apex, the hops graded **9x-240x**. A graph that transmits nothing
renders as one bright rim row above a flat baseline, and
[`private_component.py`](../../src/patchworks/surface/private_component.py) draws exactly that
picture at maximum contrast: `PrivateComponentPanel.draw` scales the marks by `moved / peak`, and on
this graph the peak is the rim's. (#241 cited this as `private_component.py:167`; the line is the
`Scatter.moved` field, and the normalisation it meant is in `draw`. Cited by name here so it does not
drift again.) The clause's
only named failure — *the traces are flat across hop distance* — is unreachable on this graph. An
unfailable pass condition in a pre-registered protocol is the defect
[#240](https://github.com/NGL321/patchworks/issues/240) is chartered against.

This is the demo's instance of a repair the map has already made once.
[ADR-0026](./0026-rim-core-influence-is-a-conduction-ratio.md) moved
[#127](https://github.com/NGL321/patchworks/issues/127)'s operative bar from amplitude to time for
the same reason: [#230](https://github.com/NGL321/patchworks/issues/230) diagnosed the deficit as
temporal, and every remedy family the map had opened was spatial. The demo inherited its amplitude
reading from an era the measurements have since left.

**ADR-0026 is the bar; this ADR is the demo's own calls against it.** The conduction ratio, its
derivation, `|loop(c)|`'s enumeration and the inbound/outbound asymmetry all live there and are not
restated here. What is decided here is what the *demo* does with them — which population it reads,
which sources it pokes, what a live single run can still falsify, and what the latency half is now
allowed to claim. Those are independent decisions with their own alternatives, which is why this is
its own ADR rather than a section of ADR-0026.

## Decision

### The depth criterion is the conduction ratio, read per event

Per cell, the e-fold decay time `τ̂_c` of the paired private-feature deviation, over `|loop(c)|`, with
the floor at

> `τ̂_c / |loop(c)| ≥ 1`

Passing, per event, is that **the event's loop closes**: the ratio holds along some path from the
event's injection site. Read **inbound** for the arm nudge (entering at proprioception and touch) and
the puck teleport (entering at vision); **outbound** for the retarget, which is injected at the drive
boundary cell attached to the apex.

Median over the 40 paired trials, with p05 / p25 / p75 / p95 alongside — ADR-0021's precedent and
[#202](https://github.com/NGL321/patchworks/issues/202)'s reason for it.

Four properties the swapped clause has and the old one lacked:

- **The failure is reachable, and it fires today.** `τ` is flat at about one tick graph-wide — 0.91
  at the apex against 0.99 at the rim (`05-timescales.md`, *What the live read says*), no
  depth→timescale gradient and slightly inverted. On this quantity the demo fails now, at every
  depth.
- **The floor is derived, not invented.** It is the loop's own length in ticks —
  [#143](https://github.com/NGL321/patchworks/issues/143)'s *"`τ` against the cell's own round
  trip"* — which satisfies the invented-constants rule the way ADR-0021 satisfied it in choosing
  `k = 1`.
- **[#181](https://github.com/NGL321/patchworks/issues/181) dissolves rather than needing an
  exception.** The index becomes `|loop(c)|`, a per-cell graph quantity. This was not a near-miss:
  [`CONTEXT.md`](../../CONTEXT.md) defines *Level* as indexed by hop distance from the sensorimotor
  boundary, so *"against hop distance"* **was** per-level indexing under the glossary's own
  definition.
- **The gate and the work become one quantity.** Stage 2 is retention. A stage-2 result now moves
  the demo's criterion instead of being invisible to it.

### Single-source, not a sweep

ADR-0026's inbound clause sweeps every rim cell and reports a fraction per stratum. The demo does
not. **It pokes the sources a human actually pokes** — one impulse at one joint, one puck teleport,
one retarget — and reads whether that event's loop closes. The sweep stays where it lives, on
[`benchmarks/detectability.py`](../../benchmarks/detectability.py).

The demo is a live interaction, and a swept read is not one. Nothing the sweep owns is lost: the
stratum fractions are the map's bar and are reported against ADR-0026, not against this protocol.

### The population is L1 predicting cells, and the partition is forced

**A boundary cell is not masked** (`graph.py:660`) — its stalk is world-shaped rather than
`n`-shaped — so it has no `H⁰` and cannot carry a conduction ratio at all. The demo's outbound clause
therefore **cannot** be a literal re-borrow of ADR-0026's outbound universal, which includes the
actuator boundary cell.

But the demo already reads the actuator, and reading it is what onset latency *is* — ADR-0026 calls
onset latency *"the acceptance demo's own instrument."* So:

- **The outbound population is L1 predicting cells.**
- **The actuator is read by onset latency**, the clause the file already had.

The two halves of the demo's evidence partition the graph with **no overlap and no gap**. This is why
the demo has two measures and not one, and it is a consequence of the mask rather than a choice about
emphasis.

### The between-event ordering is demoted to reported, not claimed on

`08` ordered the three events *arm nudge shallowest, puck teleport intermediate, retarget deepest*,
and that clause leaves the pre-registered pass condition. Two grounds, the second the stronger:

- **Both ends are supplied by the injection site.** `retarget()` writes the drive boundary cell at
  the apex; `disturb_arm` enters at the somatomotor rim. Neither clause needs the graph to do
  anything, in a world of any depth. `08` already half-knew this — *"two levels out of eight, and
  both supplied by the interface rather than discovered by the graph"* — but recorded it as an
  honest limit while still claiming on it.
- **The ordering was never accurate as a description.** The events do not differ in *which* level
  they address: **there is information modification at every level each event passes through.** What
  differs is the **deepest level each one reaches** — the arm nudge stops shallow, the puck teleport
  reaches intermediate, the retarget reaches the apex because the goal itself changes. *"Retarget is
  deepest"* describes where it was injected, not what it does.

The precedent is inside the same file: `08` already reports the arm nudge's onset without claiming on
it, *"so that the shallow rung cannot be attacked as a mechanical artefact."* The footage and the
narrative are unchanged; only the claim goes, and it was never testing the graph.

**The demoted clause is not merely deleted.** Under the conduction reading, *how deep an event
reaches* is the **length of the path along which the ratio holds** — the same observation with the
graph put back in it.

### The latency ordering is kept, with its interpretation corrected

Unlike the depth clause it can still fail: if nothing transmits, no corrective torque arrives, both
hands record the ceiling, the IQRs overlap, FAIL. Its defect was never unfailability — it was that a
PASS would be over-read. So:

> **A PASS on the latency ordering establishes that a correction travelled a longer path, not that a
> hierarchy produced it.** The hierarchy claim rests on the depth clause.

Demoting latency as well would leave the demo with one pass condition and no temporal reading at all,
which overcorrects a sound clause that had been overdrawn. **Onset latency remains the temporal
measure and the settling-time distinction is untouched.**

### Live falsifiability is kept in weakened and stated form

`05`'s *Demonstrating it* promised the readout is falsifiable live — *"if deep private state swings
with the rim, the mechanism is not working and it is visible in the moment."* A paired counterfactual
needs two forked runs; a human poking a live agent gives one. `restore` supplies the fork, so the
**establishing** measure moves to the 40 paired trials, and `05`'s live promise dies unless something
replaces it.

**The panel keeps `‖Δ private‖` as the picture and gains a live single-run `τ̂` read:** the excursion
above the agent's working baseline after an event marker, e-fold time from peak. No fork needed. It
is noisier than the paired version and confounded by the ongoing task in exactly the way ADR-0021's
floor describes — and it can still **fail in the moment**, because `τ̂` flat across depth is a flat
scatter.

This is `08`'s own split — *the live run demonstrates, the repeated runs establish* — applied to the
half that never had it. `10`'s refusal to drive the trail from `‖Δ private‖` is **untouched**; none
of this reaches the trail.

What goes from `05` is the sentence attenuation satisfies: *"the viewer watches the rim swing while
deep private state barely moves."*

### The two senses of "decay time" are separated explicitly

`08` and [#99](https://github.com/NGL321/patchworks/issues/99) both carry *never as settling or decay
time*. That guard is aimed at **onset latency** and at the body's mechanics — `03`'s timescale
ladder, 17.9x in passive joint decay ([#60](https://github.com/NGL321/patchworks/issues/60)).

`τ̂_c` is a decay time on **private features inside the graph**, which no joint can supply, and the
ladder is deliberately built **not** to align with the graph's levels. That misalignment guard, built
for the old clause, earns its keep a second time here. Both files must word the two senses so they
cannot be confused, or they will read as contradicting themselves.

### A threshold is admissible here, and the file must say why

`08`'s *"nothing sharper than non-overlapping IQRs is claimed… a ratio threshold would be a number
invented before anything was trained"* is about the **latency** half and about **invented** constants.
The bar is 1 and it is the loop's own length in ticks — derived, as ADR-0021 derived `k = 1`, and as
[#142](https://github.com/NGL321/patchworks/issues/142) struck the inherited ~0.37/hop for not being.
The rule is satisfied, not bent. Left unstated, a reader will think it was forgotten.

## Consequences

- **`08`'s pass condition is one closure and one ordering**, where it was two orderings: the loop
  closes per event, and the latency IQRs separate. The between-event depth ordering is reported
  alongside.
- **The confound register gains two entries** — the graph's own attenuation with depth, and unit
  edge delay. The first is what made the old clause unfailable. The second has **no fix on this
  graph**: every edge costs exactly one tick, so an onset ordering is a restatement of hop count and
  no guard separates hierarchy from delay. It is registered, not repaired.
- **#99 gains real work: `|loop(c)|` is computed nowhere in the tree today.** The shortest cycle
  through a cell that reaches the rim and returns is new graph machinery. ADR-0026's *"no new
  instrument"* was true of `benchmarks/detectability.py`'s deviations; it is not true of the loop
  lengths.
- **`10` owes its event marker to a second consumer.** The marker was already owed for onset; the
  live `τ̂` read needs the same marker as its `t = 0`.
- **The demo now fails today, and that is the point.** Reading `τ` flat at about one tick against
  `|loop|` of 2 at L1 and 14 at the apex, no loop closes anywhere. A pre-registered criterion that
  the current architecture fails is what pre-registration is for.
- **The pre-registration discipline survives intact.** This is a correction made *before* the run —
  the only time it can be made honestly — and it is why the writing was urgent rather than merely
  owed: pass and fail are fixed before [#100](https://github.com/NGL321/patchworks/issues/100)
  executes.
- **`05`, `08`, `10`, `CONTEXT.md` and #99's body move together**, in one pass. A pre-registration
  stated in two places that disagree is exactly the defect #240 exists to catch.

## Alternatives considered

- **A floor on the old amplitude quantity, rather than a swap.** Rejected: the deficit is temporal
  (#230), and a floor on an amplitude that attenuates by construction picks a number out of the
  attenuation curve. It also keeps the gate and the work different quantities.
- **Re-borrowing ADR-0021's bottleneck ratio as the demo's floor.** Rejected: it pre-registers a
  condition short by **1.3e9x** with no route to it, and an unreachable pass condition is as
  dishonest as an unfailable one. It would also re-import into the demo the magnitude reading
  ADR-0026 has just moved the map off.
- **Retiring the depth clause and resting on latency alone.** **Forbidden in terms.** `08` and
  `CONTEXT.md` both refuse behaviour alone as evidence, and onset latency is behavioural. It would
  also leave the demo unable to tell a hierarchy from a longer wire.
- **Re-borrowing ADR-0026's outbound universal verbatim, actuator included.** Rejected as
  impossible rather than unwise: a boundary cell is not masked, so the actuator has no `H⁰` to carry
  a ratio. The partition into predicting cells and the actuator is forced by the mask.
- **Running ADR-0026's swept inbound read inside the demo.** Rejected: the demo is a live
  interaction with a human poking specific sources, and a sweep is not one. The swept read is not
  lost — it is the map's bar and is reported against ADR-0026.
- **Dropping `05`'s live-falsifiability promise once the establishing measure went paired.**
  Rejected: it would leave the demo surface displaying a picture that can no longer contradict
  anything. Kept in weakened and **stated** form instead, with its noise and its confound named.
- **Folding this into ADR-0026 as a section.** Rejected: ADR-0026 defines the map's bar, while the
  actuator exclusion, the live/paired split and the single-source reading are the demo's own calls
  with their own alternatives — the ones above.
