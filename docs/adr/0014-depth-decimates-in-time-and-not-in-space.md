# ADR-0014: Depth decimates in time and not in space

**Status:** accepted

## Context

Settled in [#130](https://github.com/NGL321/patchworks/issues/130), and written down with the graph it
governs in [`11-the-language-graph.md`](../spec/11-the-language-graph.md).

[`05-timescales.md`](../spec/05-timescales.md) has a section titled *Depth does not supply it*, and it
is stated without qualification:

> **Unit delay is a phase shift, not a decimation.** It moves a signal in time; it removes no
> frequency content from it. […] A cell ten hops out has exactly the same input bandwidth as one at
> the rim; it is merely looking at older data.

The one ratio depth buys is **latency**, which `05` calls the wrong one. That argument is the
foundation of the whole document: it is why timescale has to come from persistence rather than from
the taper, and it is what ADR-0005 is a decision about.

**The argument is about spatial pooling, and it is sound there.** In the dome, a cell at L2 covers a
4×4 block of patches, all of which are written by the world on the *same* tick. Pooling them removes
spatial detail and no temporal detail at all: the cell still receives a fresh message every tick,
forever.

The language rim is not like that. Its one axis **is** time. A cell covering four buffer slots covers
four *ticks*, and the value it holds changes once per four ticks' worth of new information rather than
once per tick. That is a genuine decimation — the thing `05` says depth cannot do — and nothing in the
record noticed the distinction, because until now there was only one domain and in that domain the two
readings coincide.

This matters beyond bookkeeping. Left unstated, the wedge would look like a refutation of `05`, or —
worse and more likely — like a free win: a taper that supplies timescale for nothing, in a project
whose central open claim is that timescale comes from persistence.

## Decision

**Depth's relation to timescale is domain-scoped. In a spatially-indexed rim, depth is a phase shift
and supplies no timescale. In a time-indexed rim, depth decimates and supplies horizon directly.**
`05-timescales.md`'s *Depth does not supply it* is scoped to the first case rather than repealed.

Three things follow, and they are one decision rather than three because they all fall out of the same
construction.

### 1. The wedge is built in two regions, with two meanings of depth

A fast rim taper where depth is **temporal extent**, then a core where depth is **abstraction**.
Neither meaning is borrowed by the other region, and the boundary between them is the merge level.

**This is what stops horizon and abstraction collapsing into each other.** A single dilated stack over
the buffer would make them the same axis: a cell would be more abstract exactly insofar as it looked
further back, which is a claim nobody has made and which the sandbox explicitly does not hold — there,
abstraction is hop distance from the sensorimotor base and horizon is something persistence supplies.
Under the two-region shape the sandbox's distinction survives in the second region unchanged, and the
first region is a *pre*-processing taper whose depth is not abstraction at all.

### 2. The taper is `05`'s instrument, and it costs a piece of evidence

`05-timescales.md` builds an explicit clock divisor "**first, and not as a fallback** […] the
instrument that establishes the capability depends on timescale at all", then switches it off to see
whether persistence reproduces the behaviour. **The wedge's dilation is that instrument expressed as
topology.** It is legitimate and it is useful.

It is also a **forfeit, stated in the open**, on exactly the precedent `05` sets for its own banding:
the depth/timescale correspondence in the language domain is **built rather than found**, so observing
that deep cells hold longer-horizon content there **cannot be cited as the persistence mechanism
working**. Persistence remains the mechanism under test, and it is under test where it always was — in
the core, where depth is abstraction and no decimation is happening.

The prohibition that keeps the two interchangeable is untouched: **nothing in the architecture reads a
cell's timescale**. The taper is topology, and no mechanism branches on it.

### 3. The column merge is the same construction rule, so it lives here

Heard and spoken share no cell until the first core level, exactly as vision and proprioception share
none until L3. The shortest heard→spoken path is **six hops, about seven ticks**.

This is not a separate decision from the two-region split; it is the same parameter read along the
other axis. The merge level is what sets both where abstraction begins and how far the agent's own
output is from its ears. **Echo is designed out by construction**, and the merge level is the knob to
pull if echo turns out wanted rather than feared.

## Consequences

- `05-timescales.md`'s *Depth does not supply it* now carries a scope. It is not weakened in the
  sandbox, where its argument was always the right one.
- **The language domain gives up one piece of evidence for persistence** and keeps the mechanism. A
  reader who finds long horizons in a deep wedge cell has found the taper, not the thesis.
- **The wedge's taper is shallow on purpose.** Two levels, 4:1, and the depth lives in the core. A
  taper deep enough to be interesting as a horizon mechanism would be deep enough to confound the
  core's meaning of depth, which is the collapse this ADR exists to prevent.
- **The longest span a single cell covers is 16 ticks**, so long-range dependency is pushed into the
  core where it competes for the private dimension slow state needs. That is the wedge's known failure
  mode, it is measurable on `05`'s existing per-cell private-component readout against hop distance,
  and it is recorded rather than fixed.
- ADR-0009 is untouched: the wedge's core has an apex, its guaranteed private dimension is 15 as in the
  dome, and the drive attaches identically.

## Alternatives considered

- **One dilated stack over the buffer, all the way to the apex.** The natural reading of "a wedge", and
  it has the better precedent — a dilated causal convolution is a known-good sequence architecture.
  Rejected because it makes horizon and abstraction the same axis, and because a dyadic stack over 128
  slots gives 255 predicting cells with no argument for any of them, which is the count
  `06-graph-topology.md` already refused.
- **Overlapping dilation rather than disjoint pooling.** The precedent's own arrangement. Rejected
  because disjoint pooling makes the **straddle** load-bearing — no cell covers a span crossing a
  pooling boundary, so laterals repair it, which is the mechanism this architecture is about — while
  overlap covers straddles directly at higher degree, and degree is what `Σ_e m_e` charges private
  dimension for.
- **Treating the wedge as a refutation of `05` rather than as a scoping of it.** Rejected because the
  argument in *Depth does not supply it* is correct wherever pooling is spatial, and the dome is
  entirely spatial. Nothing about the sandbox changes.
- **A directed heard→spoken edge to shorten the loop while keeping the columns apart.** Rejected as a
  category error rather than on cost: an edge only one end restricts into has nothing to disagree
  about. See *A one-way edge is not a sheaf edge* in
  [`01-cell-and-sheaf.md`](../spec/01-cell-and-sheaf.md).
- **Asymmetric masking on a shared edge**, leaving the spoken side free. Legal under the contract,
  since the masks at an edge's two ends are independent. Declined as hand-specified content
  restriction, which is what `06-graph-topology.md` rejected when it turned down cross-modal laterals
  at L2 as "hand-specified wiring for something that should follow from position".

## Falsification

**If the agent converges on echo, the merge is too shallow.** In the register's shape
([#147](https://github.com/NGL321/patchworks/issues/147)), and observable on the emission entropy
readout [#129](https://github.com/NGL321/patchworks/issues/129) already specifies for the neighbouring
degenerate fixed point.

**If the core's cells show no longer horizon than the taper's**, the second region is not doing the
work this ADR assigns it, and the two-region split has bought a scoping argument and nothing else.
Readable on `05-timescales.md`'s per-cell private-component readout against hop distance, which is the
same instrument the failure mode above uses.
