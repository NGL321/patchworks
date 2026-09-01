# #183: does the drive reach, read along the channel?

`reach.py` re-reads the drive's reach **directionally**, the way
[#142](https://github.com/NGL321/patchworks/issues/142) re-read the taper.
[#120](https://github.com/NGL321/patchworks/issues/120) measured the drive's
assertion reaching *two levels from the apex*; #142 then showed that the
taper's whole per-hop budget had been read off isotropic probes against
near-rank-1 maps and was wrong by ~184x. **#142 did not re-read the drive**, so
until this ran the drive's reach was unverified in either direction — not known
to be nil, which is how it had come to be quoted.

## What #120 actually measured, and what it does not say

`benchmarks/untrained_fixed_point.py sensitivity` sets the drive boundary
cell to a different constant, re-asserts it after every external write, holds
the world still for 400 ticks, and prints **the largest absolute change** each
`(level, column)` group's node stalks show. Two things follow, and only the
first has been quoted:

1. It is **not an isotropic probe.** It moves the real assertion and reads the
   real response. The instrument artifact that cost the taper 184x does not
   apply to it in the same way, and this ticket does not get to wave it away.
2. It reports an **absolute displacement in float32** against node stalks of
   order 1-10 — its own footer says anything at 1e-6 is the representation's
   floor. A channel with a per-hop gain of ~4e-3 puts a unit assertion under
   that floor after three hops *whatever the gain is*. So "two levels from the
   apex" locates **where float32 stops resolving the signal**, not where the
   graph stops carrying it.

That second point is the half of #142's complaint that does transfer: *read
the gain rather than bounding it*.

## What this script does instead

Per hop, exactly [#142](https://github.com/NGL321/patchworks/issues/142) §3's
idiom — **two message-passing phases**, because a stalk moves the broadcast in
one phase and the broadcast moves the far stalk in the next, through the `t-1`
delay `02-tick-semantics.md` fixes — and **renormalised at every hop**, so what
accumulates is a product of gains that depth cannot push under the
representation's floor.

Three readings on the same chain, so the answer separates the channel from the
instrument:

| column | direction injected | what it stands for |
|---|---|---|
| `carried` | what the drive actually deposits, propagated | the true answer |
| `best` | the sender's own top right-singular vector | #142's ceiling — perfect cross-edge alignment |
| `iso` | a `randn` nudge | #120's equivalent reading, the control |

`body` is read along the carried direction too — perturb a cell's stalk, run
the inference phase, read its own slice back, since the prediction is written
onto the node stalk. That is the reading #142's pre-registered falsification 3
left owed, answered here for the drive's path.

### The drive's own hop is computed, not probed

The drive's node stalk is **one number** and every drive edge is **`m = 1`**
(`graph.py`: *"a map out of a one-dimensional stalk has rank at most one"*). So
the map out of it is a scalar, and the direction it deposits in an apex cell is
`F_apexᵀ` and nothing else. **There is no direction to choose, nothing was
averaged away, and no re-reading can improve it.** Whatever this hop measures
is what it is under every reading of the channel.

### The coherent read

The single-chain descent follows one cell to one neighbour, which is the right
shape for the taper — a rim perturbation starts at one patch. **The drive does
not.** It attaches to the apex *entire*: 8 edges, one scalar, every apex cell
driven with the same sign at the same instant, and `graph.py` says so — *"the
drive, at the apex level, entire... strength is fan-out, not width"*. Its
signal therefore arrives at level 6 as the sum of 8 contributions, and a probe
that follows one edge at a time cannot see whether those add or cancel.

So `coherent_descend` injects the **whole level's** displacement field at once
and reads what lands on the level below, entire, renormalising per level. Its
`iso` control randomises each cell's direction while keeping its magnitude, so
the ratio isolates **coherence** rather than direction.

## Running it

```
python prototypes/drive-reach/reach.py --dome full --ticks 1500          # untrained
python prototypes/drive-reach/reach.py --dome full --learn 30000         # #142's comparison point
python prototypes/drive-reach/reach.py --dome full --learn 100000        # where #178 says the surface settles
```

Teaching runs at ~31 s / 1000 ticks on one CPU, so `--learn 100000` is ~50
minutes. Recorded output is in `183-untrained.txt`, `183-taught-30k.txt` and
`183-taught-100k.txt`; `183-sensitivity-untrained.txt` is #120's own reading at
matching settings, kept beside them because the two instruments agreeing where
both can see is what makes the chained reading trustworthy where only one can.

## Results

| | untrained (1500) | taught (30k) | taught (100k) |
|---|---|---|---|
| drive -> apex, per apex cell | 0.0501 | 0.1000 | 0.0999 |
| per-hop down the taper, `carried` | 0.0037 | 0.0148 | 0.0192 |
| per-hop, `iso` (#120's reading) | 0.0029 | 0.0110 | 0.0105 |
| per-hop, `aligned` (#142's reading) | 0.0102 | 0.0909 | 0.1018 |
| along the channel is worth | 1.26x | 1.35x | 1.84x |
| the drive's share of the aligned reading | 36.3% | 16.3% | 18.9% |
| coherent fan-in worth | 0.96x | 0.88x | 0.94x |

**#120 survives; the phrase it is quoted as does not.** The drive reaches the
apex at 0.100 per hop and #120's own table always showed it — 1.04 at `L7core`.
Read along the channel the drive's hop is worth **1.84x**, not 184x: this
measurement was never isotropic, so #142's correction had nothing to bite on.
What is wrong with *"two levels from the apex"* is that it locates float32's
resolving limit rather than the graph's. **Every hop measured has a non-zero
gain.**

**The new finding: the transport rule builds a channel and the drive is not on
it.** The aligned reading rises 10x over 100k ticks; the drive's carried
direction rises 5.2x; its share falls from 36% to 19% and stays there. The
drive edge has one degree of freedom and no direction to choose, so there is
nothing there for the transport rule to align — **the one place in the graph
where #142's remedy structurally cannot apply.**

**The drive's own hop is settled**, 0.1000 at 30k against 0.0999 at 100k, which
is the check [#178](https://github.com/NGL321/patchworks/issues/178) earned by
finding the disagreement floor wandering 3.8x with 30k on a local high. This
quantity does not wander.

Two readings not to over-trust, both recorded on the ticket: `aligned` is the
*sender's* best direction and so is not a strict upper bound on a chained hop
(at 30k's last hop it falls below `carried`), and at 100k the aligned column
pins to 0.1666 on four consecutive hops — `1/6` to four digits, noted and not
chased.

Full reading, and what it does to the map's *standing diagnosis*, on
[#183](https://github.com/NGL321/patchworks/issues/183).
