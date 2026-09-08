# B21 (#570): the ripple test — how far does a perturbation reach, and does it stay distinguishable?

*Instrument: `prototypes/cold-start/T6/b21_ripple.py`. Raw: `570-ripple-ladder.json`
(the rock-size ladder that establishes the floor), `570-ripple-construction.json`
(4 arms x 4 modalities at construction), `570-ripple-<arm>-baseline-seed42-20000.json`
(4 arms trained to 20k, read at 1k / 5k / 20k for all four modalities), and
`570-ripple-interior.json` (the rock dropped inside the dome). Surface: branch
`worktree-b21-ripple-570` off `main` `13c015b` (B8's per-edge lanes, #558,
shipped); rig `T0`-`T6` taken from `worktree-b20-joint-span-569`.*

## The answer in one line

**A rock dropped at the rim reaches 2 to 3 hops into a graph whose every rim cell
is within 9 hops of everything, and it attenuates by a factor of 20 to 50 at each
one. Past hop 3 what is left is not the rock — it is float32 round-off amplified
to its own equilibrium, and it reads the same whether the rock is a hundred times
bigger or smaller. Magnitude and distinguishability do not die at different
distances; they die together, at the same hop, because there is nothing left at
that distance for either to be a property of. `p` does not move it, training does
not move it, and the dome's hierarchy does not shape it — a perturbation travels
no further up or down than it does laterally.**

The image the architecture exists to serve is *"drop a rock in a puddle and the
ripples reach across, distorted but global."* What this rig measures is a puddle
whose ripples die two body-lengths from the splash.

## 0. What was built, and what makes it trustworthy

The confound to kill is the world. `Agent.command` is a slice of the actuator
cell's node stalk, so a perturbed agent acts differently, the sandbox answers
differently, and the divergence that follows is the *world's* doing rather than
the graph's. So the world is put on a tape: the control run is executed live and
every `(observation, applied)` pair recorded; the perturbed arm then replays that
same tape by hand — `sheaf.tick()` then `agent.write(obs, applied)` — so the
exogenous input is bit-identical and the only difference anywhere in the system
is the rock.

Three things make the reading trustworthy, and each is measured rather than
asserted:

1. **The replay is exact.** The control put back through the replay path
   reproduces the live trajectory **bit-for-bit** (`state_max_abs_diff` 0.0,
   `command_max_abs_diff` 0.0) on every arm and every checkpoint. `replay_check`
   is in every record.
2. **A rock the maps cannot carry changes nothing at all.** A perturbation of the
   *same magnitude* drawn from the intersection of the null spaces of the rim
   cell's restriction maps produces a peak deviation of **exactly 0.0** at every
   interior cell and at the motor command, at every arm and checkpoint. `ker(F)`
   is closed in float32 too. This is a clean statement that the rig has no
   spurious coupling — and it is also why the sham could not serve as the noise
   floor.
3. **The floor comes from a rock-size ladder, so it is measured and not chosen.**
   `570-ripple-ladder.json` walks the rock over six decades. The near field tracks
   it decade for decade; the far field does not move at all.

   | `delta` | hop1 | hop2 | hop3 | hop5 | hop9 |
   |---|---|---|---|---|---|
   | 1e+00 | 6.0e-03 | 1.3e-04 | 8.2e-06 | 1.1e-05 | 3.6e-05 |
   | 1e-02 | 7.3e-05 | 1.5e-05 | 5.3e-06 | 9.3e-06 | 3.2e-05 |
   | 1e-04 | 8.4e-05 | 9.7e-06 | 8.1e-06 | 1.0e-05 | 3.0e-05 |
   | 1e-06 | **0** | **0** | **0** | **0** | **0** |

   Two things fall out of that table. **The far field is rock-independent across
   four decades** — it is the arithmetic, not the signal. And **a rock at 1e-6 of
   the cell's own state norm changes nothing anywhere, ever** — not a little, but
   exactly zero, because what it delivers at the first hop is below one unit in
   the last place of the receiving stalk.

   So the criterion for *"the rock reached here"* is **linearity, not magnitude**:
   a deviation is the rock only if shrinking the rock a hundredfold shrinks it
   too. Signal reads ~100 on that ratio; an arithmetic floor reads ~1.

**Learning is off during every reading.** The maps and the per-cell surface are
frozen at the checkpoint, so what is measured is propagation through the transport
that checkpoint learned, not propagation plus a moving target.

**It is an impulse, not a step.** `Agent.tick` is `sheaf.tick()` *then* the
world's write, and every sensorimotor rim stalk is world-written, so the injected
value survives exactly one message-passing phase before the world restores it.
That is the honest shape of dropping a rock: one splash, then watch. The
consequence is visible in the trace — the whole-graph deviation is exactly 0 after
the first replayed tick and first appears at tick 1, which is the unit delay in
`message_passing_phase` doing what it is documented to do.

## 1. The reach profile (item 1)

The dome: 414 cells, 682 edges, 150 predicting, 7 interior levels. **Every rim
cell has eccentricity 9 and reaches 413 of 414 cells** (the drive cell is the
exception). All 263 rim-to-apex chains are exactly 7 hops. Topologically, nothing
is far away.

Trained to 20k, seed 42, patch rim cell 0, median over 8 orthonormal rocks:

| arm | `k_v` | hop 1 | hop 2 | hop 3 | hop 4 | hop 5 |
|---|---|---|---|---|---|---|
| shipped | 25 | 9.7e-03 | 1.9e-04 | 7.9e-06 | 1.7e-06 | 1.8e-06 |
| `p=8` | 24 | 1.0e-02 | 3.2e-04 | 5.9e-06 | 2.0e-06 | 1.9e-06 |
| `p=16` | 16 | 1.4e-02 | 7.4e-04 | 3.0e-05 | 2.4e-06 | 2.1e-06 |
| `p=24` | 8 | 1.9e-02 | 4.6e-04 | 5.8e-06 | 2.6e-06 | 2.4e-06 |

and the linearity ratio on the same rows — the number that says whether it is the
rock:

| arm | hop 1 | hop 2 | hop 3 | hop 4 | hop 5 |
|---|---|---|---|---|---|
| shipped | 100 | 88 | **3.6** | 1.0 | 1.0 |
| `p=8` | 100 | 99 | **4.4** | 1.0 | 1.0 |
| `p=16` | 100 | 99 | **18.3** | 1.1 | 1.0 |
| `p=24` | 100 | 90 | **3.6** | 1.0 | 1.0 |

**The rock is unambiguously present at hops 1 and 2, marginal at hop 3, and gone
from hop 4 onward.** The flat ~2e-06 tail from hop 4 out to hop 9 is identical
across arms and identical across four decades of rock size: it is the float32
noise equilibrium, not a ripple.

**Attenuation is a factor of 20 to 50 per hop.** Compounded over the 7 hops of a
rim-to-apex chain that is 10 to 12 orders of magnitude — which is why the apex
never hears anything, and why no amount of operator rank at the apex could change
that. The rim cell's own state is the unit here: the rock is as large as
everything the cell holds, and by hop 3 what arrives is a millionth of the
receiving cell's own tick-to-tick motion.

## 2. Distinguishability (item 2)

Eight orthonormal rocks of equal norm are dropped at the same cell, each its own
replay. Per cell the eight deviation *trajectories* become an 8x8 Gram, read for
participation ratio (`d_eff`), mean pairwise `|cos|`, and minimum pairwise
separation against the measured floor.

**The ticket predicted a gap between the distance at which magnitude dies and the
distance at which distinguishability dies, and said the gap is the finding. There
is no gap.** On the trained arms both die at hop 2-3, together:

| arm | magnitude reaches | distinguishable to |
|---|---|---|
| shipped | hop 2 | hop 2 |
| `p=8` | hop 2 | hop 2 |
| `p=16` | hop 3 | hop 2 |
| `p=24` | hop 2 | hop 2 |

That is not distinguishability outliving magnitude, nor the reverse. Beyond hop 3
the responses have an `d_eff` of 2 to 3 and mean `|cos|` around 0.5, which looks
like healthy diversity until the linearity column is read beside it: **that
diversity is the round-off's, not the rocks'.** Eight rocks that differ by nothing
still perturb the arithmetic in eight slightly different ways. **A distinguish-
ability statistic read without a floor would have reported the far field as richly
informative, and it is empty.** This is the single most important methodological
point in the reading.

Where the rock genuinely does arrive, distinguishability is already poor:

* At hop **1** — the very first interior cell — the eight rocks arrive with
  `d_eff` between 1.9 and 3.2 of a possible 8, on the trained arms.
* The structural reason is visible and not subtle. The patch rim cell has a
  **stalk of 48 and exactly one incident edge of lane width 4**. Everything that
  cell can ever say to the rest of the graph passes through 4 dimensions, and of
  those 4 it effectively uses 2 to 3. The ceiling on how many distinct rocks could
  *ever* be told apart downstream is 4, before a single hop is taken.
* The proprioceptive rock is worse and it matters more, because it is the one on a
  short enough path to arrive: two orthogonal rocks reach the motor command with
  `d_eff` **1.007** (shipped), **1.000** (`p=8`), **1.004** (`p=16`), **1.000**
  (`p=24`) — of a possible 2. **They arrive as the same vector.** The command
  moves, and it moves the same way whichever rock you dropped.

This is precisely the ticket's *"a bell, not a medium"*: something happened, and
what happened carries no information about what was done.

## 3. Hierarchy (item 3)

Two readings, because the rim cannot answer this: from a rim cell every direction
is "up".

**Cross-modality, from the rim.** The patch rim cell is **6 hops** from the
actuator, the proprioceptive and the touch cells. Reach is 2-3. **A visual
perturbation never arrives anywhere near another modality.** The motor command's
deviation under a patch rock reads linearity **0.78 / 0.97 / 1.42 / 1.00** on the
four trained arms — one, on every arm, which is the signature of pure round-off.
Meanwhile the proprioceptive cell is 2 hops from the actuator and the touch cell
3, and both *do* move the command genuinely (linearity 92-102, deviation 0.6 to
14 times the command's own motion). **So what looks like cross-modal influence in
this architecture is entirely explained by graph distance: the modalities that are
close have effects, the one that is far has none.** Nothing crosses; some things
were never apart.

**Up against lateral against down, from inside the dome**
(`570-ripple-interior.json`, construction surface, 200 warm ticks — see caveats).
Dropping the rock at an interior cell and splitting the neighbourhood by level
relative to it:

| | hop 1 | hop 2 | hop 3 | hop 4 |
|---|---|---|---|---|
| **up** | 1.00 | 1.00 | 0.33-0.60 | 0.00 |
| **lateral** | 1.00 | 1.00 | 0.67-1.00 | 0.00-0.60 |
| **down** | 1.00 | 1.00 | 0.37-0.57 | 0.03-0.08 |

(share of cells at that hop whose deviation is the rock.)

**Reach does not respect the dome's hierarchy at all.** It is 3 hops in every
direction; if anything the *lateral* direction holds marginally longer than up or
down, which is the opposite of what a hierarchy that was doing work would show.
The dome's levels are not a gradient the signal runs along — they are a labelling
of a graph the signal decays through isotropically.

## 4. How both move with `p` (item 4)

**They do not.** Across `shipped` (`k_v` 25), `p=8` (24), `p=16` (16) and `p=24`
(8), trained to 20k, reach is 2-3 hops and distinguishability dies at hop 2 in
every arm. The construction sweep across the same four arms and all four
modalities says the same. `p = 16` buys one extra hop on the patch rock — reach 3
instead of 2 — and loses distinguishability a hop earlier on the actuator rock;
`p = 24` buys nothing at all.

This is the direct answer to what [B17](https://github.com/NGL321/patchworks/issues/565)
left open. **`p` does not shorten reach or kill distinguishability — but neither
does it buy any. Whatever `p` is doing to the operator's composed rank, it is not
touching what actually travels.** The two quantities are not in tension because
they are not connected: the operator reading is about which directions the maps
retain, and this reading is about a signal that has decayed by ten orders of
magnitude before it reaches the distance the operator reading is taken at.

## 5. What training does

Training strengthens paths that are already within reach and does nothing for
those that are not. The proprioceptive rock's effect on the motor command, as a
multiple of the command's own motion, on `shipped`:

| construction | 1k | 5k | 20k |
|---|---|---|---|
| 0.018 | 0.543 | 0.883 | 0.729 |

a fortyfold increase over training, on a 2-hop path. The patch rock's effect on
the command over the same checkpoints reads linearity 1.01, 1.27, 1.08, 0.78 —
round-off at every one. **Training makes the short reach louder. It never makes
the long reach exist.** And the distinguishability of what arrives does not
improve: the proprioceptive `d_eff` at the command is 1.02 at construction and
1.007 at 20k.

## 6. What was **not** measured

Stated as plainly as what was.

* **One seed (42) and one horizon (20k).** No seed spread anywhere in this reading.
  Given that every arm and every modality and every checkpoint agrees, a seed
  effect would have to be extraordinary — but it was not tested.
* **One rim cell per modality**, chosen as the lowest-numbered cell of that kind so
  the pick is not a knob. Patch cell 0 is one of 256; the reading does not
  establish that every patch cell behaves like it.
* **The interior reading is a construction reading**, and [B11](https://github.com/NGL321/patchworks/issues/555)'s
  warning that a construction reading is the wrong number to rule on applies to
  it. It is reported because the rim-side readings agree between construction and
  20k on every other axis, but the up/lateral/down table has not been taken on a
  trained surface.
* **Charts were not read.** The deviation is measured on node stalks only. A
  cell's persisted chart is private state that reconciliation never touches, and a
  perturbation could in principle live there longer than in the stalk. The stalk
  is what neighbours see, so it is the right object for a reach question — but
  "the effect died" is a statement about the observable, not about every variable
  in the cell.
* **32-tick horizon after the impulse.** The dome's eccentricity is 9 and one hop
  costs one tick, so this is ample for propagation; it does not rule out a slow
  accumulating effect over hundreds of ticks that this window cannot see.
* **No claim about what would fix it.** This is a reading. The instrument makes no
  architectural change and this readout proposes none.
* **The impulse is not a step.** A perturbation *held* at the rim across many
  ticks, rather than injected once, was not tried. It is a different experiment
  and might reach further; the one-shot impulse is what "drop a rock" describes.
