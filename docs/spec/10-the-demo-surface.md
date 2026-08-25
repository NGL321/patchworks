# The demo surface

What a viewer actually sees. [`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md) owns the
protocol — which events, in what order, and what passes. This file owns the **display**: the windows,
what each mark encodes, what the human's hands are bound to, and the record a run leaves behind.

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

**Nothing in this file is part of the architecture.** The panel reads privileged state — prediction
error, private components, edge disagreement — on exactly the footing `03-the-sandbox.md` gives
`info`: for looking at, never fed back. No cell reads anything the surface computes, and switching
the whole surface off must change no trajectory.

## Two windows

| | |
|---|---|
| **Scene** | MuJoCo's passive viewer, as `prototypes/sandbox/watch.py` already runs it. Its camera, its picking, its drag. |
| **Panel** | A second window: the dome, the private-component readout, and the motor strip. |

A single composed window was considered and rejected. It buys one tidy frame for capture and pays for
it by re-implementing pick-and-drag — the exact interaction that already works — inside a renderer we
would own. Compose the tidy frame at **capture** time instead, from two recordings of a run that has
already happened.

The panel must be closable, and closing it changes nothing but the view.

## The dome panel

**Stacked bands: the sensorimotor boundary at the bottom, the apex at the top**, one band per level,
each drawn at its own lattice shape ([`06-graph-topology.md`](./06-graph-topology.md), *The levels*).

Depth reads as height. That is the whole reason for the layout: "recovered at the appropriate level"
becomes something a bystander watches rather than something a caption asserts, and the vertical axis
is already the axis [`05-timescales.md`](./05-timescales.md) asks its readout to be plotted against —
hop distance from the sensorimotor rim.

Concentric rings were the runner-up and are the **fallback if the dome is ever abandoned** — the dome
is explicitly abandonable and the shape-free construction rule has no bands to stack. They lose on
the dome: rings encode depth as radius, which a first-time viewer reads as distance instead.

A force-directed drawing is ruled out permanently. The construction layout is an **index, not an
embedding** (`06-graph-topology.md`); a spring layout would invent a geometry the topology explicitly
declines, and invite exactly the confusion with `01-cell-and-sheaf.md`'s charts that the index
existed to avoid.

### Colour is prediction error, normalised per cell

One primary channel, and it is prediction error: how wrong each cell currently is.

It is the encoding that makes the acceptance claim self-evident. Perturb the world and the shallow
bands light while the core stays dark; retarget and the core lights while the base stays dark. The
two hands of `08` produce visibly different pictures, which is the demo.

**Normalisation is per cell, against that cell's own running statistics.** Raw norms are not
comparable across the dome — the sensory funnel carries 12,288 numbers at the base and eight core
cells carry 32 dimensions each, so a raw map would show the taper's shape and nothing else, and
"which level lit up" would stop meaning anything. Surprise is deviation from a cell's *own*
baseline, and a per-cell scale is also what [ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md)
already declines to pin down globally.

Two consequences, accepted:

- **A chronically-wrong cell renders calm.** Correct for surprise, blind to chronic failure. The raw
  map stays available behind a debug flag, and a falsification sweep should use it.
- **The statistics need a warm-up.** A panel opened on a cold run shows noise until each cell has a
  baseline. The panel says so on screen rather than pretending.

### The trail

A cell's glow decays at **that cell's own measured persistence** — the estimate
[`05-timescales.md`](./05-timescales.md) already defines, not a second definition of timescale.

This costs one exponential per cell and renders the multi-timescale claim for free: rim cells go dark
almost immediately, core cells stay lit for hundreds of ticks. A wave of glow climbing the bands is
message passing, watched.

### The boundary band

**L0 draws the agent's own 64×64 render, tiled into the 16×16 patch lattice.**

Boundary cells run no body and make no prediction ([ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md)),
so they have no prediction error, and colouring them on that map would be a fabrication in the
largest, most eye-catching band on screen. The render costs nothing — it already exists every tick —
and it ties the abstract stack to the world: the picture at the bottom of the panel is the thing the
arm is doing in the other window.

### The somatomotor strip

Beside the tiled render, because that is where the cluster actually attaches — one region of the base,
not the vision lattice: **3 proprioceptive, 3 touch, 1 actuator**.

These are boundary cells too, so again no prediction error. What each one has is an **edge**, and edge
disagreement is drawn on the same colormap, honestly earned. Touch marks light on contact, which
gives a bystander the "it hit the puck" beat for nothing.

The actuator is one cell whose stalk the arm reads three components of, and it draws **decomposed**:
three paired bars, **commanded as an outline, applied as a fill**. This is
[`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md)'s efference copy made visible —
the cell writes back what was actually applied, post-clip and post-saturation, so saturation reads as
the fill falling short of its outline.

It also renders a falsification test rather than describing one. `04`'s route-selection signature is
the arm **stalling mid-swing** — near-zero commanded torque with standing motor-side disagreement —
and that is an outline near zero beside a disagreement bar that stands. `04` already assumes this
readout exists; here is where it exists.

**That bar is the one mark drawn raw rather than normalised**, and it has to be. Per-cell
normalisation makes a chronically-wrong mark render calm — accepted above for the colour channel,
where the raw map behind the debug flag is the answer — but a stall that *stands* is exactly a
chronic reading, and a bar that faded as the stall persisted would be the display absorbing the
failure it exists to show. So the three torque bars are drawn against the largest torque the strip
has seen and the disagreement bar against the largest a boundary mark has shown; zero stays zero
under both, which is what keeps *near-zero commanded* legible as near zero. Settled in #94.

**That last sentence is wrong, and the divisor it names is an open question.** Both scales are
running maxima, so both are ratchets that one tick can raise for the rest of the run, and #94's
review demonstrated each one erasing the signature this bar exists to render:

- *The largest a boundary mark has shown* is shared across every boundary mark, so a single tick in
  which the **drive** edge spikes permanently flattens the **actuator's** bar — 18 px to 1 px on the
  small dome, and the drive is already the largest boundary mark within 120 ticks of a real run. The
  two marks the panel most needs to read independently are coupled through one divisor.
- *The largest torque the strip has seen* is set by the **commanded** row, which
  `04-action-and-the-boundary.md` deliberately leaves unclipped. One spike to 500 draws every later
  full-strength command as a single pixel at the zero line — *near-zero commanded torque*, fabricated
  from a command at its limit.

The obvious repair — divide each mark by its **own** running peak — is not one: a mark sits at its
own peak whenever it is quiet, so the bar stands at 97% of full height through a completely quiet
run, claiming a standing disagreement at all times. A bar rendering an absolute claim needs an
absolute reference, and disagreement has no natural unit to supply one. **Unresolved; do not read
the height of the standing bar as a quantity until it is.** The pair of *torque* bars is unaffected
by this — a fill falling short of its outline is a comparison within one tick — and so is every mark
on the colour channel, which is normalised per cell and documented as such.

### The drive mark

The apex band carries a mark for the **drive boundary cell** at the internal rim, drawn like the
somatomotor strip: edge disagreement on the same colormap, no prediction error of its own.

Not decoration. `08`'s third named near-miss — **task-invariant behaviour**, where the trajectory is
the same across tasks differing only in the render — is diagnosed by the drive edge's disagreement
being *non-trivial* while behaviour does not vary. Without this mark that near-miss is
indistinguishable on screen from the demo working, and it is the one that falsifies the drive rather
than the demo ([ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md)).

**Which near-miss is read off the raw map, not the live one.** *Non-trivial while behaviour does not
vary* is a standing reading, and per-cell normalisation renders a standing reading calm — the
consequence accepted above. That is not a hole: diagnosing a near-miss is a falsification sweep, and
a sweep is already told to read the raw map. The live mark is what makes the drive *visible* at all
and what shows it arriving; the sweep is what reads it standing. Settled in #94.

### Edges: thresholded, and off by default

Only the edges carrying the most disagreement this tick are drawn, and the draw is a toggle.

Drawing all of them is a hairball at ~150 predicting cells over a sparse mask — it would grey out the
bands underneath and hide the thing the panel exists to show. Drawing none gives up the live **route**
through the graph: the trail shows that influence propagated, but not which cells were carrying it
during a reconciliation round. Thresholding shows the route as a route.

**The threshold is derived from the tick's own scale, never hand-set.** A fixed constant would make
the panel's picture of "the route" an artifact of the constant — the same objection
`05-timescales.md` raises against hand-set thresholds on the change gate, and it applies to a display
for the same reason it applies to a mechanism.

Default off, so the README capture stays clean and the live demo can turn it on.

## The private-component panel

A second panel, below the dome: `‖Δ(private component)‖` per cell, against hop distance from the
sensorimotor rim.

This is `05-timescales.md`'s readout and `08`'s **depth** measurement, and it is displayed during each
of the three events. The private component is the node-stalk directions masked out on every incident
edge, known at construction, so it is a fixed projection computed per tick.

**It stays off the dome's marks deliberately.** Folding it in as brightness or dot size beside
prediction-error-as-hue puts two quantities on one mark and makes neither readable — and
readability is the whole point, because this panel has to be able to *disagree*: if deep private
state swings as far as the rim, the mechanism is not working, and `08` counts that a failure even
when every recovery looks perfect. A scatter against depth either slopes or it does not.

For the same reason the trail is **not** driven by `‖Δ private‖`. That would make the display's decay
and the claim the display tests the same number, so the panel could never contradict the thesis.

## Onset, and the near-misses

`08` measures **onset latency**: ticks from an event to the first corrective torque. The surface owes
two things for it — an **event marker** the hands drop into the record when they fire, and a **tick
counter** on the motor strip running from the most recent marker. Onset is then read off the strip
rather than reconstructed afterward.

The commanded/applied bars already carry "first corrective torque", so no new quantity is needed.

The three near-misses `08` names in advance are each legible on this surface, which is what stops
convincing footage from passing: **restart** is a shallow-band-only picture with a home-pose
trajectory, **stall** is the outline-near-zero-beside-standing-disagreement signature above, and
**task-invariant behaviour** is the drive mark lit while the trajectory does not vary.

## The hands

| gesture | call | |
|---|---|---|
| ctrl-drag a **link** | `disturb_arm(joint, impulse)` | `08` event 1 |
| ctrl-drag a **puck** | `perturb(puck, xy)` | `08` event 2 |
| **right-click a puck, then click a zone** | `retarget(goal_puck, goal_zone)` | `08` event 3 |
| `r` | rearrange without resetting the arm | setup |

Ctrl-drag is MuJoCo's own gesture and is already bound; the **referent** decides which hand it is,
which is why events 1 and 2 need no new binding and read as one motion with two targets.

Retarget is the one that needed designing, because it has no world-side handle to grab. Click-then-
click is chosen over a keypress because it reads to a bystander with no caption — you point at the
thing, then point at where you now want it — and the third event of the demo is precisely the one a
viewer must understand without help. Number keys cycling the 3×3 pairs remain as the headless and
scripted path.

## The trace

**One renderer, over a tick record.** Live mode feeds it that record directly at ~10 Hz; replay mode
feeds it from disk. Not two code paths.

A tick holds `qpos`, `qvel`, `ctrl`, task, RNG — `03-the-sandbox.md`'s snapshot/restore contract,
unchanged — plus per-cell prediction error and `‖Δ private‖` (~150 cells × 2 floats), plus
**per-edge disagreement** (~700 floats) and the **actuator's commanded and applied rows** (six),
plus any event markers that fired. The trace is that contract plus those arrays, not a new format.

The last two are what every mark a **boundary cell** gets is drawn from — the somatomotor strip, the
actuator's paired bars, the drive mark and the edge overlay below — and they are in the record for
the reason the first two are: a quantity the panel could only be handed live would make the live feed
and the file two code paths, which is the split this section exists to refuse. Corrected here in #94,
which built those marks and found the record one array short of being able to draw them off disk.

**State, not frames.** The scene re-renders from MuJoCo offscreen at capture time, so capture
resolution is chosen when rendering rather than baked into the recording, and the README GIF, a
falsification sweep, and a debugging pass all read the same file. It also keeps the live budget near
zero: the env runs ~400 ticks/s with rendering on one laptop CPU core, and a 10 Hz panel decimated
off a state log does not threaten that.

## The front door

The README opens with **two short loops, side by side**, captioned *I moved the puck* and *I changed
the goal* — `08`'s events 2 and 3, scene and dome in each frame.

One loop cannot carry the thesis. The claim is not that lights move; it is that **where** they move
depends on what you did, and a contrast needs two pictures. Two loops make the architecture's central
claim legible to a visitor in ten seconds with nothing explained, which is the bar `08` sets for the
demo itself. The cost is one extra capture from a trace already on disk.

## Known exposure

- **The panel is not the toolkit.** Nothing here names how the marks are drawn. The panel consumes a
  tick record and draws marks, so nothing above was contingent on the stack decision
  ([`09-the-build-stack.md`](./09-the-build-stack.md)) and the drawing library remains a build-time
  choice. If that independence turns out to be false — if some encoding cannot be drawn at 10 Hz in
  whatever is chosen — the encoding is what gives, not the record it reads.
- **Per-cell normalisation hides chronic failure**, as above. The raw map exists; nothing enforces
  that anyone looks at it.
- **The surface displays two levels because the interface supplies two.** `08` states this limit for
  the protocol and it is inherited here verbatim: a panel showing eight bands does not thereby show
  eight levels being used, and a reader who mistakes band count for demonstrated depth has been
  misled by the picture rather than by any claim in it.
