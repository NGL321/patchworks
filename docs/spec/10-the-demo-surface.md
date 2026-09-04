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
| **Scene** | MuJoCo's passive viewer, as `prototypes/sandbox/watch.py` already runs it. Its picking, its drag, and its camera — held top-down, for the reason *The hands* gives. |
| **Panel** | A second window: the dome, the private-component readout, and the motor strip. |

A single composed window was considered and rejected. It buys one tidy frame for capture and pays for
it by re-implementing pick-and-drag — the exact interaction that already works — inside a renderer we
would own. Compose the tidy frame at **capture** time instead, from two recordings of a run that has
already happened.

The panel must be closable, and closing it changes nothing but the view.

### What to run

Both windows, live, with the hands bound to the scene one — and the same panel from a saved trace:

```bash
mjpython -m patchworks.surface.watch --ticks 2000 --save run.npz
python   -m patchworks.surface.watch --replay run.npz
```

`--pitch` sizes a lattice slot, `--scale` sizes the window, `--edges` and `--raw` are the two toggles
this file already describes, and `--fps` paces a replay. `mjpython` is MuJoCo's requirement for the
passive viewer on macOS and applies to the live half only; a replay opens no scene window and runs
under plain `python`. Built in #122, **ahead of** #119 rather than on top of it: the dispatcher that
was to own the command does not exist yet, so this ships its own `argparse` entry point with the
parsing and the doing in separate functions. When #119 lands, `patchworks watch` becomes its
subcommand by calling `patchworks.surface.watch`'s `live` and `replay` directly.

**The two windows are two processes, and on macOS they must be.** Under `mjpython` no Python thread is
the Cocoa main thread — the launcher's own docstring says it runs the interpreter on a separate thread
to leave the main one free for Cocoa — and Cocoa refuses to make a window off that thread, aborting
the process out of GLFW rather than raising. Off `mjpython`, `glfw.init()` on a second thread does not
return. So the panel is drawn by a child process that is its own Cocoa main thread, fed one uint8
frame at a time down a pipe. Each of those was measured on the reference laptop in #122, and
`src/patchworks/surface/window.py` is where they are written down.

That split buys the rest of this section for free: the panel holds no agent, the pipe carries pixels
one way, and a child that goes away is a closed window and nothing else. On top of it, **each end
holds a mailbox of one frame and drops rather than queues**, so the run never waits on the display —
a window being dragged, which blocks that process's event loop on macOS for as long as the mouse is
down, costs frames and not ticks.

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

**The two bars beside each other are scaled differently, and they have to be.** One has an absolute
reference available and the other does not.

**The three torque bars are absolute, against the action space's own bound.** The sandbox declares
`spaces.Box(-1.0, 1.0, (nu,))` and `Agent.act` clips the command to it before the arm reads it, so
the *applied* row is bounded by that on every tick of every run. Drawing both rows against it is what
makes `04`'s *near-zero commanded torque* an absolute claim — near zero means near zero, always, with
no dependence on what the run has happened to see. The commanded row is deliberately unclipped, so it
can ask for more than the bound; the bound draws to one row short of the bar's reach and the overrun
takes the row that leaves, so a saturating command stands one row above the fill that met the bound
beside it. That is *the fill falling short of its outline*, drawn. Note this is the only shape
saturation can take: the rows can differ only once the command has passed the bound, and when they do
the applied row is exactly *at* it.

**The disagreement bar is relative, and shared: each boundary mark against the largest of them on the
tick being drawn.** Disagreement has no natural unit, so no absolute reference exists to supply, and
comparison between the marks is what this bar is for. Sharing one divisor is what makes them
comparable; recomputing it every tick is what stops it being a ratchet. **Its height is therefore a
comparison and never a quantity** — some mark is at full height on every tick, and under a relative
reading that is the reading rather than a defect. Do not read it as a magnitude.

Both replace an earlier attempt to drive these bars from running maxima, which #94's review showed
erasing the signature they exist to render: one tick in which the drive spiked flattened the
actuator's bar for the rest of the run, and one unclipped command drew every later full-strength one
at the zero line. The obvious repair — each mark against its *own* peak — is worse, since a mark sits
at its own peak whenever it is quiet and the bar would then stand through a silent run. Ruled on in
#94.

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

This is `05-timescales.md`'s readout, and it is displayed during each of the three events. The private
component is the node-stalk directions masked out on every incident edge, known at construction, so
it is a fixed projection computed per tick.

**The panel has a second reading, and it is the one `08` passes on.** `‖Δ private‖` against depth is
the *picture*; `08`'s depth criterion is the **conduction ratio**, `τ̂_c / world_loop(c) ≥ 1`
([ADR-0027](../adr/0027-the-demos-depth-criterion-is-a-conduction-time.md); the divisor was
`|loop(c)|` until [#383](https://github.com/NGL321/patchworks/issues/383) moved it). The amplitude scatter
was the criterion and is not any more, for a reason this panel's own design already knew: the
channel's attenuation supplies a falling scatter whether or not anything is retained, so the picture
and the claim were the same number. So the panel draws **two** scatters against depth, both per
cell — `‖Δ private‖`, the picture, and **`τ̂`**, the quantity that decides.

- **The `τ̂` scatter is the live single-run read.** The excursion above the agent's working baseline
  after an event marker, e-fold time from peak. No fork, so no `restore` and no paired branch — it is
  what a viewer poking a live agent can be shown. It is noisier than the paired version `08`
  establishes on, and confounded by the ongoing task; `05`'s *Demonstrating it* states both.
- **Draw the bar.** `τ̂ = world_loop(c)` is a line on the panel, per cell, because the criterion is a
  comparison and a scatter without its bar cannot be read against one. Points above the line are
  cells whose loop closes. **The line moved with the divisor**
  ([#383](https://github.com/NGL321/patchworks/issues/383)) and it moved *up* at every cell — from a
  flat 2 to 3–9 at L1 and from 14 to 15–16 at the apex — so the panel is drawing a harder bar than
  the one it was specified with. It is also **no longer flat within a level**: `|loop(c)|` was one
  exact value per level, and `world_loop(c)` is a range whose per-level bands overlap, so the bar is
  a per-cell line and cannot be drawn as one step per depth column.

**It stays off the dome's marks deliberately.** Folding either scatter in as brightness or dot size
beside prediction-error-as-hue puts two quantities on one mark and makes neither readable — and
readability is the whole point, because this panel has to be able to *disagree*: **`τ̂` flat across
depth is a flat scatter**, sitting under the bar at every depth, and `08` counts that a failure even
when every recovery looks perfect. A scatter against depth either clears its line or it does not.

For the same reason the trail is **not** driven by `‖Δ private‖`, and this is **untouched** by the
criterion moving. That would make the display's decay and the claim the display tests the same
number, so the panel could never contradict the thesis. Nothing above reaches the trail.

## Onset, and the near-misses

`08` measures **onset latency**: ticks from an event to the first corrective torque. The surface owes
two things for it — an **event marker** the hands drop into the record when they fire, and a **tick
counter** on the motor strip running from the most recent marker. Onset is then read off the strip
rather than reconstructed afterward.

The commanded/applied bars already carry "first corrective torque", so no new quantity is needed.

**The counter terminates.** `08` ceilings onset at **100 ticks** (*What disqualifies a snapshot*),
so the strip's counter stops there and says it stopped rather than running on: a trial that reaches
the ceiling is a non-recovery the protocol keeps, and a counter still climbing past the value the
record will hold would show the viewer a number the run does not report.

**The event marker now has a second consumer**, and it is the private-component panel's `τ̂` read
above: the live single-run `τ̂` is measured from the marker, which is its `t = 0` for the excursion
above baseline. One marker, two readers — the marker is not duplicated and the two must not drift
apart, because a `τ̂` timed from a different origin than the onset it is displayed beside is two
clocks on one panel.

The three near-misses `08` names in advance are each legible on this surface, which is what stops
convincing footage from passing: **restart** is a shallow-band-only picture with a home-pose
trajectory, **stall** is the outline-near-zero-beside-standing-disagreement signature above, and
**task-invariant behaviour** is the drive mark lit while the trajectory does not vary.

## The hands

| gesture | call | |
|---|---|---|
| ctrl-drag a **link** | `disturb_arm(joint, impulse)` | `08` event 1 |
| ctrl-drag a **puck** | `perturb(puck, xy)` | `08` event 2 |
| **left-double-click a puck, then left-double-click a zone** | `retarget(goal_puck, goal_zone)` | `08` event 3 |
| `r` | rearrange without resetting the arm | setup |

Ctrl-drag is MuJoCo's own gesture and is already bound; the **referent** decides which hand it is,
which is why events 1 and 2 need no new binding and read as one motion with two targets. It is the
*translating* ctrl-drag — MuJoCo's `mjPERT_TRANSLATE`, ctrl + the right button. Ctrl + the left
button is `mjPERT_ROTATE`, which turns a reference orientation and moves nothing, so it fires no
hand.

**A gesture is planar, and an out-of-plane drag fires nothing — corrected in #123.** The world has
zero z degrees of freedom (`03-the-sandbox.md`; `arena.xml` has hinges about z and slides in x and y,
and no z slide anywhere), while MuJoCo hands over a drag in three dimensions. Taking the planar
shadow of one was the bug reported: a drag that was mostly z with a millimetre of planar residue
passed the minimum-drag gate and fired a hand with the residue as its argument — a puck teleported a
millimetre in a direction the human never expressed.

**Where the z comes from is the mouse, not the camera**, which is what #123 assumed and the
implementation had to correct. MuJoCo's plain ctrl-drag is `mjMOUSE_MOVE_V`, a translate in the
*vertical* plane: pull across the screen and the grabbed point moves in the world's xy, pull up the
screen and it moves in world z — **at every camera elevation, straight down included**. The shifted
drag, `mjMOUSE_MOVE_H`, is planar whichever way it goes; MuJoCo's own help table says as much in one
line, *Object Translate: Ctrl [Shift] right drag*. `tests/test_gestures.py` pins both against
`mjv_movePerturb` itself, because the wrong reading is easy to hold and hard to notice.

So there are two constraints, they do different jobs, and **both are kept**:

- **The drag is refused, and this is the enforcement.** The gate is on the out-of-plane component
  measured against the planar one — about six degrees — not on the planar magnitude alone. A drag
  outside it fires no hand, leaves no marker, and **warns**, naming the shifted drag as the way to
  express a planar pull. A human given nothing and told nothing is left wondering which of the two
  happened, and "look from above" would have been advice that does not help. The warning is for a
  pull that named a link or a puck: a drag that named the table, a wall, or nothing is a
  **miss**, fires nothing as it always did, and says nothing — blaming the pull for what was an aim
  would teach the human the wrong lesson about their own gesture.
- **The camera's tilt is held straight down**, re-asserted every tick rather than set once at
  startup, and what it holds is the **picture**: the arena's plane fills the screen, so a drag's
  planar half is the motion the human watches themselves make. It is re-assertion and not
  prevention — the passive viewer offers no way to disable its own mouse camera, so a rotation lasts
  until the next tick, 20 ms, and is then put back. Only the tilt is held; panning, zooming and
  spinning leave the plane square to the screen and stay the human's.

The table above still names the ctrl-drag, unchanged: **which of MuJoCo's two translates the demo
should teach** — the plain one, which fires only when it is pulled across the screen, or the shifted
one, which is planar in every direction — is a binding question #123 did not settle, and is left open
rather than decided in passing.

#123 ruled for both constraints, for the redundancy, and **neither is to be removed as redundant with
the other** — but the division of labour is the one above, and the camera hold must not be read as
enforcing the plane, because it cannot.

Both relax in one place — the tolerance the gesture layer is built with, and the loop's
`hold_camera` — and **neither relaxation is a third dimension**. Both hands take xy, so lifting the
tolerance stops the refusing and restores the planar shadow this ticket was filed on. A world whose
pucks can move in z changes the hands; these seams are only what would otherwise be welded across
its path.

**The perturbation ghost stays three-dimensional.** MuJoCo draws the drag as a spring in 3D and
nothing on this surface can make it do otherwise. Held top-down that is mostly harmless — the planar
half of a drag is drawn where the puck would go — but an out-of-plane pull is drawn going somewhere,
towards or away from the camera, and then fires nothing at all. That gap is MuJoCo's and cannot be
closed here; the warning is what stands in for closing it. Whether the refusal, the message and the
un-tiltable view read well at the window is, like the double-click below, a claim only a human at the
window can settle.

Retarget is the one that needed designing, because it has no world-side handle to grab. Click-then-
click is chosen over a keypress because it reads to a bystander with no caption — you point at the
thing, then point at where you now want it — and the third event of the demo is precisely the one a
viewer must understand without help. Number keys cycling the 3×3 pairs remain as the headless and
scripted path.

**The button, corrected in #116.** This section named a right-click until #96 found that a raw
button is not observable through `mujoco.viewer.launch_passive` at all: it offers one event hook,
`key_callback`, and one piece of readable mouse state, `MjvPerturb`, which has no button field.
What that struct *does* report deterministically is three acts — a **selection**, written only on a
**left** double-click (any other button moves the camera's `lookat` or starts it tracking); a
**translating** ctrl-drag, the one act that moves a point to a place; and a **rotating** ctrl-drag,
which carries an orientation and no place. The shift key is not a field either, and it is what
chooses the plane the translating drag moves in — see the planar note above.

So a pointing is a left double-click, and retarget is two of them. A **drag** from puck to zone
would read more plainly still, and was preferred if it could be had, but it is closed rather than
declined: the only drag that reports where it ended is the translating one, and a translating
ctrl-drag on a puck already means `perturb`. Two hands cannot fire from one gesture on one referent,
and the drag the viewer can tell apart from it carries no position to name a zone with.
Re-implementing picking — authorised in #116 for this gesture alone — was therefore not needed, and
the scene window stays MuJoCo's passive viewer with its picking and drag inherited entire.

What this section argued for is *point at the thing, then point at where you now want it*, and that
is met exactly. Whether two double-clicks read as well to a bystander as the single click imagined
here is the one claim on this page that only a human at the window can settle.

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

**Every record is one tick's reading, and a trace holds no record that no tick produced** (#116).
The consequence is a marker fired on a run's *last* tick: markers ride on the next capture, so the
last one has no capture left to ride, and it stays where the recorder can still show it rather than
reaching the trace. Nothing is mis-recorded — a marker carries the tick it fired on, not the tick it
was captured with — and what is lost is the last event, in the trace only. The alternative was a
fabricated trailing record, and there is nothing honest to put in its arrays: zeros read back as a
graph that agreed on every edge, and *not captured* is a shape the rest of the trace disagrees with.
**So a run that must not lose its last event declares one tick more than it measures** — one
iteration for the marker to be yielded on, at the cost of one tick and no semantics. That is the
falsification sweep's obligation, alongside restarting the onset counter once per trial, and neither
is checkable from inside the surface.

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
- **A replay reconstructs the trail's time constants rather than reading them.** A trace is state and
  the record's arrays, deliberately — the section above says so — and a measured persistence is a
  number about the *biases*, which a trace does not hold. So `--replay` measures a sheaf of its own
  and `--seed` is what makes that the run's own body. Every other mark on both panels comes off the
  record and is exact; this one is a reconstruction, and a replay on the wrong seed draws a real run's
  marks with another body's decay.
- **On the untrained body the trail is not visible at all**, which is a fact about the body and not a
  defect in the encoding. Measured in #122 on the default spec, seeded: every predicting cell's
  measured persistence is between 0.5 and 1.7 ticks, and a capture is one tick in five, so a glow is
  down to at most 6% of itself by the next frame. *Rim cells go dark almost immediately, core cells
  stay lit for hundreds of ticks* is what the trail renders **once the body passes
  [`05-timescales.md`](./05-timescales.md)'s go/no-go**, and until then the panel is drawing an honest
  picture of a body with no timescale separation in it. Nothing on this surface should be changed to
  hide that.
- **The surface displays two levels because the interface supplies two.** `08` states this limit for
  the protocol and it is inherited here verbatim: a panel showing eight bands does not thereby show
  eight levels being used, and a reader who mistakes band count for demonstrated depth has been
  misled by the picture rather than by any claim in it.
