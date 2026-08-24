# Patchworks

An embodied graph architecture in which many small predictors, each working in its own metric
space, are reconciled into a coherent model of a world none of them sees whole.

---

## The claim

A planar arm pushes pucks around a table. It is controlled by a graph of ~150 small predictors.
Every one of them is the same frozen network. None of them sees the whole scene — the camera is
tiled so finely that no single cell ever sees a whole puck. None of them is trained by
backpropagation through the graph; each learns from what it can see and nothing else.

They agree with each other, or they don't, and the disagreement is the only error signal there is.

The bet is that this is enough — that prediction, reconciled locally across a graph, produces
something that plans, acts, and recovers when you interfere with it.

## What "recovers" means

Push the arm off course and it corrects in a few ticks, near the sensors. Slide a puck out from
under a task it had already committed to and it takes longer, because the correction has to travel
further in — through cells that hold their content across hundreds of ticks and don't give it up
for one contrary neighbour.

That difference is the whole demonstration. Not that it recovers, but that **where** it recovers
from depends on what you did to it. A hierarchy you can see by poking.

## Status

**This is a specification, not a system.** Nothing has been trained. What exists is a complete
design — ten spec files, eleven decision records, and a MuJoCo sandbox you can run today — worked
out to the point where a build can start cold, with no architectural question left open.

The sandbox is real and honest about its difficulty: a scripted controller that knows every
position exactly solves 12 of 48 tasks.

## Read it

The spec is written in order and reads that way.

| | |
|---|---|
| [`01-cell-and-sheaf`](docs/spec/01-cell-and-sheaf.md) | What one cell is, and how it is glued to its neighbours |
| [`02-tick-semantics`](docs/spec/02-tick-semantics.md) | Predict, then reconcile. One step, never a solve |
| [`03-the-sandbox`](docs/spec/03-the-sandbox.md) | The arm, the pucks, and a Gymnasium contract with no episodes |
| [`04-action-and-the-boundary`](docs/spec/04-action-and-the-boundary.md) | Why an action is just a prediction the world has to make true |
| [`05-timescales`](docs/spec/05-timescales.md) | Why some cells are slow, without any cell having a clock |
| [`06-graph-topology`](docs/spec/06-graph-topology.md) | The dome: 264 cells at the rim tapering to 8 at the apex |
| [`07-local-learning-rule`](docs/spec/07-local-learning-rule.md) | Two rules, sharing no objective |
| [`08-the-acceptance-demo`](docs/spec/08-the-acceptance-demo.md) | The three things a human does to it, and what is measured |
| [`09-the-build-stack`](docs/spec/09-the-build-stack.md) | PyTorch, one laptop |
| [`10-the-demo-surface`](docs/spec/10-the-demo-surface.md) | Two windows |

[`CONTEXT.md`](CONTEXT.md) is the glossary — this project uses ordinary words in narrow senses,
and that file is where they are pinned down. [`docs/adr/`](docs/adr/) holds the eleven decisions
that were hard enough to need a reason written down.

## Run the sandbox

```
cd prototypes/sandbox && python watch.py
```

A live viewer, a scripted pusher, and three pucks. Ctrl-drag one to interfere with it by hand —
which is half of the acceptance demo, working today, against an agent that doesn't exist yet.
Setup is in [`prototypes/sandbox/README.md`](prototypes/sandbox/README.md).
