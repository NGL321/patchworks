"""Patchworks: an embodied graph architecture.

See docs/spec/ for the architecture and CONTEXT.md for the vocabulary.

Built so far:

* :mod:`patchworks.body` — the cell body, `encode` / `K` / `decode`, and the
  per-cell surface that adapts on it: the biases, and the operators `K`.
  `encode` is the body's only nonlinearity.
* :mod:`patchworks.graph` — the dome: :func:`~patchworks.graph.build_graph`,
  the structural masks, and the construction diagnostics.
* :mod:`patchworks.restriction` — the adapting surface's other half: one
  masked, gauge-fixed linear map per edge endpoint.
* :mod:`patchworks.tick` — the tick's two phases, the unit delay, and the
  reconciliation gain, over the state they run on.
* :mod:`patchworks.agent` — the graph wired to the world, and the ordering
  between them: external writes are a tick's last word.
* :mod:`patchworks.learning` — the learning phase: both halves of the local
  learning rule, the prediction rule and the transport rule, run over the tick's
  detached state through `torch.func` rather than ambient autograd.
* :mod:`patchworks.diagnostics` — the diagnostics that run on a cadence: the
  paired per-edge instrument (Dirichlet energy and effective rank, never one
  without the other), the topology-only `H¹` baseline, and the run-time
  `dim H⁰` and minimum achievable energy. Outside the architecture by
  construction — it reads, and no cell can reach it.
* :mod:`patchworks.timescale` — the clock divisor, as an instrument. Outside
  the architecture by construction: it drives the tick from outside, default
  off, and no cell can reach it.
* :mod:`patchworks.bias_selection` — bias selection: the driven-trajectory
  measurement, the overlapping bands, the fold-margin check, and the go/no-go
  that can kill the timescale mechanism before anything is trained.
* :mod:`patchworks.sandbox` — the world the agent lives in, promoted from
  prototypes/sandbox. Not imported here; import it directly.
* :mod:`patchworks.surface` — the demo surface: the tick record, and the one
  renderer that reads it. Outside the architecture by construction — display
  only, and no cell can reach it. Not imported here; import it directly.
* :mod:`patchworks.cli` — the `patchworks` command: `doctor`, `check`, `dome`
  and `demo`, and the `mjpython` re-exec so that nobody has to know about it.
  Outside the architecture by the same rule as the surface, and for the same
  reason. Not imported here — importing it would put argparse in the path of
  every `import patchworks`; `python -m patchworks` and the console script both
  reach it directly.
"""

from . import (
    agent,
    bias_selection,
    body,
    diagnostics,
    graph,
    learning,
    restriction,
    tick,
    timescale,
)

__all__ = [
    "agent",
    "bias_selection",
    "body",
    "diagnostics",
    "graph",
    "learning",
    "restriction",
    "tick",
    "timescale",
]

__version__ = "0.0.0"
