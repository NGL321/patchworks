"""Patchworks: an embodied graph architecture.

See docs/spec/ for the architecture and CONTEXT.md for the vocabulary.

Built so far:

* :mod:`patchworks.body` — the shared frozen cell body, `encode` / `step` /
  `decode`, and the per-cell biases that are its whole adapting surface.
* :mod:`patchworks.graph` — the dome: :func:`~patchworks.graph.build_graph`,
  the structural masks, and the construction diagnostics.
* :mod:`patchworks.restriction` — the adapting surface's other half: one
  masked, gauge-fixed linear map per edge endpoint.
* :mod:`patchworks.tick` — the tick's two phases, the unit delay, and the
  reconciliation gain, over the state they run on.
* :mod:`patchworks.agent` — the graph wired to the world, and the ordering
  between them: external writes are a tick's last word.
* :mod:`patchworks.learning` — the learning phase: both halves of the local
  learning rule, the bias rule and the transport rule, run over the tick's
  detached state through `torch.func` rather than ambient autograd.
* :mod:`patchworks.timescale` — the clock divisor, as an instrument. Outside
  the architecture by construction: it drives the tick from outside, default
  off, and no cell can reach it.
* :mod:`patchworks.bias_selection` — bias selection: the driven-trajectory
  measurement, the overlapping bands, the fold-margin check, and the go/no-go
  that can kill the timescale mechanism before anything is trained.
* :mod:`patchworks.sandbox` — the world the agent lives in, promoted from
  prototypes/sandbox. Not imported here; import it directly.
"""

from . import (
    agent,
    bias_selection,
    body,
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
    "graph",
    "learning",
    "restriction",
    "tick",
    "timescale",
]

__version__ = "0.0.0"
