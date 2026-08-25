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
* :mod:`patchworks.learning` — the learning phase: the local learning rule,
  run over the tick's detached state through `torch.func` rather than ambient
  autograd.
* :mod:`patchworks.timescale` — the clock divisor, as an instrument. Outside
  the architecture by construction: it drives the tick from outside, default
  off, and no cell can reach it.
* :mod:`patchworks.sandbox` — the world the agent lives in, promoted from
  prototypes/sandbox. Not imported here; import it directly.

The rest of prototypes/ is not promoted yet; a later ticket brings
prototypes/regional-spectra into this package.
"""

from . import agent, body, graph, learning, restriction, tick, timescale

__all__ = [
    "agent",
    "body",
    "graph",
    "learning",
    "restriction",
    "tick",
    "timescale",
]

__version__ = "0.0.0"
