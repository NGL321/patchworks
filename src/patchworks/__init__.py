"""Patchworks: an embodied graph architecture.

See docs/spec/ for the architecture and CONTEXT.md for the vocabulary.

Built so far:

* :mod:`patchworks.body` — the shared frozen cell body, `encode` / `step` /
  `decode`, and the per-cell biases that are its whole adapting surface.
* :mod:`patchworks.graph` — the dome: :func:`~patchworks.graph.build_graph`,
  the structural masks, and the construction diagnostics.
* :mod:`patchworks.timescales` — bias selection: the driven-trajectory
  measurement, the overlapping bands, the fold-margin check, and the go/no-go
  that can kill the timescale mechanism before anything is trained.
* :mod:`patchworks.sandbox` — the world the agent lives in, promoted from
  prototypes/sandbox. Not imported here; import it directly.
"""

from . import body, graph, timescales

__all__ = ["body", "graph", "timescales"]

__version__ = "0.0.0"
