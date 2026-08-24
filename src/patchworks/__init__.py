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
* :mod:`patchworks.sandbox` — the world the agent lives in, promoted from
  prototypes/sandbox. Not imported here; import it directly.

The rest of prototypes/ is not promoted yet; a later ticket brings
prototypes/regional-spectra into this package.
"""

from . import agent, body, graph, restriction, tick

__all__ = ["agent", "body", "graph", "restriction", "tick"]

__version__ = "0.0.0"
