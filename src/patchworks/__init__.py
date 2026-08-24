"""Patchworks: an embodied graph architecture.

See docs/spec/ for the architecture and CONTEXT.md for the vocabulary.

Built so far:

* :mod:`patchworks.body` — the shared frozen cell body, `encode` / `step` /
  `decode`, and the per-cell biases that are its whole adapting surface.
* :mod:`patchworks.graph` — the dome: :func:`~patchworks.graph.build_graph`,
  the structural masks, and the construction diagnostics.
* :mod:`patchworks.sandbox` — the world the agent lives in, promoted from
  prototypes/sandbox. Not imported here; import it directly.

The rest of prototypes/ is not promoted yet; a later ticket brings
prototypes/regional-spectra into this package.
"""

from . import body, graph

__all__ = ["body", "graph"]

__version__ = "0.0.0"
