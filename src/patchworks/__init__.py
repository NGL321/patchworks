"""Patchworks: an embodied graph architecture.

See docs/spec/ for the architecture and CONTEXT.md for the vocabulary.

Built so far:

* :mod:`patchworks.body` — the shared frozen cell body, `encode` / `step` /
  `decode`, and the per-cell biases that are its whole adapting surface.

Nothing under prototypes/ is promoted here yet; later tickets bring
prototypes/sandbox and prototypes/regional-spectra into this package.
"""

__version__ = "0.0.0"
