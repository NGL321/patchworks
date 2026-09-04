# The holonomy read (#453)

The three runs behind [#453](https://github.com/NGL321/patchworks/issues/453)'s
reading, kept because the ticket is [#454](https://github.com/NGL321/patchworks/issues/454)'s
cutoff and a cutoff that fires on a number nobody can re-open is a cutoff that
fired on folklore.

- `read.log` — the rig's whole output, one process: the free arms, the two
  trained trajectories, and the report.
- `read.json` — every cycle's columns for every arm, written as each checkpoint
  landed rather than at exit (`holonomy_read.read`'s `sink`, and
  `spectral_floor_read`'s reason: this run is hours long on a shared box).

Reproduce with:

    python benchmarks/holonomy_read.py read

**The surface.** Every arm was trained in this process, on `main` at the commit
this directory lands on, and nothing is differenced against stored JSON — the
map's 2026-09-04 standing rule, which exists because ADR-0031 deleted the
sparsity pressure and made every `lambda = 0.4` number a number about a build
that no longer exists.

**The limit, stated rather than discovered.** One seed (42) carries the two
trained arms; eight draws carry the free ones. A 100k trained trajectory is
~1.6 h on this box, so a three-seed spread on the trained arms would have been a
day of wall clock. The spread that is affordable is published where it is
affordable.
