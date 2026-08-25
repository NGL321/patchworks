"""The clock divisor, as an instrument.

A cell updating every `k` ticks, `k` fixed by hand. `docs/spec/05-timescales.md`
(*The clock divisor, as an instrument*) and
`docs/adr/0005-timescale-is-persistence-not-a-schedule.md` are the whole of what
this module implements.

It is **built first, and not as a fallback.** It is the rig that establishes
that the capability depends on timescale at all: force `k`, confirm the
long-horizon behaviour appears, and the variable is isolated before anything is
spent on making it emerge. Then switch it off and ask whether persistence
reproduces it. Because active predictive coding's `T1`/`T2` *is* a fixed clock
divisor, that comparison is a comparison **against APC** rather than against
nothing.

**Nothing in the architecture reads a cell's timescale.** That prohibition is
what keeps the divisor and the persistence mechanism interchangeable, and it is
the reason this module is shaped the way it is:

* **The divisor drives the sheaf from outside; the sheaf does not know it
  exists.** :class:`ClockDivisor` holds a reference to nothing — not the sheaf,
  not the agent — and nothing holds a reference to it but the caller who
  switched it on. So a divisor is not reachable from any object a cell's
  computation is handed, and a learning rule written against the sheaf, the
  body, the biases or the maps **cannot** consult one, whatever it wants.
  `tests/test_timescale.py` asserts both halves: the object walk, and an AST
  scan over every other module of the package, which is what keeps the
  prohibition true of modules that do not exist yet.
* **It composes the tick's public phases rather than editing them.** No line of
  :mod:`patchworks.tick` changes, which is what makes "with the divisor off the
  tick is bit-identical" structural rather than something to be careful about:
  with no cell held this class calls :meth:`~patchworks.tick.Sheaf.tick` and
  gets out of the way.

**What a divisor decimates is the cell's own body update, and nothing else.**
A held cell does not run `encode` / `step` / `decode` this tick: its chart, its
prediction and the node stalk components it would have overwritten stand. The
message-passing phase runs for every cell every tick regardless — every edge
still carries a value, the unit delay is untouched, and the one descent step of
ADR-0002 still happens. That is not a compromise with the tick contract but the
point: **slow content lives in the private features**
(`docs/spec/05-timescales.md`), reconciliation cannot move them, and so the
private component of a held cell's stalk is exactly what the persistence
mechanism would have held. Freezing the reconciled component too would decimate
something persistence never decimates, and the instrument would stop being
interchangeable with the mechanism it measures — as well as severing the motor
pathway, whose commanded components reconciliation is what fills.

**The phase.** A cell with divisor `k` updates on the ticks where
`sheaf.ticks % k == 0`, so every cell updates on the first tick and `k = 1` is
indistinguishable from off. Nothing in the record fixes an offset, so there is
none.
"""

from __future__ import annotations

from typing import Iterator, Mapping

import torch

from .agent import Agent, TickOutcome
from .graph import Dome
from .tick import Sheaf

__all__ = ["ClockDivisor"]


class ClockDivisor:
    """A per-cell update divisor `k`, set by hand and switched from outside.

    Construct one with the cells to divide and how far to divide them::

        divisor = ClockDivisor(dome, {apex_cell: 8})
        for _ in range(ticks):
            divisor.tick(agent)

    Constructed empty it is **off**, and the mode is switchable without a
    rebuild: the same built graph runs one way through :meth:`Agent.tick
    <patchworks.agent.Agent.tick>` and the other through :meth:`tick` here, and
    :meth:`divide` and :meth:`clear` move between them mid-run. Nothing about
    the dome, the body, the biases or the maps depends on which is in use.

    Only a predicting cell can be divided. A boundary cell runs no body and has
    no chart (ADR-0006), so there is nothing of its own for a schedule to
    decimate — the world writes it every tick and that write is the tick's last
    word.
    """

    def __init__(self, dome: Dome, divisors: Mapping[int, int] | None = None) -> None:
        self.dome = dome
        #: Which predicting row of the sheaf's population each cell id is.
        self._row = {cell_id: row for row, cell_id in enumerate(dome.predicting)}
        self._divisors: dict[int, int] = {}
        for cell_id, k in (divisors or {}).items():
            self.divide(cell_id, k)

    # -- setting it by hand ------------------------------------------------

    @classmethod
    def uniform(cls, dome: Dome, k: int) -> "ClockDivisor":
        """Every predicting cell on the same divisor. The degenerate instrument."""
        return cls(dome, {cell_id: k for cell_id in dome.predicting})

    @classmethod
    def by_level(cls, dome: Dome, divisors: Mapping[int, int]) -> "ClockDivisor":
        """One divisor per level of the construction layout, `{level: k}`.

        The layout is read **here**, once, from outside the graph, which is what
        a construction layout is for (`docs/spec/06-graph-topology.md`) — the
        same act as the agent building its write tables from it. Nothing reads
        it afterwards, and no cell reads it ever.

        This is the shape the instrument is usually wanted in: the deep levels
        forced slow while the rim runs free, which is APC's `T1`/`T2` written
        over a graph that has more than two levels.
        """
        return cls(
            dome,
            {
                cell_id: divisors[dome.cells[cell_id].index.level]
                for cell_id in dome.predicting
                if dome.cells[cell_id].index.level in divisors
            },
        )

    def divide(self, cell_id: int, k: int) -> None:
        """Divide one cell's updates by `k`. `k = 1` is the same as not dividing it."""
        if cell_id not in self._row:
            kind = (
                "a boundary cell"
                if 0 <= cell_id < len(self.dome.cells)
                else "not a cell of this dome"
            )
            raise ValueError(
                f"cell {cell_id} is {kind}; only a predicting cell has a body and a "
                "chart for a divisor to hold (ADR-0006)"
            )
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(
                f"a clock divisor is a whole number of ticks, k >= 1; got {k!r}"
            )
        self._divisors[cell_id] = k

    def clear(self) -> None:
        """Switch the instrument off, without rebuilding anything."""
        self._divisors.clear()

    @property
    def divisors(self) -> dict[int, int]:
        """`{cell id: k}` for the divided cells, as a copy. Empty means off."""
        return dict(self._divisors)

    def __bool__(self) -> bool:
        return bool(self._divisors)

    def __repr__(self) -> str:
        return f"ClockDivisor({self._divisors or 'off'})"

    # -- what it does to a tick --------------------------------------------

    def held(self, ticks: int) -> tuple[int, ...]:
        """The cell ids not running their body on tick `ticks`.

        Observable from outside, which is the only place a timescale is ever
        observable from. Nothing inside the graph calls this.
        """
        return tuple(
            cell_id for cell_id, k in sorted(self._divisors.items()) if ticks % k
        )

    def _held_rows(self, ticks: int) -> torch.Tensor | None:
        """The population rows to hold this tick, or `None` if there are none."""
        rows = [self._row[cell_id] for cell_id in self.held(ticks)]
        return torch.tensor(rows, dtype=torch.long) if rows else None

    def advance(self, sheaf: Sheaf) -> None:
        """One whole tick of the graph, under the schedule.

        With no cell held this is :meth:`~patchworks.tick.Sheaf.tick`, called —
        not reimplemented, so the off mode cannot drift from #86's tick by so
        much as a rounding.

        With cells held it is the same two phases in the same order, driven from
        outside through the public phases #86 made public for exactly this kind
        of caller. The held cells' chart, prediction and node stalk components
        are taken before the inference phase and put back after it, so the body
        runs over the whole population either way — the arithmetic a cell does
        is unchanged and only whether its result is committed differs, which is
        what makes the divisor a schedule rather than an input to anything.
        """
        if sheaf.dome is not self.dome:
            raise ValueError("the clock divisor was built for a different dome")
        rows = self._held_rows(sheaf.ticks)
        if rows is None:
            sheaf.tick()
            return
        with torch.no_grad():
            # Cloned, though indexing by a row tensor already copies: a later
            # hand that made `rows` a slice would otherwise take views, and the
            # restore below would silently become a no-op.
            positions = sheaf.layout.predicting_positions[rows]
            charts = sheaf.charts[rows].clone()
            prediction = sheaf.prediction[rows].clone()
            stalks = sheaf.stalks[positions].clone()
            sheaf.inference_phase()
            sheaf.charts[rows] = charts
            sheaf.prediction[rows] = prediction
            sheaf.stalks[positions] = stalks
            sheaf.message_passing_phase()
        sheaf.ticks += 1
        sheaf.assert_no_tape()

    def tick(self, agent: Agent) -> TickOutcome:
        """One whole tick of the agent under the schedule, world included.

        :meth:`patchworks.agent.Agent.tick` with the graph's half replaced by
        :meth:`advance`. The world's half is the agent's own :meth:`act
        <patchworks.agent.Agent.act>` on its own :meth:`command
        <patchworks.agent.Agent.command>`, unchanged and uninspected — the
        ordering is the agent's, and a schedule has no business editing it.
        """
        self.advance(agent.sheaf)
        outcome = agent.act(agent.command())
        agent.sheaf.assert_no_tape()
        return outcome

    def run(
        self, agent: Agent, ticks: int, *, seed: int | None = None
    ) -> Iterator[TickOutcome]:
        """:func:`patchworks.agent.run`, under the schedule.

        Arranges the world when called rather than on the first `next()`, for
        the reason that function gives, so the same demo can be driven both ways
        from the same starting arrangement.
        """
        observation, _info = agent.env.reset(seed=seed)
        agent.observe(observation)

        def ticking() -> Iterator[TickOutcome]:
            for _ in range(ticks):
                yield self.tick(agent)

        return ticking()
