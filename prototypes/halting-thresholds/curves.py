"""Synthetic error curves with known ground truth, for judging halting rules.

**Throwaway.** Built for [issue #156](https://github.com/NGL321/patchworks/issues/156).

The register's live-readable costs are signatures on quantities a run produces.
No run produces them yet — the conversion edit is unwritten and the graph does
not transmit — so the rules cannot be calibrated against recorded curves. They
can still be *discriminated*, which is the cheaper and more urgent question:
given a curve whose cause we chose, does the rule fire for the right reason?

Everything here is generated. Nothing here is a measurement of Patchworks.

## The decomposition

ADR-0004, as amended by ADR-0007 and #49 and #141, says a persistent mid-depth
residual has **four** causes, and only one of them is entry 1:

    e(t) = reducible(t) + curvature + lag(t) + selfint + gauge + noise

* ``reducible``  falls with learning. Not a cost.
* ``curvature``  the chart cannot follow the piece. **This is entry 1.**
* ``lag``        ADR-0007's tolerated floor. Drains under a quiescent hold.
* ``selfint``    stalk too narrow (#49). Known at construction, box-counting.
* ``gauge``      the frozen readout's shared direction (#141). Known at
                 construction, and shared across *unrelated* edges.

The two construction-time causes are not simulated as curve shapes, because
they are not read off a curve — they are read forwards. They appear here only
as flags on a regime, so a rule that forgets to check them can be caught.

## The quiescent hold

ADR-0004's disambiguator: hold the world still and sweep configurations; lag
drains, curvature does not. That is modelled literally — during a hold the
``lag`` term decays with its own time constant and every other term stands. It
is the whole reason entry 1 does not need an open-ended waiting game.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Regime:
    """A named error-curve generator with the verdict it deserves."""

    name: str
    reducible: float          # amplitude of the learning-reducible part
    tau: float                # ticks; its decay constant
    curvature: float          # entry 1's irreducible part
    lag: float                # ADR-0007's floor, drains under hold
    tau_hold: float           # ticks; how fast lag drains once the world stops
    noise: float              # additive sd, as a fraction of the initial level
    selfint_at_construction: bool = False   # #49's box-counting criterion failed
    gauge_at_construction: bool = False     # #141's shared direction present
    should_halt: bool = False               # the verdict a correct rule reaches
    because: str = ""

    def curve(self, ticks: int, rng: np.random.Generator) -> np.ndarray:
        t = np.arange(ticks, dtype=float)
        e = self.reducible * np.exp(-t / self.tau) + self.curvature + self.lag
        scale = self.reducible + self.curvature + self.lag
        return e + rng.normal(0.0, self.noise * scale, ticks)

    def under_hold(
        self, level_at_entry: float, hold: int, rng: np.random.Generator
    ) -> np.ndarray:
        """The curve during a quiescent hold entered at ``level_at_entry``.

        The world stops, so ``reducible`` stops falling — there is no new
        evidence — and ``lag`` drains. Whatever the curve was standing on that
        is *not* lag is what remains, which is exactly the reading ADR-0004
        asks for.
        """
        t = np.arange(hold, dtype=float)
        standing = max(level_at_entry - self.lag, 0.0)
        e = standing + self.lag * np.exp(-t / self.tau_hold)
        scale = max(level_at_entry, 1e-12)
        return e + rng.normal(0.0, self.noise * scale, hold)


#: The bank. Four causes plus the impatience case, which is the one that makes
#: a pure window/slope rule unsafe at any window length.
BANK: tuple[Regime, ...] = (
    Regime(
        name="learning",
        reducible=1.0, tau=400.0, curvature=0.005, lag=0.005,
        tau_hold=60.0, noise=0.06,
        should_halt=False,
        because="the chart is fine and the error is still falling",
    ),
    Regime(
        name="curvature",
        reducible=0.6, tau=250.0, curvature=0.35, lag=0.02,
        tau_hold=60.0, noise=0.06,
        should_halt=True,
        because="entry 1: what is left does not fall and does not drain",
    ),
    Regime(
        name="lag-floor",
        reducible=0.6, tau=250.0, curvature=0.01, lag=0.34,
        tau_hold=60.0, noise=0.06,
        should_halt=False,
        because="ADR-0007's floor is tolerated, not represented; it drains",
    ),
    # The two slow regimes are a pair, and they are the point of ``tau_max``.
    # Ground truth here is **budget-relative**, which the prototype did not
    # expect and which is the honest reading: at a budget of 3000 with
    # tau_max = 1000, a cell learning with tau = 6000 will not have finished
    # inside any run this project can afford, and "irreducible" and "slower
    # than we will ever wait" are the same claim about it.
    Regime(
        name="slow-affordable",
        reducible=0.9, tau=800.0, curvature=0.01, lag=0.01,
        tau_hold=60.0, noise=0.06,
        should_halt=False,
        because="tau 800 < tau_max 1000; falling slowly but inside the budget, "
                "so halting here is impatience",
    ),
    Regime(
        name="slow-unaffordable",
        reducible=0.9, tau=6000.0, curvature=0.01, lag=0.01,
        tau_hold=60.0, noise=0.06,
        should_halt=True,
        because="tau 6000 > tau_max 1000; beyond the honoured budget, where "
                "'does not fall with learning' is a claim about this run",
    ),
    Regime(
        name="self-intersection",
        reducible=0.6, tau=250.0, curvature=0.33, lag=0.02,
        tau_hold=60.0, noise=0.06,
        selfint_at_construction=True,
        should_halt=False,
        because="#49's cause, indistinguishable on the curve and ruled out "
                "forwards; halting on it books the wrong cost",
    ),
    Regime(
        name="gauge-direction",
        reducible=0.6, tau=250.0, curvature=0.33, lag=0.02,
        tau_hold=60.0, noise=0.06,
        gauge_at_construction=True,
        should_halt=False,
        because="#141's cause, shared across unrelated edges and known at "
                "construction; entry 6, not entry 1",
    ),
)
