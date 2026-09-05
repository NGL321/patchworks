"""What #228's `c_v` ruling did to conduction at the actuator (#487).

[#228](https://github.com/NGL321/patchworks/issues/228) raised `c_v` at the
actuator from 2 to 3, lowering its reconciliation gain from `0.5γ` to `0.333γ`.
[#487](https://github.com/NGL321/patchworks/issues/487) asks what that does to
the conduction ratio there, and answers it **before and after, on the same rig
and the same seeds, at 30k and 100k**::

    python prototypes/actuator-conduction-487/read.py --arm clamped  --seed 0 --ticks 100000
    python prototypes/actuator-conduction-487/read.py --arm preclamp --seed 0 --ticks 100000

**The two arms differ in exactly one function.** `preclamp` restores
:func:`patchworks.restriction.overlap_counts` to its pre-#228 body — the
pigeonhole form without the pinned-incidence branch — before the graph is built,
so `Sheaf.__init__` reads the old gain and nothing else in the build changes.
Every reader of the count reaches it through `gain_denominators`, which resolves
`overlap_counts` in its own module globals at call time, so the one patch covers
the gain, the fold-margin nomination and the projection alike.

**The actuator's retention operator is available in closed form, and that is a
fact about the cell rather than a shortcut.** The actuator runs no body
(`inference_phase` writes only predicting stalks) and is never written by the
world (ADR-0016: it is *read*), so message passing is the whole of its dynamics::

    x ← x − g · Σ_e F_eᵀ (F_e x − y_e)   ⇒   dx ← A x,  A = I − g · Σ_e F_eᵀ F_e

Holding the neighbours' `y_e` exogenous — the same linearisation
[#274](https://github.com/NGL321/patchworks/issues/274) reads `ρ_full` under —
the paired deviation at the actuator evolves by `A` exactly. So `τ = −1/ln ρ(A)`
is read off a 6×6 eigendecomposition rather than fitted to a decay, and the gain
enters it analytically. :func:`empirical` checks the closed form against a
measured peak-to-`1/e` decay on the running graph rather than trusting it.

**What this is not.** ADR-0026's `τ̂` is the decay of a paired deviation
**projected onto the cell's private features**, and the actuator has none —
`Σ_e m_e = 12` against a stalk of 6, so `p_v = max(0, n − Σ_e m_e) = 0`, and
`Dome.private_projection` is `[150, 32]`, the predicting cells only. The number
here is read on the **whole stalk**, which ADR-0026 names as *arrival rather than
retention* and refuses as a reading site. It is reported under its own name,
`tau_stalk`, and never as `τ̂`. The gap that forces the substitution is #487's to
report, not this rig's to close.

Like every script here **it asserts nothing** and its exit code does not move.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in ("src", "tools", "benchmarks", "tests"):
    sys.path.insert(0, str(ROOT / _p))

from patchworks import restriction as R  # noqa: E402
from patchworks.graph import CellKind, Dome  # noqa: E402

#: The actuator on `DEFAULT_SPEC`. Resolved from the mask at run time as well —
#: this is the reading the record quotes, not an index the rig depends on.
ACTUATOR = 262

#: The checkpoints #487 names. #178 has cost this map the 30k mistake four times,
#: so both are read **on one trajectory** rather than in separate runs: 30k is a
#: prefix of the 100k run, and the pair is printed together or not at all.
CHECKPOINTS = (30000, 100000)

#: Paired trials per checkpoint. The closed form needs none; these are for
#: :func:`empirical` and for the L1 neighbours' `τ̂`, where the cost is `HOLD`
#: ticks a trial and the spread over apex sources is what is being bought.
TRIALS = 8


def preclamp_overlap_counts(dome: Dome, *, c: int = R.GAUGE_C) -> torch.Tensor:
    """`overlap_counts` as it stood before #228: no pinned-incidence branch.

    `min(deg(v), max(c, ceil(deg(v) / n_v)))`, copied from the pre-#228 body
    rather than reconstructed from the docstring. On `DEFAULT_SPEC` it differs
    from the shipped function at exactly one cell — the actuator, 2 against 3 —
    and :func:`arm_check` asserts that, so an arm that silently moved a second
    cell is visible as the run failing rather than as a number.
    """
    return torch.tensor(
        [
            float(min(deg, max(c, -(-deg // cell.stalk))))
            for cell, deg in zip(dome.cells, dome.degrees, strict=True)
        ],
        dtype=torch.float32,
    )


def apply_arm(arm: str) -> None:
    """Install the arm's `overlap_counts`. Must run before the graph is built."""
    if arm == "preclamp":
        R.overlap_counts = preclamp_overlap_counts
    elif arm != "clamped":
        raise ValueError(f"unknown arm {arm!r}")


def arm_check(dome: Dome, arm: str) -> dict:
    """The manipulation check: which cells the arm moved, and the gain it set.

    Reported into the JSON rather than printed alone. #487's whole question is
    about one cell's gain, so a run that cannot show the gain it ran under is
    not evidence about it.
    """
    from patchworks.tick import DEFAULT_GAMMA, reconciliation_gain

    counts = R.overlap_counts(dome)
    baseline = preclamp_overlap_counts(dome)
    moved = [
        int(i) for i in range(len(dome.cells)) if float(counts[i]) != float(baseline[i])
    ]
    gain = reconciliation_gain(dome, gamma=DEFAULT_GAMMA)
    return {
        "arm": arm,
        "c_v_actuator": float(counts[ACTUATOR]),
        "gain_actuator": float(gain[ACTUATOR]),
        "gain_over_gamma": float(gain[ACTUATOR]) / DEFAULT_GAMMA,
        "cells_differing_from_preclamp": moved,
    }


def actuator_id(dome: Dome) -> int:
    """The actuator off the mask, so the rig does not depend on `262`."""
    found = [c.id for c in dome.cells if c.kind is CellKind.ACTUATOR]
    if len(found) != 1:
        raise ValueError(f"expected one actuator, found {found}")
    return found[0]


def world_loop_actuator(dome: Dome, cell: int) -> int:
    """`world_loop` at the actuator, from `loop_length`'s own enumeration.

    `benchmarks/loop_length.world_loops` runs its `min` over `dome.predicting`,
    so the actuator is absent from it — the same omission #487 reports. The
    divisor is recomputed here from **that module's** `adjacency` and
    `distances_from` and its `WORLD_TICK`, over the same `(a, p)` pairs with the
    same `a != p` ban, widened only in the population. A second enumeration
    would be free to drift from the one under the operative bar, which is the
    one place the record can least afford it.
    """
    import loop_length

    neighbours = loop_length.adjacency(dome)
    actuators = tuple(sorted(c.id for c in dome.cells if c.kind is CellKind.ACTUATOR))
    sensory = tuple(sorted(c.id for c in dome.cells if c.kind in loop_length.SENSORY))
    reach = {c: loop_length.distances_from(neighbours, c) for c in set(actuators) | set(sensory)}
    candidates = [
        reach[a][cell] + loop_length.WORLD_TICK + reach[p][cell]
        for a in actuators
        for p in sensory
        if a != p and cell in reach[a] and cell in reach[p]
    ]
    return min(candidates)


def retention_operator(sheaf, dome: Dome, cell: int, commanded: int) -> dict:
    """`A = I − g·Σ_e F_eᵀF_e` at the actuator, read on the block that retains.

    **Only the commanded components retain, and that is the world's doing.**
    `Agent.write` sets `stalks[_efference_slice] = applied` at the end of every
    tick, so the actuator's `joints` efference components are overwritten from
    outside — byte-identically in both branches of a paired fork, which drives
    the deviation there to zero each tick. The `commanded = actuator_stalk −
    joints` components are written by nobody and carry whatever the graph left.
    So the deviation's recurrence is `Δx ← P A Δx` with `P` the projector onto
    the commanded block, and the retained operator is `A[:commanded, :commanded]`.

    This is ADR-0026's own logic arriving at the actuator rather than a new
    convention: the ADR excludes sensory boundary cells because *the world's
    write voids what arrives*, and at the actuator the write voids exactly the
    efference half. Reading `ρ` on the whole 6×6 stalk would credit the cell
    with retention in directions the world resets — the amplitude-style error
    the ADR replaced the bottleneck ratio to be rid of. Both radii are reported;
    `tau_closed` is the commanded one, and `rho_A_full` is kept beside it so the
    difference the projection makes is visible rather than asserted.

    The eigenvalues of `M = Σ_e F_eᵀF_e` come back too. They explain the result
    rather than decorate it: `trace(M) = Σ_e ‖F_e‖_F²` is pinned to `deg(v)` by
    the exact gauge at a fully-pinned cell, so learning cannot change how much
    there is — only how it is spread — and `ρ` is set by the direction the maps
    cover least.
    """
    n = dome.cells[cell].stalk
    M = torch.zeros(n, n, dtype=torch.float64)
    for edge_id in dome.incident[cell]:
        edge = dome.edges[edge_id]
        side = 0 if edge.u == cell else 1
        F = sheaf.maps.maps[R.pair_index(edge_id, side)][: edge.m, :n].detach().double()
        M += F.T @ F
    eigenvalues = torch.linalg.eigvalsh(M)
    gain = float(sheaf.gain[cell])
    A = torch.eye(n, dtype=torch.float64) - gain * M
    retained = A[:commanded, :commanded]
    rho = float(torch.linalg.eigvals(retained).abs().max())
    rho_full = float((1.0 - gain * eigenvalues).abs().max())
    tau = float("inf") if rho >= 1.0 else float(-1.0 / np.log(rho))
    return {
        "gain": gain,
        "commanded": commanded,
        "M_eigenvalues": [float(x) for x in eigenvalues],
        "M_trace": float(M.trace()),
        "M_lambda_min": float(eigenvalues.min()),
        "M_lambda_max": float(eigenvalues.max()),
        "A_commanded_eigenvalues": [
            [float(z.real), float(z.imag)] for z in torch.linalg.eigvals(retained)
        ],
        "rho_A_commanded": rho,
        "rho_A_full": rho_full,
        "tau_closed": tau,
        "tau_full_stalk": (
            float("inf") if rho_full >= 1.0 else float(-1.0 / np.log(rho_full))
        ),
    }


def operator_identity(
    agent, env, dome: Dome, cell: int, commanded: int, seed: int, draws: int = 4
) -> dict:
    """Is `dx ← A dx` a fact about the running graph, or only about the algebra?

    #274 earned its correction by reconstructing the stalk the run actually left
    behind rather than reading the recurrence off the source, and the same check
    is owed here. **At the first tick after the nudge the step is exactly `A δ`**,
    and the unit delay is why: `message_passing_phase` reconciles against the
    partner's slot in *last* tick's broadcast, so a deviation introduced at the
    actuator cannot have reached `y_e` yet. Every later tick mixes in the
    neighbours' response, which is the composite :func:`empirical` reads and not
    this operator.

    Reported as the relative residual `‖measured − A δ‖ / ‖A δ‖` over random
    unit deviations. Float32 machine precision is a pass; anything larger means
    the closed form is not the cell's dynamics and the reading built on it is
    void.
    """
    import detectability as D
    import untrained_fixed_point as ufp

    n = dome.cells[cell].stalk
    A = torch.eye(n, dtype=torch.float64) - retention_matrix(agent.sheaf, dome, cell)
    keep = torch.zeros(n, dtype=torch.float64)
    keep[:commanded] = 1.0
    generator = torch.Generator().manual_seed(seed + 77)
    observation, _info = env.reset(seed=seed * 1000)
    agent.observe(observation)
    applied = np.zeros(env.action_space.shape, dtype=np.float64)
    D.hold_still(agent, observation, applied, D.HOLD)
    state = ufp.snapshot(agent.sheaf)

    residuals = []
    for _ in range(draws):
        quiet, held = D.branch(agent, state, observation, applied, 1, None, record=(cell,))
        # Supported on the commanded block, because that is the only place a
        # deviation at this cell can survive the world's write to be read again.
        delta = D.unit(n, generator) * keep
        delta = delta / delta.norm().clamp_min(1e-30)
        _r, moved = D.ratios(
            agent, state, quiet, observation, applied, ((cell, delta),), D.PROBE, 1,
            sustained=None, record=(cell,),
        )
        measured = (moved[cell] - held[cell])[0].double()
        predicted = (A @ delta.double()) * keep
        residuals.append(
            float((measured - predicted).norm() / predicted.norm().clamp_min(1e-30))
        )
    ufp.restore(agent.sheaf, state)
    return {
        "draws": draws,
        "relative_residual": residuals,
        "relative_residual_max": max(residuals),
    }


def retention_matrix(sheaf, dome: Dome, cell: int) -> torch.Tensor:
    """`g · Σ_e F_eᵀF_e` at a cell — the subtracted half of `A`."""
    n = dome.cells[cell].stalk
    M = torch.zeros(n, n, dtype=torch.float64)
    for edge_id in dome.incident[cell]:
        edge = dome.edges[edge_id]
        side = 0 if edge.u == cell else 1
        F = sheaf.maps.maps[R.pair_index(edge_id, side)][: edge.m, :n].detach().double()
        M += F.T @ F
    return float(sheaf.gain[cell]) * M


def empirical(
    agent, env, dome: Dome, cell: int, commanded: int, seed: int, trials: int, window: int
) -> dict:
    """The measured decay at the actuator, so the closed form is checked not trusted.

    `apex-to-rim`, which is ADR-0026's **outbound** direction and the one whose
    universal names this cell: a unit deviation into one apex cell, the world
    held still, the two branches differenced, and the deviation read at the
    actuator. `τ` is ADR-0026's own reading rule — peak-to-`1/e` in integer
    ticks, :func:`detectability.tau_hat` — taken on the **whole stalk**, because
    the actuator has no private block to project onto. That substitution is the
    finding, and it is labelled `tau_stalk` wherever it appears.

    The three L1 predicting neighbours are read in the same fork under their own
    private projection, which **is** ADR-0026's `τ̂`. They are the cells the
    actuator's gain reaches first, and they are in the outbound universal on the
    ADR's own terms, so they are where a side effect that the actuator's own
    number cannot carry would show up.
    """
    import detectability as D
    import untrained_fixed_point as ufp

    neighbours = sorted(dome.edges[e].other(cell) for e in dome.incident[cell])
    rows = {c: i for i, c in enumerate(dome.predicting)}
    projection = dome.private_projection.to(torch.float64)
    record = tuple(dict.fromkeys((cell,) + tuple(neighbours) + tuple(dome.predicting)))
    pool = D.apex(dome)
    generator = torch.Generator().manual_seed(seed + 1)

    stalk_taus, neighbour_taus = [], {c: [] for c in neighbours}
    for i in range(trials):
        observation, _info = env.reset(seed=seed * 1000 + i)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        D.hold_still(agent, observation, applied, D.HOLD)
        state = ufp.snapshot(agent.sheaf)
        source = int(pool[i % len(pool)])
        quiet, held = D.branch(
            agent, state, observation, applied, window, None, record=record
        )
        nudge = ((source, D.unit(dome.cells[source].stalk, generator)),)
        _ratio, moved = D.ratios(
            agent, state, quiet, observation, applied, nudge, D.PROBE, window,
            sustained=None, record=record,
        )
        ufp.restore(agent.sheaf, state)

        # Read on the commanded block, for :func:`retention_operator`'s reason:
        # the efference components are reset from outside every tick, so a
        # deviation there is the world's write and not the cell's retention.
        difference = (moved[cell] - held[cell])[:window]
        deviation = difference[:, :commanded].norm(dim=-1).numpy()
        tau, _censored, _peak = D.tau_hat(deviation[:, None])
        stalk_taus.append(float(tau[0]))
        for c in neighbours:
            private = ((moved[c] - held[c])[:window] * projection[rows[c]]).norm(dim=-1).numpy()
            t, _c, _p = D.tau_hat(private[:, None])
            neighbour_taus[c].append(float(t[0]))
    return {
        "trials": trials,
        "window": window,
        "tau_stalk_per_trial": stalk_taus,
        "tau_stalk_median": float(np.median(stalk_taus)),
        "neighbours": {
            str(c): {
                "tau_hat_per_trial": v,
                "tau_hat_median": float(np.median(v)),
                "p_v": int(dome.private_dimensions[rows[c]]),
            }
            for c, v in neighbour_taus.items()
        },
    }


def checkpoint(
    agent, env, dome, cell, commanded, divisor, seed, at, trials, window, arm
) -> dict:
    closed = retention_operator(agent.sheaf, dome, cell, commanded)
    identity = operator_identity(agent, env, dome, cell, commanded, seed)
    measured = empirical(agent, env, dome, cell, commanded, seed, trials, window)
    row = {
        "operator_identity": identity,
        "arm": arm,
        "seed": seed,
        "ticks": at,
        "actuator": cell,
        "world_loop": divisor,
        **closed,
        "conduction_ratio_closed": closed["tau_closed"] / divisor,
        **measured,
        "conduction_ratio_stalk": measured["tau_stalk_median"] / divisor,
    }
    print(
        f"  [{arm} seed {seed} @ {at}] gain={closed['gain']:.4f} "
        f"rho_cmd={closed['rho_A_commanded']:.5f} tau_closed={closed['tau_closed']:.2f} "
        f"tau_stalk={measured['tau_stalk_median']:.2f} "
        f"ratio_closed={row['conduction_ratio_closed']:.3f} "
        f"identity_residual={identity['relative_residual_max']:.2e}",
        flush=True,
    )
    return row


def run(arm: str, seed: int, ticks: int, trials: int, window: int, out: pathlib.Path,
        checkpoints: tuple[int, ...] = CHECKPOINTS) -> None:
    apply_arm(arm)
    import untrained_fixed_point as ufp

    env, agent = ufp.build("real", "train", seed)
    dome = agent.dome
    cell = actuator_id(dome)
    commanded = int(agent.commanded)
    divisor = world_loop_actuator(dome, cell)
    check = arm_check(dome, arm)
    print(json.dumps(check, indent=2), flush=True)
    print(f"actuator {cell}: world_loop = {divisor}, stalk = {dome.cells[cell].stalk}, "
          f"commanded = {commanded}, efference = {dome.cells[cell].stalk - commanded}, "
          f"Sigma_e m_e = {sum(dome.edges[e].m for e in dome.incident[cell])}", flush=True)

    marks = [c for c in checkpoints if c <= ticks]
    rows, done, started = [], 0, time.time()
    for mark in marks:
        span = mark - done
        print(f"training {span} ticks to {mark} (float32)...", flush=True)
        ufp.taught(agent, span, seed)
        done = mark
        rows.append(
            checkpoint(
                agent, env, dome, cell, commanded, divisor, seed, mark, trials, window, arm
            )
        )
        # Written as each checkpoint lands: a 100k run that dies at 90k should
        # still leave its 30k reading behind.
        out.write_text(
            json.dumps(
                {"arm_check": check, "actuator": cell, "world_loop": divisor,
                 "commanded": commanded, "stalk": dome.cells[cell].stalk,
                 "elapsed_s": time.time() - started, "checkpoints": rows},
                indent=2,
            )
        )
        print(f"  wrote {out} ({time.time() - started:.0f}s elapsed)", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("clamped", "preclamp"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=100000)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--checkpoints",
        default=",".join(str(c) for c in CHECKPOINTS),
        help="comma-separated tick marks to read at; the default pair is #487's",
    )
    args = p.parse_args(argv)
    marks = tuple(int(x) for x in args.checkpoints.split(","))
    out = pathlib.Path(
        args.out
        or (pathlib.Path(__file__).parent / f"487-{args.arm}-seed{args.seed}-{args.ticks}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    run(args.arm, args.seed, args.ticks, args.trials, args.window, out, marks)


if __name__ == "__main__":
    main()
