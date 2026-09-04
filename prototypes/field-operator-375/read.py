"""Non-normality of the **field-level** operator, on the driven rig (#375 §2).

[#375](https://github.com/NGL321/patchworks/issues/375) §2 orders one read
before any coherent-structure hunt, and gates the hunt on it. Every source
[#374](https://github.com/NGL321/patchworks/issues/374) found that explains how
a linearly-stable driven medium holds structure locates the capacity in
**non-normality** -- and in all of them it is a property of the **coupling
structure**, not of a node's own operator. [#166](https://github.com/NGL321/patchworks/issues/166)
measured the cell operator `K` at `0.0504` on the instrument ADR-0023 names,
`|K^T K - K K^T|_F / |K|_F^2`, which leaves the cell overwhelmingly normal. The
counterpart with the coupling in it has never been measured. This is it.

**The field state closes on `(chart, evidence)`, and the operator is exact.**
One tick is `inference_phase` then `message_passing_phase` (`tick.py:754-853`).
Write `z` for the persisted charts and `s` for the predicting cells' node
stalks. Then, linearised in the activation region the run is in::

    z(t+1)  =  K (J_chart z(t) + J_stalk s(t))
    s(t+1)  =  A_v D z(t+1)  +  g_v Sum_{p in v} F_p^T F_pbar D z_{u(pbar)}(t)

with `A_v = I - g_v Sum_e F_ev^T F_ev`, exactly #274's relay. The second term is
the whole of the coupling, and it is a **fact about the unit delay**: what cell
`v` reconciles against is `broadcast(t-1)`, and `broadcast(t-1) = F (D z(t) + b)`
because the broadcast was formed from the prediction that *this* tick's chart
decoded to. So a neighbour's chart enters `v`'s evidence with no extra state,
and `(z, s)` is closed. :func:`check_broadcast_is_last_prediction` checks that
identity against the buffer the run actually left behind, and
:func:`check_one_tick` finite-differences the whole assembled operator against a
real tick. Neither is assumed.

**The comparator is the same operator with the coupling deleted.** Non-normality
is not basis-free, so a field number is only worth its comparison. The
block-diagonal part of `M` -- drop the `F_p^T F_pbar` term, keep everything else
-- is the same graph with every cell talking to nobody, in the same coordinates.
It is the honest null, and it contains #274's per-cell loop
`K(J_chart + J_stalk A D)` as its own diagonal.

Three reads, in the ticket's order:

- **non-normality**, `|M^T M - M M^T|_F / |M|_F^2`, ADR-0023's instrument, on
  the field operator and on its uncoupled comparator;
- **Henrici departure** `sqrt(|M|_F^2 - Sum |lambda_i|^2)`, normalised by
  `|M|_F` (`--eigs`);
- **transient amplification** `|M^t|_2` against `rho^t`, which is what
  non-normality is *for* and the only one of the three a coherent structure
  could live in.

Usage::

    PYTHONPATH=src python prototypes/field-operator-375/read.py --ticks 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_BENCH = str(Path(__file__).resolve().parents[2] / "benchmarks")
if _BENCH not in sys.path:
    sys.path.append(_BENCH)
from untrained_fixed_point import build, teaching  # noqa: E402

from patchworks.bias_selection import _map_jacobian  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402

#: #206's ladder, truncated to what this read is affordable at.
CHECKPOINTS = (0, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 30_000)


# -- the pieces, read off the live agent ----------------------------------


def relay_matrices(agent) -> torch.Tensor:
    """`[predicting, n, n]`: `A_v = I - g_v Sum_{e in v} F_ev^T F_ev`.

    #274's `relay_matrices`, unchanged and for the same reason: read off the
    live maps, so a transport step that has moved them moves this too.
    """
    sheaf = agent.sheaf
    maps = sheaf.maps
    with torch.no_grad():
        gram = torch.bmm(maps.maps.transpose(1, 2), maps.maps)
        per_cell = torch.zeros(
            len(agent.dome.cells), maps.stalk_width, maps.stalk_width, dtype=gram.dtype
        )
        per_cell.index_add_(0, maps.owner, gram)
        n = agent.dome.shape.n
        rows = torch.tensor(list(agent.dome.predicting), dtype=torch.long)
        block = per_cell[rows][:, :n, :n]
        gain = sheaf.gain[rows].view(-1, 1, 1)
        return torch.eye(n, dtype=block.dtype).expand_as(block) - gain * block


def encode_halves(agent) -> tuple[torch.Tensor, torch.Tensor]:
    """`(J_chart, J_stalk)` in the region the run is in, `[predicting, k, .]`.

    Read off `encode_parts` on the *live* buffers, which is what ADR-0019 asks
    of a live read: the body measured is the body running.

    **At the point the next tick will read, not the point the last one did.**
    #274 evaluates at `prior_charts` / `prior_evidence` -- the pair the last
    inference phase ran on -- because it is reporting the loop the run has just
    taken. This operator is the one the run is *about to* apply, and it is
    checked by finite-differencing an actual next tick, so it has to be
    linearised at that tick's own inputs. The two differ by one tick, and with
    a region-dependent `encode` that is not a rounding difference:
    :func:`check_one_tick` reads ~0.85 relative error at the wrong point and
    ~1e-9 at this one.
    """
    sheaf = agent.sheaf
    body = sheaf.body
    k = body.shape.k
    with torch.no_grad():
        pre_activation, _ = body.encode_parts(
            sheaf.charts, sheaf.evidence(), sheaf.biases
        )
        active = (pre_activation > 0).to(pre_activation.dtype)
        jacobian = _map_jacobian(
            body.encode_hidden_weight, body.encode_output_weight, active
        ).double()
    return jacobian[:, :, :k], jacobian[:, :, k:]


def coupling_blocks(agent) -> list[tuple[int, int, torch.Tensor]]:
    """`(v, u, g_v (F_p^T F_pbar)[:n,:n])` for every ordered adjacent predicting pair.

    The whole of the cross-cell term. `p` is `v`'s endpoint of the shared edge
    and `pbar` is `u`'s, so this is one restriction *out* of `u` and one *back
    into* `v` -- restriction maps composed along the graph, which is the object
    #375 §2 says has never been measured.

    A boundary partner is absent by construction, not by omission: the world
    overwrites its stalk after every tick, so its contribution to `v` is
    exogenous and its derivative is zero. Padded columns past a cell's own
    stalk are structurally zero on both sides, which is what makes the `[:n,:n]`
    slice the whole of the product (#274's argument for the same slice).
    """
    dome = agent.dome
    sheaf = agent.sheaf
    maps = sheaf.maps.maps.detach().double()
    n = dome.shape.n
    row_of = {cell_id: i for i, cell_id in enumerate(dome.predicting)}
    blocks: list[tuple[int, int, torch.Tensor]] = []
    for edge in dome.edges:
        ends = (edge.u, edge.v)
        for side, cell_id in enumerate(ends):
            partner = ends[1 - side]
            if cell_id not in row_of or partner not in row_of:
                continue
            p = pair_index(edge.id, side)
            p_bar = pair_index(edge.id, 1 - side)
            gain = float(sheaf.gain[cell_id])
            product = gain * (maps[p].T @ maps[p_bar])[:n, :n]
            blocks.append((row_of[cell_id], row_of[partner], product))
    return blocks


def assemble(agent) -> dict[str, object]:
    """The field operator `M` on `(z, s)`, and its uncoupled comparator.

    `M` is dense `[cells(k+n), cells(k+n)]` in the block order `z` then `s`.
    The comparator is not assembled here: a block-diagonal matrix's commutator
    is block-diagonal, so every norm it is wanted for sums over `[k+n, k+n]`
    blocks, and those are what is returned.
    """
    dome = agent.dome
    sheaf = agent.sheaf
    k, n = dome.shape.k, dome.shape.n
    cells = len(dome.predicting)

    chart_half, stalk_half = encode_halves(agent)
    operator = sheaf.operators.K.detach().double()
    decode = sheaf.body.decode_weight.detach().double()
    relay = relay_matrices(agent).double()

    k_chart = operator @ chart_half              # dz'/dz, per cell
    k_stalk = operator @ stalk_half              # dz'/ds, per cell
    read_back = relay @ decode                   # ds'/dz', per cell

    # The per-cell (k+n) x (k+n) block -- the whole cell, coupling deleted.
    diagonal = torch.zeros(cells, k + n, k + n, dtype=torch.float64)
    diagonal[:, :k, :k] = k_chart
    diagonal[:, :k, k:] = k_stalk
    diagonal[:, k:, :k] = read_back @ k_chart
    diagonal[:, k:, k:] = read_back @ k_stalk

    width = k + n
    size = cells * width
    field = torch.zeros(size, size, dtype=torch.float64)
    for i in range(cells):
        a = i * width
        field[a : a + width, a : a + width] = diagonal[i]
    coupling = coupling_blocks(agent)
    for v, u, product in coupling:
        a, b = v * width, u * width
        # ds_v/dz_u: the neighbour's chart, decoded, restricted out and back.
        field[a + k : a + width, b : b + k] += product @ decode
    return {
        "field": field,
        "diagonal": diagonal,
        "coupling_terms": len(coupling),
        "width": width,
        "cells": cells,
    }


# -- the checks -----------------------------------------------------------


def owned_pairs(agent) -> torch.Tensor:
    """`[pairs]` bool: the endpoints a predicting cell holds."""
    dome = agent.dome
    owned = torch.zeros(agent.sheaf.maps.pairs, dtype=torch.bool)
    predicting = set(dome.predicting)
    for edge in dome.edges:
        for side, cell_id in enumerate((edge.u, edge.v)):
            if cell_id in predicting:
                owned[pair_index(edge.id, side)] = True
    return owned


def check_broadcast_is_last_prediction(agent) -> dict[str, float]:
    """Is `broadcast(t-1)` the restriction of the prediction `z(t)` decodes to?

    The load-bearing identity. If it holds, the neighbour's chart enters `v`'s
    evidence with no extra state and `(z, s)` is closed -- which is why this
    operator is `6600 x 6600` and not `17512 x 17512`. Checked only on pairs a
    predicting cell owns; a boundary cell's broadcast is the world's.
    """
    sheaf = agent.sheaf
    with torch.no_grad():
        prediction = sheaf.body.decode(sheaf.charts, sheaf.biases)
        rebuilt = sheaf.layout.empty(dtype=prediction.dtype)
        rebuilt[sheaf.layout.predicting_positions] = prediction
        outgoing = sheaf.maps.restrict(rebuilt[sheaf.layout.pair_positions])
        owned = owned_pairs(agent)
        actual = sheaf.broadcast[owned]
        error = torch.linalg.vector_norm(outgoing[owned] - actual, dim=-1)
        scale = torch.linalg.vector_norm(actual, dim=-1).clamp_min(1e-30)
    return {
        "pairs_checked": int(owned.sum()),
        "max_relative_error": float((error / scale).max()),
        "median_relative_error": float((error / scale).median()),
    }


def promote(agent) -> None:
    """Put the run's own buffers in float64, so the check can be a real one.

    The tick runs in float32, and a finite difference in float32 floors at
    ~3e-4 relative however good the algebra is -- the difference of two
    order-one numbers carrying a 1e-3 signal has about that much left. Measured:
    the check reads `3.2e-4` at its best scale in float32 and falls monotonically
    to `6.1e-12` in float64, which is the signature of round-off and not of a
    missing term. `--float64` is therefore the *verification* mode; the reported
    numbers come from the float32 run the rest of the record was read on.
    """
    sheaf = agent.sheaf
    for name in (
        "stalks", "charts", "broadcast", "incoming", "prediction",
        "prior_charts", "prior_evidence",
    ):
        setattr(sheaf, name, getattr(sheaf, name).double())
    sheaf._gain_per_component = sheaf._gain_per_component.double()


def check_one_tick(
    agent, field: torch.Tensor, *, scale: float = 1e-3, trials: int = 3, seed: int = 0
) -> dict[str, float]:
    """Finite-difference the assembled operator against a real tick.

    Perturbs `(z, s)`, restores `broadcast` to the value the identity above says
    a perturbed chart implies, runs `inference_phase` then
    `message_passing_phase` with both rules off, and compares the difference in
    `(z, s)` against `M d`. This is the only thing standing between the numbers
    below and an algebra error.
    """
    sheaf = agent.sheaf
    dome = agent.dome
    k, n = dome.shape.k, dome.shape.n
    cells = len(dome.predicting)
    names = (
        "charts", "stalks", "broadcast", "incoming", "prediction",
        "prior_charts", "prior_evidence",
    )
    saved = {name: getattr(sheaf, name).clone() for name in names}
    ticks = sheaf.ticks
    fold_state = sheaf.fold_read.state()
    owned = owned_pairs(agent)
    generator = torch.Generator().manual_seed(seed)

    def run_once(delta_z: torch.Tensor, delta_s: torch.Tensor) -> torch.Tensor:
        for name in names:
            setattr(sheaf, name, saved[name].clone())
        sheaf.ticks = ticks
        sheaf.fold_read.load(fold_state)
        with torch.no_grad():
            sheaf.charts = sheaf.charts + delta_z
            # broadcast is F(D z + b): a perturbed chart implies a perturbed
            # broadcast, and leaving it stale would measure a different system.
            prediction = sheaf.body.decode(sheaf.charts, sheaf.biases)
            rebuilt = sheaf.layout.empty(dtype=prediction.dtype)
            rebuilt[sheaf.layout.predicting_positions] = prediction
            outgoing = sheaf.maps.restrict(rebuilt[sheaf.layout.pair_positions])
            broadcast = sheaf.broadcast.clone()
            broadcast[owned] = outgoing[owned]
            sheaf.broadcast = broadcast
            sheaf.incoming = (
                broadcast.reshape(-1, 2, sheaf.maps.edge_width)
                .flip(1)
                .reshape_as(broadcast)
            )
            sheaf.stalks[sheaf.layout.predicting_positions] += delta_s
            sheaf.inference_phase()
            sheaf.message_passing_phase()
            out = torch.cat(
                [
                    sheaf.charts.double(),
                    sheaf.stalks[sheaf.layout.predicting_positions].double(),
                ],
                dim=-1,
            )
        return out.reshape(-1)

    base = run_once(torch.zeros(cells, k), torch.zeros(cells, n))
    errors: list[float] = []
    for _ in range(trials):
        step = torch.randn(cells, k + n, generator=generator) * scale
        observed = run_once(step[:, :k], step[:, k:]) - base
        predicted = field @ step.reshape(-1).double()
        error = torch.linalg.vector_norm(observed - predicted)
        magnitude = torch.linalg.vector_norm(predicted).clamp_min(1e-30)
        errors.append(float(error / magnitude))
    for name in names:
        setattr(sheaf, name, saved[name])
    sheaf.ticks = ticks
    sheaf.fold_read.load(fold_state)
    return {
        "trials": trials,
        "perturbation_scale": scale,
        "max_relative_error": float(np.max(errors)),
        "median_relative_error": float(np.median(errors)),
    }


# -- the reads ------------------------------------------------------------


def non_normality(matrix: torch.Tensor) -> float:
    """`|M^T M - M M^T|_F / |M|_F^2` -- ADR-0023's instrument, #166's `0.0504`."""
    gram = matrix.T @ matrix
    gram -= matrix @ matrix.T
    frobenius = float(torch.linalg.matrix_norm(matrix, ord="fro"))
    return float(torch.linalg.matrix_norm(gram, ord="fro")) / max(frobenius**2, 1e-30)


def block_non_normality(blocks: torch.Tensor) -> float:
    """The same instrument on a block-diagonal operator, without assembling it.

    A block-diagonal matrix's commutator is block-diagonal with the blocks'
    commutators on it, so both Frobenius norms sum over blocks.
    """
    gram = blocks.transpose(1, 2) @ blocks - blocks @ blocks.transpose(1, 2)
    commutator = float(torch.linalg.matrix_norm(gram, ord="fro").pow(2).sum().sqrt())
    frobenius_squared = float(torch.linalg.matrix_norm(blocks, ord="fro").pow(2).sum())
    return commutator / max(frobenius_squared, 1e-30)


def per_block_non_normality(blocks: torch.Tensor) -> np.ndarray:
    """The instrument per block -- #166 reported the median of exactly this."""
    gram = blocks.transpose(1, 2) @ blocks - blocks @ blocks.transpose(1, 2)
    commutator = torch.linalg.matrix_norm(gram, ord="fro")
    frobenius = torch.linalg.matrix_norm(blocks, ord="fro")
    return (commutator / frobenius.pow(2).clamp_min(1e-30)).numpy()


def amplification(
    matrix: torch.Tensor, horizon: int, *, iterations: int = 50, seed: int = 0
) -> list[float]:
    """`|M^t|_2` for `t = 1..horizon`, by power iteration through matvecs.

    Never forms `M^t`. The whole point of the curve is that a near-normal
    operator has `|M^t|_2 ~ rho^t` from `t = 1`, while a non-normal one bulges
    above it before it decays -- and the bulge is the only place a transiently
    sustained structure could live.
    """
    generator = torch.Generator().manual_seed(seed)
    out: list[float] = []
    power = matrix.clone()
    for _ in range(horizon):
        vector = torch.randn(power.shape[1], generator=generator, dtype=torch.float64)
        vector /= torch.linalg.vector_norm(vector)
        value = 0.0
        for _ in range(iterations):
            moved = power.T @ (power @ vector)
            norm = torch.linalg.vector_norm(moved)
            if norm == 0:
                value = 0.0
                break
            vector = moved / norm
            value = float(norm)
        out.append(float(np.sqrt(value)))
        power = matrix @ power
    return out


def block_amplification(blocks: torch.Tensor, horizon: int) -> list[float]:
    """`max_v |B_v^t|_2` over the uncoupled blocks -- the comparator's curve."""
    powers = blocks.clone()
    out: list[float] = []
    for _ in range(horizon):
        out.append(float(torch.linalg.matrix_norm(powers, ord=2).amax()))
        powers = torch.bmm(blocks, powers)
    return out


def block_henrici(blocks: torch.Tensor) -> dict[str, float]:
    """:func:`henrici` on a block-diagonal operator, without assembling it.

    A block-diagonal matrix's spectrum is the union of its blocks' and its
    Frobenius norm sums over them, so both terms of the departure decompose.
    Cheap where the dense form is a `6600 x 6600` nonsymmetric eigenproblem.
    """
    eigenvalues = torch.linalg.eigvals(blocks)
    frobenius_squared = float(torch.linalg.matrix_norm(blocks, ord="fro").pow(2).sum())
    mass = float(eigenvalues.abs().pow(2).sum())
    departure = float(np.sqrt(max(frobenius_squared - mass, 0.0)))
    return {
        "departure": departure,
        "departure_over_frobenius": departure / max(np.sqrt(frobenius_squared), 1e-30),
        "spectral_radius": float(eigenvalues.abs().amax()),
    }


def henrici(matrix: torch.Tensor) -> dict[str, float]:
    """`sqrt(|M|_F^2 - Sum |lambda_i|^2)`, and the same over `|M|_F`.

    Zero exactly for a normal matrix. Expensive: a dense nonsymmetric
    eigendecomposition of the whole field operator.
    """
    eigenvalues = torch.linalg.eigvals(matrix)
    frobenius_squared = float(torch.linalg.matrix_norm(matrix, ord="fro").pow(2))
    mass = float(eigenvalues.abs().pow(2).sum())
    departure = float(np.sqrt(max(frobenius_squared - mass, 0.0)))
    return {
        "departure": departure,
        "departure_over_frobenius": departure / max(np.sqrt(frobenius_squared), 1e-30),
        "spectral_radius": float(eigenvalues.abs().amax()),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "max": float(q[4]),
    }


def block_diagonal(diagonal: torch.Tensor) -> torch.Tensor:
    return torch.block_diag(*[diagonal[i] for i in range(diagonal.shape[0])])


def row(agent, reached: int, *, horizon: int, eigs: bool) -> dict:
    parts = assemble(agent)
    field, diagonal = parts["field"], parts["diagonal"]
    k = agent.dome.shape.k

    # #166's own object, so the field number lands against a reproduced anchor
    # rather than a quoted one.
    cell_operator = agent.sheaf.operators.K.detach().double()

    # Per cell, the (k+n) block's own spectral radius. The aggregate `rho` is
    # the *max* over these, so a field that reads expansive can still be a
    # dissipative medium with a few hot cells -- and which of those it is
    # decides whether #374's driven-dissipative literature applies at all. #274
    # measured the median cell contracting on the chart loop alone; this is the
    # same question asked of the whole cell, relay included, and never as a
    # graph-wide average (#181).
    block_radii = torch.linalg.eigvals(diagonal).abs().amax(dim=-1).numpy()

    out: dict[str, object] = {
        "ticks": reached,
        "size": int(field.shape[0]),
        "block_spectral_radius": quantiles(block_radii),
        "cells_expansive": int((block_radii >= 1.0).sum()),
        "cells": int(diagonal.shape[0]),
        "coupling_terms": parts["coupling_terms"],
        "coupling_share_of_norm": float(
            torch.linalg.matrix_norm(field - block_diagonal(diagonal), ord="fro")
            / torch.linalg.matrix_norm(field, ord="fro")
        ),
        "non_normality": {
            "field": non_normality(field),
            "uncoupled": block_non_normality(diagonal),
            "cell_K_166": float(np.median(per_block_non_normality(cell_operator))),
            "per_cell_uncoupled": quantiles(per_block_non_normality(diagonal)),
        },
        "checks": {
            "broadcast_identity": check_broadcast_is_last_prediction(agent),
            "one_tick": check_one_tick(agent, field),
        },
    }
    if horizon:
        curve = amplification(field, horizon)
        out["amplification"] = {
            "field": curve,
            "uncoupled": block_amplification(diagonal, horizon),
        }
    if eigs:
        out["henrici"] = {
            "field": henrici(field),
            "uncoupled": block_henrici(diagonal),
        }
        if horizon:
            radius = out["henrici"]["field"]["spectral_radius"]
            powers = [radius**t for t in range(1, horizon + 1)]
            out["amplification"]["rho_powers"] = powers
            # The transient excess: what the operator does above what its own
            # spectrum permits. `1` everywhere is a normal operator; a bulge
            # above `1` is the only place a transiently sustained structure
            # could live, and its size is what #374's sources are about.
            out["amplification"]["excess_over_rho"] = [
                a / p for a, p in zip(out["amplification"]["field"], powers)
            ]
            uncoupled_radius = out["henrici"]["uncoupled"]["spectral_radius"]
            out["amplification"]["uncoupled_excess_over_rho"] = [
                a / uncoupled_radius**t
                for t, a in enumerate(out["amplification"]["uncoupled"], start=1)
            ]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dome", default="real", choices=("small", "real"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=2_000)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument(
        "--eigs",
        action="store_true",
        help="Henrici departure: a dense eigendecomposition of the whole field "
        "operator, minutes not seconds",
    )
    parser.add_argument(
        "--float64",
        action="store_true",
        help="run the tick itself in double, which is what makes "
        "check_one_tick a machine-precision check rather than a float32 one",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("prototypes/field-operator-375")
    )
    args = parser.parse_args(argv)
    if args.float64:
        torch.set_default_dtype(torch.float64)

    started = time.time()
    checkpoints = [c for c in CHECKPOINTS if c <= args.ticks]
    rows: list[dict] = []

    if 0 in checkpoints:
        # The undriven control: one tick off construction, so the buffers the
        # read needs exist and nothing has learned. `K = a.I` and the maps are
        # the draw, so this is the closest thing to a zero the rig has.
        _, control = build(args.dome, args.split, args.seed)
        if args.float64:
            promote(control)
        control.sheaf.inference_phase()
        control.sheaf.message_passing_phase()
        control.sheaf.ticks += 1
        rows.append(row(control, 0, horizon=args.horizon, eigs=args.eigs))
        print(f"--- construction ({time.time() - started:.0f}s)", flush=True)
        print(json.dumps(rows[-1]["non_normality"], indent=2), flush=True)
        print(json.dumps(rows[-1]["checks"], indent=2), flush=True)
        del control

    _, agent = build(args.dome, args.split, args.seed)
    if args.float64:
        promote(agent)
    remaining = [c for c in checkpoints if c > 0]
    for _outcome in teaching(agent, args.ticks, seed=args.seed):
        reached = agent.sheaf.ticks
        if remaining and reached >= remaining[0]:
            remaining.pop(0)
            rows.append(row(agent, reached, horizon=args.horizon, eigs=args.eigs))
            print(f"--- tick {reached} ({time.time() - started:.0f}s)", flush=True)
            print(json.dumps(rows[-1]["non_normality"], indent=2), flush=True)
            print(json.dumps(rows[-1]["checks"], indent=2), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"375-{args.dome}-{args.split}-seed{args.seed}-{args.ticks}.json"
    payload = {
        "dome": args.dome,
        "split": args.split,
        "seed": args.seed,
        "ticks": args.ticks,
        "horizon": args.horizon,
        "eigs": args.eigs,
        "seconds": time.time() - started,
        "rows": rows,
    }
    (args.out / name).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
