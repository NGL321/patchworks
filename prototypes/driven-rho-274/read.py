"""`rho` of the **full** chart loop, read on the driven rig (#274).

[#271](https://github.com/NGL321/patchworks/issues/271) found that `tau`'s
instrument omits a real term of the chart's recurrence. `05:253`,
:func:`patchworks.bias_selection.measure` and the #206 rig's `regional_tau` all
slice `encode`'s Jacobian to `[:, :, :k]` -- the chart half -- on the stated
ground that *"the node stalk half is where this tick's evidence entered and is
not part of the loop"* (`bias_selection.py:592-595`). That is true of a boundary
cell and false of all 150 predicting cells: `inference_phase` overwrites the
predicting stalk with `D z + b`, `message_passing_phase` subtracts
`g_v Sum_e F_ev^T (F_ev x_v - y_e(t-1))`, and `evidence()` hands the result back
into `encode` next tick. So the stalk half is the cell's own chart returning
through a damped aperture, and the loop's true one-tick Jacobian is

    K @ (J_chart + J_stalk @ A_v @ D),   A_v = I - g_v Sum_e F_ev^T F_ev

#271 read this **at construction**, with no environment attached, so no boundary
stalk was ever written by a world. Its comparative claim (~2.2x, flat by depth)
is robust to that; its absolute `rho ~ 1.0` is not quotable until it is re-read
on a driven run. This is that read: the `real` dome, `split=train`, both rules
on, through the same `build`/`teaching` harness
`prototypes/live-fold-read-206/read.py` uses, at the same checkpoints, per cell
and never as a graph-wide average.

**The recurrence is checked, not assumed.** :func:`check_relay_identity`
reconstructs `evidence(t+1)` from the analytic form
`A_v (D z + b) + g_v Sum_e F_ev^T y_e` and compares it against the stalk the run
actually left behind, and :func:`check_chart_only_matches_206` reproduces the
#206 rig's `regional_tau` exactly, so the two readings differ in the relay term
and in nothing else.

Usage::

    PYTHONPATH=src python prototypes/driven-rho-274/read.py --ticks 30000
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

#: The same ceiling `bias_selection` and the #206 rig clamp at, so every `tau`
#: printed here is the same quantity read at a different moment.
RHO_CEILING = 1.0 - 1e-6

#: #206's checkpoint ladder, truncated to the horizon actually run.
CHECKPOINTS = (
    100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 30_000, 50_000,
    75_000, 100_000,
)


def relay_matrices(agent) -> torch.Tensor:
    """`[predicting cells, n, n]`: `A_v = I - g_v Sum_{e in v} F_ev^T F_ev`.

    Read off the live maps, so a transport step that has moved them since the
    last checkpoint moves this too. Padded rows and masked columns are
    structurally zero on both sides, so the sum over a pair's full padded map is
    the sum over its real one -- the same property `spread` relies on.
    """
    sheaf = agent.sheaf
    maps = sheaf.maps
    with torch.no_grad():
        gram = torch.bmm(maps.maps.transpose(1, 2), maps.maps)  # [pairs, s, s]
        per_cell = torch.zeros(
            len(agent.dome.cells), maps.stalk_width, maps.stalk_width,
            dtype=gram.dtype,
        )
        per_cell.index_add_(0, maps.owner, gram)
        n = agent.dome.shape.n
        rows = torch.tensor(list(agent.dome.predicting), dtype=torch.long)
        block = per_cell[rows][:, :n, :n]
        gain = sheaf.gain[rows].view(-1, 1, 1)
        return torch.eye(n, dtype=block.dtype).expand_as(block) - gain * block


def spectra(agent) -> dict[str, np.ndarray]:
    """Both radii, per predicting cell, in the region the run is in.

    `chart` is the #206 rig's object, bit for bit; `full` adds the relay term.
    Eigenvalues in float64 for the same reason #271 took them there: the
    difference under test is a factor of two on a radius near 1.
    """
    sheaf = agent.sheaf
    body = sheaf.body
    k = body.shape.k
    with torch.no_grad():
        pre_activation, _ = body.encode_parts(
            sheaf.prior_charts, sheaf.prior_evidence, sheaf.biases
        )
        active = (pre_activation > 0).to(pre_activation.dtype)
        jacobian = _map_jacobian(
            body.encode_hidden_weight, body.encode_output_weight, active
        ).double()
        chart_half, stalk_half = jacobian[:, :, :k], jacobian[:, :, k:]
        operator = sheaf.operators.K.double()
        relay = stalk_half @ relay_matrices(agent).double() @ body.decode_weight.double()

        chart_only = operator @ chart_half
        full = operator @ (chart_half + relay)
        radii = {
            "chart": torch.linalg.eigvals(chart_only).abs().amax(dim=-1),
            "full": torch.linalg.eigvals(full).abs().amax(dim=-1),
        }
        # How much of the loop the omitted term is, in norm, independent of the
        # spectrum: a radius can move for a reason a norm cannot see.
        share = (
            torch.linalg.matrix_norm(operator @ relay, ord=2)
            / torch.linalg.matrix_norm(chart_only, ord=2).clamp_min(1e-30)
        )
    out = {key: value.numpy() for key, value in radii.items()}
    out["relay_norm_share"] = share.numpy()
    return out


def tau_of(rho: np.ndarray) -> np.ndarray:
    return -1.0 / np.log(np.clip(rho, 1e-12, RHO_CEILING))


def check_relay_identity(agent) -> dict[str, float]:
    """Is the stalk the run left behind the analytic relay of the chart?

    `evidence(t+1) = A_v (D z + b) + g_v Sum_e F_ev^T y_e(t)`. Reconstructed
    here from the live buffers and compared against the node stalk itself. If
    this holds, `d evidence / d chart = A_v @ D` is a fact about the run rather
    than a claim about the source.
    """
    sheaf = agent.sheaf
    with torch.no_grad():
        prediction = sheaf.prediction
        relayed = torch.einsum("cij,cj->ci", relay_matrices(agent), prediction)
        # `+ g_v Sum_e F_ev^T y_e`, the exogenous half of the same subtraction.
        incoming_pull = sheaf.maps.spread(sheaf.incoming)
        scattered = torch.zeros_like(sheaf.stalks)
        scattered.index_add_(
            0, sheaf.layout.pair_positions.reshape(-1), incoming_pull.reshape(-1)
        )
        scattered.mul_(sheaf._gain_per_component)
        reconstructed = relayed + scattered[sheaf.layout.predicting_positions]
        actual = sheaf.evidence()
        error = torch.linalg.vector_norm(reconstructed - actual, dim=-1)
        scale = torch.linalg.vector_norm(actual, dim=-1).clamp_min(1e-30)
    return {
        "max_relative_error": float((error / scale).max()),
        "median_relative_error": float((error / scale).median()),
    }


#: Where the identity is checked. Tick 1 is exact -- `TransportRule` refuses
#: below `ticks < 2`, so the maps that wrote the stalk are still the maps read
#: here. From tick 2 on the rule has stepped since, so the residual is the
#: transport step's own size and not an error in the algebra; it is read at 10
#: as well so that size is on the record rather than assumed small.
IDENTITY_TICKS = (1, 10)


def check_chart_only_matches_206(agent, chart_rho: np.ndarray) -> float:
    """The chart-only radius here against the #206 rig's own `regional_tau`."""
    # Loaded by path under its own name: both rigs are called `read`, and a
    # plain import would find this one and cross-check it against itself.
    import importlib.util  # noqa: PLC0415

    source = Path(__file__).resolve().parents[1] / "live-fold-read-206" / "read.py"
    spec = importlib.util.spec_from_file_location("live_fold_read_206", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return float(np.abs(module.regional_tau(agent) - tau_of(chart_rho)).max())


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "max": float(q[4]),
    }


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def row(agent, reached: int, p_v: np.ndarray, degree: np.ndarray) -> dict:
    read = spectra(agent)
    chart, full = read["chart"], read["full"]
    ratio = full / np.maximum(chart, 1e-30)
    return {
        "ticks": reached,
        "rho_chart": quantiles(chart),
        "rho_full": quantiles(full),
        "ratio": quantiles(ratio),
        "relay_norm_share": quantiles(read["relay_norm_share"]),
        "tau_chart": quantiles(tau_of(chart)),
        "tau_full": quantiles(tau_of(full)),
        "cells_expansive_full": int((full >= 1.0).sum()),
        "cells_expansive_chart": int((chart >= 1.0).sum()),
        "corr_p_v_delta_rho": correlation(p_v, full - chart),
        "corr_degree_delta_rho": correlation(degree, full - chart),
        "per_cell": {
            "rho_chart": chart.tolist(),
            "rho_full": full.tolist(),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dome", default="real", choices=("small", "real"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=30_000)
    parser.add_argument("--out", type=Path, default=Path("prototypes/driven-rho-274"))
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    _, agent = build(args.dome, args.split, args.seed)
    dome = agent.dome
    cells = [dome.cells[c] for c in dome.predicting]
    levels = np.array([cell.index.level for cell in cells])
    p_v = dome.private_dimensions.numpy().astype(float)
    degree = np.zeros(len(cells))
    position = {cell.id: i for i, cell in enumerate(cells)}
    for edge in dome.edges:
        for cell_id in (edge.u, edge.v):
            if cell_id in position:
                degree[position[cell_id]] += 1
    checkpoints = [c for c in CHECKPOINTS if c <= args.ticks]

    print(f"dome={args.dome} split={args.split} seed={args.seed} ticks={args.ticks}")
    print(f"{len(cells)} predicting cells; levels 1..{levels.max()}")

    # The undriven control: the same instrument at tick 0, which is where #271
    # read the whole of its table.
    control = row(agent, 0, p_v, degree)
    print(
        f"  undriven control (tick 0): rho chart median "
        f"{control['rho_chart']['median']:.3f}, full median "
        f"{control['rho_full']['median']:.3f}, ratio median "
        f"{control['ratio']['median']:.2f}x"
    )

    rows = []
    checked: dict[str, float] = {}
    start = time.time()
    for reached, _outcome in enumerate(teaching(agent, args.ticks, args.seed), start=1):
        if reached in IDENTITY_TICKS:
            at = check_relay_identity(agent)
            checked[f"relay_identity_tick_{reached}"] = at
            print(
                f"  relay identity at tick {reached}: max relative error "
                f"{at['max_relative_error']:.2e}, median "
                f"{at['median_relative_error']:.2e}"
            )
        if reached == 10:
            checked["chart_only_vs_206_tau_max_abs"] = check_chart_only_matches_206(
                agent, spectra(agent)["chart"]
            )
            print(
                f"  chart-only tau against the #206 rig: "
                f"{checked['chart_only_vs_206_tau_max_abs']:.2e}\n"
            )
        if reached in checkpoints:
            entry = row(agent, reached, p_v, degree)
            rows.append(entry)
            rate = reached / (time.time() - start)
            print(
                f"  {reached:>7} ticks: rho chart median "
                f"{entry['rho_chart']['median']:6.3f}  rho full median "
                f"{entry['rho_full']['median']:6.3f}  ratio median "
                f"{entry['ratio']['median']:5.2f}x  expansive "
                f"{entry['cells_expansive_full']:>3}/{len(cells)}  "
                f"tau full median {entry['tau_full']['median']:8.3f}  "
                f"corr(p_v, d rho) {entry['corr_p_v_delta_rho']:+.3f}  "
                f"({rate:.0f} tick/s)",
                flush=True,
            )

    final = rows[-1]
    chart = np.array(final["per_cell"]["rho_chart"])
    full = np.array(final["per_cell"]["rho_full"])
    print("\n  AT THE HORIZON, by level (a reporting axis only; nothing is concluded from it)")
    print("    level  cells  p_v  rho chart  rho full  ratio")
    by_level = []
    for level in sorted(set(levels.tolist())):
        mask = levels == level
        entry = {
            "level": int(level), "cells": int(mask.sum()),
            "p_v": float(np.median(p_v[mask])),
            "rho_chart": float(np.median(chart[mask])),
            "rho_full": float(np.median(full[mask])),
            "ratio": float(np.median(full[mask] / np.maximum(chart[mask], 1e-30))),
        }
        by_level.append(entry)
        print(
            f"    L{entry['level']:<5} {entry['cells']:>5} {entry['p_v']:>4.0f}  "
            f"{entry['rho_chart']:>9.3f} {entry['rho_full']:>9.3f} "
            f"{entry['ratio']:>6.2f}x"
        )

    print(
        f"\n    Cells with rho_full >= 1: {final['cells_expansive_full']}/{len(cells)}; "
        f"with rho_chart >= 1: {final['cells_expansive_chart']}/{len(cells)}."
    )
    print(
        f"    corr(p_v, delta rho) = {final['corr_p_v_delta_rho']:+.3f}; "
        f"corr(degree, delta rho) = {final['corr_degree_delta_rho']:+.3f}."
    )

    tag = args.tag or f"{args.dome}-{args.split}-seed{args.seed}-{args.ticks}"
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dome": args.dome, "split": args.split, "seed": args.seed,
        "ticks": args.ticks, "cells": len(cells),
        "elapsed_minutes": (time.time() - start) / 60,
        "checks": checked,
        "undriven_control": control,
        "by_level_at_horizon": by_level,
        "levels": levels.tolist(),
        "p_v": p_v.tolist(),
        "degree": degree.tolist(),
        "cell_ids": [cell.id for cell in cells],
        "checkpoints": rows,
    }
    (args.out / f"274-{tag}.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  Written to {args.out}/274-{tag}.json")


if __name__ == "__main__":
    main()
