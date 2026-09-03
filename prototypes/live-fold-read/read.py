"""What the live fold read reports, and how long the burn-in is (#202).

ADR-0019 moved the fold-margin check off construction and onto the run, and
[#197](https://github.com/NGL321/patchworks/issues/197) built the instrument
(:class:`patchworks.tick.FoldRead`) and deliberately did not run it. This runs
it. `02-tick-semantics.md` states the bound as holding *after a burn-in* and
names no count; [#178](https://github.com/NGL321/patchworks/issues/178) found
0.90x permitted at 2,000 ticks, which says the breach is real and says nothing
about when it ends.

**The horizon is 100,000 ticks, not 30,000.** #178 read the disagreement floor
to 100k and found the quantity wanders 3.8x with no trend, with 30k sitting
near a local high. Any burn-in read to 30k would inherit that.

**The comparison is `offset < margin`, with no second gain in it.** `02` writes
the bound `gain_v x offset < margin_v`, inherited from the demoted `gamma x
floor` form in which the divisor was the *un-gained* disagreement floor.
:attr:`patchworks.tick.FoldRead.offset` is read off the **gained** displacement
-- `tick.py` applies `_gain_per_component` and then observes -- so the live
comparison is direct, and is exactly what
:attr:`patchworks.tick.FoldRead.reconciliation_reaches` computes. Multiplying by
`gain_v` again would divide the offset by 8 a second time and report a burn-in
that is not the run's. :func:`check_gain_convention` asserts this against the
un-gained delta rather than trusting the reading.

**Three quantities, read per cell and never per level.**
[#190](https://github.com/NGL321/patchworks/issues/190) made `gain_v` uniform
across the interior and #160 struck the depth claim without replacing it, so a
per-level summary can only hide the binding cell. Levels are kept as a
reporting axis and nothing is concluded from them.

1. **The burn-in** -- the tick beyond which no cell breaches again.
2. **The attribution** -- whether region crossings land on breaching ticks. Dwell
   says a cell left its region; only the comparison says whether reconciliation
   is what moved it (ADR-0007's named failure).
3. **ADR-0005's precondition** -- measured dwell against each cell's own `tau`,
   falsifiable on the run for the first time since #160 re-sourced it off the
   margin proxy.

Usage::

    PYTHONPATH=src python prototypes/live-fold-read/read.py --ticks 100000
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

from patchworks.bias_selection import (  # noqa: E402
    DEFAULT_SAFETY_FACTOR,
    _map_jacobian,
)

#: `tau = -1/ln rho` diverges as `rho -> 1`; `bias_selection` clamps the radius
#: for the same reason, and this uses the same ceiling so the two `tau` are the
#: same quantity read at two moments.
RHO_CEILING = 1.0 - 1e-6

#: Where the run is read. #178's 30k is kept as the control it now is, and the
#: horizon is 100k on its finding.
CHECKPOINTS = (
    100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000,
    30_000, 50_000, 75_000, 100_000,
)


def check_gain_convention(agent) -> tuple[float, float]:
    """Assert `FoldRead.offset` is the **gained** displacement, not the raw one.

    The whole answer turns on this: `02`'s written bound carries a `gain_v` the
    instrument has already applied. Rather than read that off the source and
    hope, reconstruct the un-gained delta the way #178's `per_cell_floor` does
    and check the ratio is the gain.
    """
    sheaf = agent.sheaf
    contribution = sheaf.maps.spread(sheaf.broadcast - sheaf.incoming)
    delta = torch.zeros_like(sheaf.stalks)
    delta.index_add_(0, sheaf.layout.pair_positions.reshape(-1), contribution.reshape(-1))
    ungained = torch.linalg.vector_norm(
        delta[sheaf.layout.predicting_positions], dim=-1
    )
    gained = sheaf.fold_read.offset
    gain = sheaf.gain[torch.tensor(list(agent.dome.predicting))]
    live = ungained > 1e-12
    ratio = (gained[live] / ungained[live]).median().item()
    return ratio, float(gain.median())


def regional_tau(agent) -> np.ndarray:
    """`[predicting cells]`: `tau = -1/ln rho(K @ J_chart)` in the region it is in.

    The chart's **direct** round trip, the same object
    :func:`patchworks.bias_selection.measure` reads on the construction sweep --
    `encode`'s Jacobian restricted to the chart half of its input, then the
    cell's own `K`. Read off the body's own forward path, re-run on the pair the
    last inference phase actually read (`prior_charts`, `prior_evidence`), so a
    swapped body cannot leave this measuring one that is not running.

    **It is one of two routes from `chart(t)` to `chart(t+1)`, and this rig
    reports only this one** (#271, #274). `decode` writes `D chart + b` into the
    node stalk, message passing damps it by `A_v = I - g_v Sum_e F_ev^T F_ev`,
    and `encode`'s stalk half returns it next tick, so the full recurrence is
    `K (J_chart + J_stalk A_v D)`. Unlike the construction sweep this rig *can*
    reach `sheaf.maps` and `sheaf.gain`, so the omission here is not a
    limitation -- it is simply that this rig predates the finding.

    **The side-by-side reading is `prototypes/driven-rho-274/read.py`**, which
    reports both radii per cell on the same driven build and pins itself to this
    function: its `check_chart_only_matches_206` runs *this* code on the same
    live state and agrees to ~1e-5, so the two rigs differ in the relay term and
    in nothing else. This one is left as it ran, because `206-read.json` beside
    it is what it produced and a rig edited after its run is no longer the rig
    that ran. Read the two together; do not quote this `tau` as the loop's.
    Driven, over nine seeds, the full loop's `rho` is 1.70x to 2.12x this one.
    """
    sheaf = agent.sheaf
    body = sheaf.body
    with torch.no_grad():
        pre_activation, _ = body.encode_parts(
            sheaf.prior_charts, sheaf.prior_evidence, sheaf.biases
        )
        active = (pre_activation > 0).to(pre_activation.dtype)
        through_encode = _map_jacobian(
            body.encode_hidden_weight, body.encode_output_weight, active
        )[:, :, : body.shape.k]
        jacobian = torch.bmm(sheaf.operators.K, through_encode)
        rho = torch.linalg.eigvals(jacobian).abs().amax(dim=-1)
        tau = -1.0 / rho.clamp(min=1e-12, max=RHO_CEILING).log()
    return tau.numpy()


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "max": float(q[4]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dome", default="real", choices=("small", "real"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=Path("prototypes/live-fold-read"))
    args = parser.parse_args(argv)

    _, agent = build(args.dome, args.split, args.seed)
    read = agent.sheaf.fold_read
    cells = [cell for cell in agent.dome.cells if not cell.is_boundary]
    levels = np.array([cell.index.level for cell in cells])
    n = len(cells)
    checkpoints = [c for c in CHECKPOINTS if c <= args.ticks]

    print(f"dome={args.dome} split={args.split} seed={args.seed} ticks={args.ticks}")
    print(f"{n} predicting cells, {len(agent.dome.cells)} total; levels 1..{levels.max()}")
    print("the bound is `offset < margin` on the gained displacement (see module docstring)\n")

    # Per tick, three 150-vectors and nothing that averages.
    ratio_max = np.zeros(args.ticks)          # worst cell's offset/margin, per tick
    breaching = np.zeros(args.ticks, dtype=np.int32)   # how many cells breach
    last_breach = np.full(n, -1, dtype=np.int64)       # per cell, last tick it breached
    breach_count = np.zeros(n, dtype=np.int64)
    # The attribution: does a crossing land on the tick after a breach?
    crossings_total = np.zeros(n, dtype=np.int64)
    crossings_after_breach = np.zeros(n, dtype=np.int64)
    prior_breach = np.zeros(n, dtype=bool)
    prior_crossings = np.zeros(n)
    checkpoint_rows = []
    prior_checkpoint = (0, np.zeros(n))

    start = time.time()
    for reached, _outcome in enumerate(teaching(agent, args.ticks, args.seed), start=1):
        margin = read.margin.numpy()
        offset = read.offset.numpy()
        ratio = np.divide(offset, margin, out=np.full(n, np.inf), where=margin > 0)
        breach = ratio >= 1.0
        i = reached - 1
        ratio_max[i] = ratio.max()
        breaching[i] = int(breach.sum())
        last_breach[breach] = reached
        breach_count += breach

        crossings = read.crossings.numpy()
        crossed = (crossings - prior_crossings) > 0
        crossings_total += crossed
        crossings_after_breach += crossed & prior_breach
        prior_crossings = crossings.copy()
        prior_breach = breach.copy()

        if reached == 10:
            ratio_check, gain = check_gain_convention(agent)
            print(
                f"  gain convention: FoldRead.offset / un-gained delta = "
                f"{ratio_check:.6f}, gain_v = {gain:.6f} "
                f"-> {'gained' if abs(ratio_check - gain) < 1e-4 else 'MISMATCH'}\n"
            )

        if reached in checkpoints:
            tau = regional_tau(agent)
            since, since_crossings = prior_checkpoint
            window = reached - since
            window_dwell = window / (1.0 + (crossings - since_crossings))
            row = {
                "ticks": reached,
                "breaching_cells": int(breach.sum()),
                "ratio": quantiles(ratio),
                "margin": quantiles(margin),
                "offset": quantiles(offset),
                "cumulative_dwell": quantiles(read.dwell.numpy()),
                "window_dwell": quantiles(window_dwell),
                "window": window,
                "tau": quantiles(tau),
                "dwell_over_tau": quantiles(window_dwell / np.maximum(tau, 1e-12)),
                "cells_dwell_ge_safety_tau": int(
                    (window_dwell >= DEFAULT_SAFETY_FACTOR * tau).sum()
                ),
                "cells_dwell_le_2": int((window_dwell <= 2.0).sum()),
            }
            checkpoint_rows.append(row)
            prior_checkpoint = (reached, crossings.copy())
            rate = reached / (time.time() - start)
            print(
                f"  {reached:>7} ticks: breaching {row['breaching_cells']:>3}/{n}  "
                f"max ratio {row['ratio']['max']:9.3f}  median ratio "
                f"{row['ratio']['median']:7.4f}  window dwell median "
                f"{row['window_dwell']['median']:7.2f}  tau median "
                f"{row['tau']['median']:8.3f}  ({rate:.0f} tick/s)",
                flush=True,
            )

    elapsed = time.time() - start
    ticks = args.ticks

    # -- the burn-in -------------------------------------------------------
    graph_last = int(last_breach.max())
    print(f"\n  Read in {elapsed/60:.1f} min.\n")
    print("  THE BURN-IN")
    print(f"    Graph-wide: the last tick any cell breached is {graph_last}.")
    if graph_last >= ticks:
        print(
            f"    The bound is still being breached at the horizon ({ticks}). "
            "There is no burn-in at this horizon: the count does not exist."
        )
    else:
        print(f"    `offset < margin` holds at every cell from tick {graph_last + 1} on.")
    tail = min(10_000, ticks)
    tail_breaches = int((breaching[ticks - tail:] > 0).sum())
    print(
        f"    In the last {tail} ticks, {tail_breaches} ticks carried at least one "
        f"breaching cell ({tail_breaches/tail*100:.2f}%)."
    )
    ever = int((last_breach > 0).sum())
    print(f"    {ever}/{n} cells breached at least once; {n - ever} never did.")
    per_cell = {
        "never_breached": int((last_breach < 0).sum()),
        "last_breach_quantiles": quantiles(last_breach[last_breach > 0].astype(float))
        if ever
        else None,
        "breach_fraction": quantiles(breach_count / ticks),
    }
    if ever:
        q = per_cell["last_breach_quantiles"]
        print(
            f"    Per cell, the last breach: median {q['median']:.0f}, "
            f"p75 {q['p75']:.0f}, max {q['max']:.0f}."
        )

    # Where the breach density actually falls away, in decades.
    print("\n    Breach density by decade of the run:")
    edges = [0, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, ticks]
    decades = []
    for lo, hi in zip(edges, edges[1:]):
        if lo >= ticks:
            break
        hi = min(hi, ticks)
        window = breaching[lo:hi]
        entry = {
            "from": lo, "to": hi,
            "mean_breaching_cells": float(window.mean()),
            "ticks_with_any_breach": int((window > 0).sum()),
            "share_of_ticks": float((window > 0).mean()),
        }
        decades.append(entry)
        print(
            f"      {lo:>6}-{hi:<6}  mean breaching cells {entry['mean_breaching_cells']:7.2f}"
            f"   ticks with any breach {entry['share_of_ticks']*100:6.2f}%"
        )

    # -- the attribution ---------------------------------------------------
    print("\n  THE ATTRIBUTION: does reconciliation cost a cell its region?")
    total_c = int(crossings_total.sum())
    after_b = int(crossings_after_breach.sum())
    breach_ticks = float(breach_count.sum()) / (n * ticks)
    print(f"    {total_c} region crossings over the run; {after_b} landed on the tick after a breach.")
    print(
        f"    Crossings after a breach: {after_b/max(total_c,1)*100:.2f}%. "
        f"Breaching cell-ticks overall: {breach_ticks*100:.2f}%."
    )
    print(
        "    A crossing rate on breaching ticks no higher than the base rate means "
        "reconciliation is not what moved the cell."
    )

    # -- ADR-0005's precondition -------------------------------------------
    print("\n  ADR-0005's PRECONDITION: measured dwell against each cell's own tau")
    final = checkpoint_rows[-1]
    print(
        f"    At the horizon, window dwell median {final['window_dwell']['median']:.2f} ticks, "
        f"tau median {final['tau']['median']:.3f}."
    )
    print(
        f"    Cells with dwell >= {DEFAULT_SAFETY_FACTOR} x tau: "
        f"{final['cells_dwell_ge_safety_tau']}/{n}. "
        f"Cells with dwell <= 2 ticks: {final['cells_dwell_le_2']}/{n}."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dome": args.dome, "split": args.split, "seed": args.seed, "ticks": ticks,
        "cells": n, "elapsed_minutes": elapsed / 60,
        "burn_in": {
            "graph_last_breach": graph_last,
            "tail_window": tail,
            "tail_ticks_with_breach": tail_breaches,
            "holds_from": None if graph_last >= ticks else graph_last + 1,
            "per_cell": per_cell,
            "decades": decades,
        },
        "attribution": {
            "crossings": total_c,
            "crossings_after_breach": after_b,
            "breaching_cell_tick_fraction": breach_ticks,
        },
        "checkpoints": checkpoint_rows,
    }
    (args.out / "202-read.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        args.out / "202-per-tick.npz",
        ratio_max=ratio_max.astype(np.float32),
        breaching=breaching,
        last_breach=last_breach,
        breach_count=breach_count,
        levels=levels,
        cell_ids=np.array([cell.id for cell in cells]),
        crossings_total=crossings_total,
        crossings_after_breach=crossings_after_breach,
        final_margin=read.margin.numpy(),
        final_offset=read.offset.numpy(),
        final_tau=regional_tau(agent),
        final_cumulative_dwell=read.dwell.numpy(),
    )
    print(f"\n  Written to {args.out}/202-read.json and 202-per-tick.npz")


if __name__ == "__main__":
    main()
