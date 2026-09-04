"""#132's actual comparison: the box-counting dimension of a *piece*, both domains.

[#132](https://github.com/NGL321/patchworks/issues/132) asks whether `k_piece` is a
per-domain number, and says *"ADR-0004's criterion is the piece's box-counting
dimension, and nothing in the record estimates it for language."* The ticket's §5
then reports that the language term cannot be taken, because nothing builds a
language graph.

**§5 is right about the operator half and wrong about this half.** ADR-0004 defines
`k` as *the dimension of the piece* -- a property of the **set of situations a cell
must chart**, which is a fact about the world, not about the graph that charts it.
Its self-intersection criterion is explicitly *read forwards*: *"known before
anything runs -- an embedding is generic once the coordinate count exceeds twice the
piece's box-counting dimension -- which makes self-intersection predictable at
construction."* A quantity knowable before anything runs does not need a builder.

So this rig takes the reading ADR-0004 specifies, symmetrically, in both domains,
with no graph in either:

* **Dome.** The piece of an **L1 vision cell** -- a predicting cell, per #132's own
  withdrawal of #126's sensory-boundary stalk. `vision_sides = (8, 4)` over a 16x16
  patch grid, so one L1 cell owns a 2x2 block of patch cells: an **8x8x3 block of
  the 64x64 render**, 192 raw numbers. Sampled by **configuration sweep**, which is
  ADR-0004's own procedure (*"hold the world still, sweep configurations"*) and is
  the right sampler here for a second reason: #120 measured the arm dead from tick
  5000, so a driven trajectory would report the dimension of the agent's paralysis
  rather than of the piece.

* **Language.** The piece of an **L1 wedge cell** -- `11-the-language-graph.md`,
  `fan = 4`, so one L1 cell owns **4 consecutive buffer slots**. Heard is a 97-way
  one-hot per slot: 388 raw numbers. Sampled by sliding a 4-slot window along a
  character stream.

Both are then read with Grassberger-Procaccia, and the answer is reported **as a
scaling curve, never as a single number**, because a correlation dimension quoted
without its scaling region is the error itself.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

import numpy as np

_BENCH = str(Path(__file__).resolve().parents[2] / "benchmarks")
if _BENCH not in sys.path:
    sys.path.append(_BENCH)

#: The wedge's fan: how many buffer slots one L1 language cell owns. `11`.
LANG_FAN = 4

#: Printable ASCII plus idle and turn-boundary. `11`'s heard stalk.
HEARD_STALK = 97

#: One L1 vision cell owns a 2x2 block of 4x4-px patch cells.
DOME_BLOCK_PX = 8


# -- the estimator ---------------------------------------------------------


def correlation_curve(points, radii, pairs: int = 400_000, seed: int = 0):
    """`C(r)`: the fraction of point pairs closer than each `r`.

    Sampled pairs rather than the full `O(N^2)` matrix, which is what makes the
    two domains runnable at the same sample size -- the estimator must be
    identical across the comparison or the comparison is of estimators.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    i = rng.integers(0, n, pairs)
    j = rng.integers(0, n, pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d = np.linalg.norm(points[i] - points[j], axis=1)
    return np.array([(d < r).mean() for r in radii]), d


def dimension_report(points, label: str, seed: int = 0) -> dict:
    """The scaling curve plus the best-fit slope over its straightest decade."""
    points = np.asarray(points, dtype=np.float64)
    # Scale-free by construction: radii are set from the sample's own median
    # distance, so the two domains' different raw units cannot enter the
    # comparison.
    _, sample = correlation_curve(points, np.array([1.0]), seed=seed)
    positive = sample[sample > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    radii = np.geomspace(1e-3, 4.0, 40) * median
    _, dists = correlation_curve(points, np.array([1.0]), seed=seed + 1)

    # **The atom at zero is removed before the slope is fitted, and reported
    # separately.** Coincident pairs -- two sweep configurations the cell's
    # aperture cannot tell apart -- put a step of their own mass at `r = 0`, and
    # `C(r)` can then never fall below it. A fit confined to small `C` would
    # land on the flat shelf under that step and report dimension ~0 for any
    # piece, however rich. The coincidence rate is itself a reading (it is
    # ADR-0004's self-intersection, measured in the piece rather than in a
    # stalk), so it is kept as `zero_distance_fraction` and excluded from the
    # power law, which is a statement about *distinct* points.
    zero_fraction = float((dists == 0).mean())
    nonzero = dists[dists > 0]
    curve = np.array([(nonzero < r).mean() for r in radii])

    logr, logc = np.log(radii), np.log(np.clip(curve, 1e-9, None))
    slopes = np.gradient(logc, logr)

    # **The scaling region is bounded at both ends, and the upper bound is the
    # one that matters here.** `C(r) -> 1` as `r` covers the whole set, so the
    # curve always ends in a plateau of slope 0; a window chosen for flatness
    # alone lands on that plateau and reports dimension 0 for anything. The
    # genuine power law lives where `C` is small but resolved, so the fit is
    # confined to `C in [1e-3, 0.1]` and the window is then the straightest run
    # inside it. Reported with its own spread so a reader can see whether a
    # slope was ever entitled to be quoted.
    live = (curve > 1e-3) & (curve < 0.1)
    idx = np.flatnonzero(live)
    best = None
    if len(idx) >= 4:
        for width in range(len(idx), 3, -1):
            for start in range(0, len(idx) - width + 1):
                w = idx[start:start + width]
                spread = float(slopes[w].max() - slopes[w].min())
                if best is None or spread < best[0]:
                    best = (spread, w)
            if best is not None and best[0] < 0.5:
                break
    window = best[1] if best is not None else idx
    slope = (
        float(np.polyfit(logr[window], logc[window], 1)[0])
        if len(window) > 1
        else float("nan")
    )

    # **The decisive statistic is upstream of the slope.** A correlation
    # dimension presumes the pair-distance distribution is continuous; where it
    # is supported on a handful of atoms, `C(r)` is a staircase and any slope
    # fitted across a riser is a fit to the riser's width, not to a power law.
    # So the atoms are counted first, at a tolerance well below the smallest
    # spacing a one-hot window can produce, and reported beside the slope.
    scaled = nonzero / median if median else nonzero
    atoms, counts = np.unique(np.round(scaled, 6), return_counts=True)
    heavy = atoms[counts >= max(4, 0.001 * len(scaled))]

    return {
        "label": label,
        "points": int(len(points)),
        "ambient": int(points.shape[1]),
        "median_pair_distance": median,
        "distinct_distances": int(len(atoms)),
        "heavy_atoms": heavy.tolist()[:24],
        "heaviest_atom_mass": (
            float(counts.max() / len(scaled)) if len(scaled) else 1.0
        ),
        # A cell whose aperture never varies is constant, not discrete: it has
        # no distinct pairs at all, and calling that a discrete piece would
        # count the arena floor as a finding.
        "distance_is_discrete": bool(len(scaled) and len(heavy) <= 16),
        "zero_distance_fraction": zero_fraction,
        "radii_over_median": (radii / median).tolist(),
        "correlation": curve.tolist(),
        "local_slope": slopes.tolist(),
        "scaling_window": [int(window[0]), int(window[-1])] if len(window) else None,
        "scaling_spread": float(best[0]) if best is not None else None,
        "correlation_dimension": slope,
    }


# -- the dome's piece ------------------------------------------------------


def dome_sweep(samples: int, seed: int):
    """`[samples, 64, 64, 3]`: the render over a configuration sweep.

    One sweep, shared by every L1 cell, so the cells are read on the *same*
    situations and a difference between them is a difference of aperture rather
    than of sample.
    """
    import mujoco
    from patchworks.sandbox.env import ARM_JOINTS, PlanarPushSandbox

    env = PlanarPushSandbox(split="train", image_size=64)
    env.reset(seed=seed)
    ids = [
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINTS
    ]
    limits = env.model.jnt_range[ids].copy()
    rng = np.random.default_rng(seed)

    frames = np.empty((samples, 64, 64, 3), dtype=np.uint8)
    for s in range(samples):
        env.data.qpos[env._arm_qadr] = rng.uniform(limits[:, 0], limits[:, 1])
        for a in env._puck_qadr:
            env.data.qpos[a:a + 2] = rng.uniform(-0.22, 0.22, 2)
        mujoco.mj_forward(env.model, env.data)
        frames[s] = env._camera_image()
    return frames


def dome_piece(frames, cell_row: int, cell_col: int):
    """`[samples, 192]`: one L1 vision cell's 8x8x3 block of the sweep."""
    r0, c0 = cell_row * DOME_BLOCK_PX, cell_col * DOME_BLOCK_PX
    block = frames[:, r0:r0 + DOME_BLOCK_PX, c0:c0 + DOME_BLOCK_PX, :]
    return block.reshape(len(frames), -1).astype(np.float64) / 255.0


# -- language's piece ------------------------------------------------------


def alphabet_index(text: str):
    """Characters to `11`'s 97-way heard alphabet: printable ASCII, idle, turn."""
    table = {c: i for i, c in enumerate(string.printable[:95])}
    return np.array([table.get(c, 95) for c in text], dtype=np.int64)


def language_piece(samples: int, seed: int, text: str):
    """`[samples, 388]`: an L1 wedge cell's 4-slot heard window over a stream."""
    codes = alphabet_index(text)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(codes) - LANG_FAN, samples)
    out = np.zeros((samples, LANG_FAN * HEARD_STALK), dtype=np.float64)
    for s, st in enumerate(starts):
        for slot in range(LANG_FAN):
            out[s, slot * HEARD_STALK + codes[st + slot]] = 1.0
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text", default=None, help="character stream for the language piece")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.text:
        text = Path(args.text).read_text(encoding="utf-8", errors="replace")
    else:
        docs = sorted((Path(__file__).resolve().parents[2] / "docs").rglob("*.md"))
        text = "\n".join(d.read_text(encoding="utf-8", errors="replace") for d in docs)

    report = {"samples": args.samples, "seed": args.seed, "stream_chars": len(text)}
    print(f"language stream: {len(text)} characters", flush=True)

    lang = language_piece(args.samples, args.seed, text)
    report["language"] = dimension_report(lang, "language L1 (4 heard slots)", args.seed)
    print(
        f"  language  d_corr {report['language']['correlation_dimension']:.3f}"
        f"  ambient {report['language']['ambient']}"
        f"  identical-pair fraction {report['language']['zero_distance_fraction']:.4f}",
        flush=True,
    )

    frames = dome_sweep(args.samples, args.seed)
    # Every L1 vision cell, never a graph-wide average: #127's standing rule,
    # and #181's per-edge-not-per-level form. The cells differ enormously --
    # most of the render is arena floor -- and a mean over them would report
    # the floor.
    cells = []
    for row in range(8):
        for col in range(8):
            piece = dome_piece(frames, row, col)
            entry = dimension_report(piece, f"dome L1 vision ({row},{col})", args.seed)
            entry["cell"] = [row, col]
            entry["variation"] = float(piece.std(axis=0).mean())
            cells.append(entry)
    report["dome_cells"] = cells

    live = [c for c in cells if c["variation"] > 1e-3]
    dims = np.array([c["correlation_dimension"] for c in live], dtype=float)
    dims = dims[np.isfinite(dims)]
    report["dome_summary"] = {
        "cells": len(cells),
        "cells_with_variation": len(live),
        "d_corr_median": float(np.median(dims)) if dims.size else None,
        "d_corr_range": [float(dims.min()), float(dims.max())] if dims.size else None,
        "max_distinct_distances": max(c["distinct_distances"] for c in cells),
        "any_discrete": any(c["distance_is_discrete"] for c in live),
    }
    s = report["dome_summary"]
    print(
        f"  dome      {s['cells_with_variation']}/{s['cells']} cells vary; "
        f"d_corr median {s['d_corr_median']}, range {s['d_corr_range']}; "
        f"max distinct distances {s['max_distinct_distances']}; "
        f"any discrete {s['any_discrete']}",
        flush=True,
    )

    out = args.out or str(Path(__file__).parent / "132-piece-dimension.json")
    Path(out).write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
