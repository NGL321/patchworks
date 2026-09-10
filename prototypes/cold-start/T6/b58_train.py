"""B58 (#630) §B: does the staggered frame survive training, or does the rule walk off it?

Every number in [B56 (#628)](https://github.com/NGL321/patchworks/issues/628) §0 is a
**construction** reading, and [B49 (#616)](https://github.com/NGL321/patchworks/issues/616)'s
standing rule says a reading states which object it is on. #630 names the
construction-to-trained gap as its central risk, with B56's own demonstration that
the gap is large: `GAMMA = 0.6075` was read off the stagger-1 row of a construction
sweep and lay **below the trained surface's entire operating range**.

So this asks the one question the construction table cannot: put the staggered frame
on the trained surface and see whether it stays there.

**This stands up no new rig.** It wraps `b56_channel.run_arm` unchanged and patches
only the *initialisation*, so every column, checkpoint, surrogate and null is B56's
and the arms are comparable to B56's own by construction. The patch is
`b42_stagger.build_staggered`'s body applied to an already-built agent.

Arms, each `reserve_p12` seed 42 to 2,000 ticks:

* **`s19_baseline`** -- staggered at 19, the shipped transport rule and nothing else.
  Stagger 19 is the **maximum-differentiation exact point** at `p = 12`
  (`channel_return` 1.0000 at differentiation 0.7321). If the rule walks off the
  frame, this is where it shows, with no term holding it on.
* **`s19_channel`** -- stagger 19 plus clause 1, B56's surviving candidate. Does a
  term that drives `channel_return` hold a frame that already has it at 1.0000?
* **`s0_baseline`** -- stagger 0, which **is** B42's flat bundle and
  [B48 (#615)](https://github.com/NGL321/patchworks/issues/615)'s null:
  `channel_return` 1.0000 at differentiation 0.0000. #630's named risk is that a
  staggered frame relaxes toward this. Running it makes the null a measured
  trajectory rather than a construction point --
  [B50 (#618)](https://github.com/NGL321/patchworks/issues/618)'s
  *the null is not static*, which B56 carried.

B56's own `baseline` and `channel` arms are the random-init controls and are not
re-run; their files are already on this branch.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b58_train.py run --arms s19_baseline
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b56 = _load("b58t_b56", _HERE / "b56_channel.py")

from patchworks.restriction import pair_index  # noqa: E402
import construction_grading as cg  # noqa: E402


#: (stagger, B56 arm name, dome arm). The B56 arm decides the *term*; the stagger
#: decides the *initialisation*; the dome arm decides `p`. Nothing else differs from
#: B56's own run.
#:
#: `p8_s22_baseline` is the trained price at the `p` **both** curves on #630 point to
#: -- this ticket's differentiation ceiling (0.8036, its highest anywhere) and
#: [B44 (#608)](https://github.com/NGL321/patchworks/issues/608)'s graded community
#: band, which B22 found at `p = 8` alone. Stagger 22 is that arm's ceiling stagger.
ARMS = {
    "s19_baseline": (19, "baseline", "reserve_p12"),
    "s19_channel": (19, "channel", "reserve_p12"),
    "s0_baseline": (0, "baseline", "reserve_p12"),
    "p8_s22_baseline": (22, "baseline", "reserve_p8"),
}


def apply_stagger(agent, seed: int, stagger: int) -> None:
    """`b42_stagger.build_staggered`'s body, applied to an agent already built.

    Verbatim in the arithmetic that matters -- one orthonormal frame per cell inside
    the permitted window, the `i`-th incident edge taking rows `(i·stagger + j) mod
    k_v` -- so the construction sweep's staggers and these are the same object. The
    generator is offset identically (`seed + 9001`), so `s19_baseline` at tick 0 reads
    B56 §0's stagger-19 row.
    """
    dome = agent.dome
    maps = agent.sheaf.maps
    n = dome.shape.n
    gen = torch.Generator().manual_seed(seed + 9001)
    frames, widths = {}, {}
    for cell in dome.predicting:
        width = int(dome._permitted[cell])
        a = torch.randn(width, width, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        full = torch.zeros(n, n, dtype=torch.float64)
        full[:width, :width] = q
        frames[cell] = full
        widths[cell] = width
    with torch.no_grad():
        for cell in dome.predicting:
            width = widths[cell]
            for slot, edge_id in enumerate(dome.incident[cell]):
                edge = dome.edges[edge_id]
                try:
                    idx = pair_index(edge_id, cg.side_of(dome, edge_id, cell))
                except Exception:
                    continue
                rows = [(slot * stagger + j) % width for j in range(edge.m)]
                block = frames[cell][rows]
                t = maps.maps[idx]
                rr = min(block.shape[0], t.shape[0])
                cc = min(block.shape[1], t.shape[1])
                t[:rr, :cc] = block[:rr, :cc].to(t.dtype)


def run_one(name: str, seed: int, ticks: int) -> None:
    """One arm, through B56's `run_arm`, with only the initialisation patched."""
    stagger, base_arm, dome_arm = ARMS[name]
    out = _HERE / f"630-stagger-{name}-seed{seed}-{ticks}.json"
    if out.exists():
        print(f"[B58] {out.name} already at the horizon, skipping", flush=True)
        return

    original = b56.b33.arms_mod.build_arm

    def staggered_build_arm(arm, s, *args, **kwargs):
        env, agent = original(arm, s, *args, **kwargs)
        apply_stagger(agent, s, stagger)
        return env, agent

    started = time.time()
    b56.b33.arms_mod.build_arm = staggered_build_arm
    prior_arm = b56.ARM
    b56.ARM = dome_arm
    try:
        record = b56.run_arm(base_arm, seed, ticks, out)
    finally:
        b56.b33.arms_mod.build_arm = original
        b56.ARM = prior_arm
    record["issue"] = 630
    record["reading"] = "does the staggered frame survive training"
    record["stagger"] = stagger
    record["b56_arm"] = base_arm
    record["dome_arm"] = dome_arm
    record["arm"] = name
    out.write_text(__import__("json").dumps(record, indent=1), encoding="utf-8")
    print(
        f"[B58] {name} (stagger {stagger}, term {base_arm}) done in "
        f"{(time.time() - started) / 60:.1f} min -> {out.name}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=b56.TICKS)
    args = p.parse_args()
    # Sequential, never parallel: long runs get killed when parallelised, and the
    # record is written as each checkpoint lands (#555's note, and this map's).
    for name in args.arms:
        run_one(name, args.seed, args.ticks)


if __name__ == "__main__":
    main()
