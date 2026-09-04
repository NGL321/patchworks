"""Read the two arms against each other, checkpoint by checkpoint.

[#132](https://github.com/NGL321/patchworks/issues/132) §2 objects that a per-cell
`K` spectrum read through today's near-rank-1 channel is *"a read of the collapse,
not of the domain"*. That objection is **testable rather than assumable**: run the
same read on the post-floor surface and on `main`'s, at matched ticks, and see
whether the statistics move.

Also prints the same statistics from #166's own recorded run, which was taken before
[ADR-0031](../../docs/adr/0031-the-sparsity-pressure-is-deleted.md) deleted the
sparsity pressure from the transport rule, so the three columns separate two
different questions: what the **floor** changes (post vs pre), and what **`main` has
already changed** since #166 published its headline (pre vs #166).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
LEGACY = HERE.parent / "chart-double-duty-166"

FIELDS = [
    ("stable_rank", lambda c: c["stable_rank"]["median"]),
    ("effective_rank", lambda c: c["effective_rank"]["median"]),
    ("rho_K", lambda c: c["memory"]["rho_K"]["median"]),
    ("summed_tau", lambda c: c["memory"]["summed_tau_per_cell"]["median"]),
    ("nonnormality", lambda c: c["nonnormality"]["median"]),
]


def ladder(path: Path) -> dict:
    record = json.loads(path.read_text())
    return {c["ticks"]: c for c in record["checkpoints"]}


def main() -> None:
    arms = {}
    for arm in ("postfloor", "prefloor"):
        for path in sorted(HERE.glob(f"132-{arm}-real-train-seed*-100000.json")):
            seed = int(path.stem.split("seed")[1].split("-")[0])
            arms[(arm, seed)] = ladder(path)
    legacy = {}
    for path in sorted(LEGACY.glob("166-real-train-seed*-30000.json")):
        seed = int(path.stem.split("seed")[1].split("-")[0])
        legacy[seed] = ladder(path)

    if not arms:
        print("no 100k runs yet")
        return

    ticks = sorted({t for lad in arms.values() for t in lad})
    for name, get in FIELDS:
        print(f"\n{name}  (median over cells)")
        print(f"{'ticks':>8s} {'postfloor':>22s} {'prefloor':>22s} {'#166 (pre-0031)':>18s}")
        for t in ticks:
            def span(arm):
                vals = [get(lad[t]) for (a, _), lad in arms.items() if a == arm and t in lad]
                if not vals:
                    return "-"
                return (
                    f"{min(vals):.4f}"
                    if len(vals) == 1
                    else f"{min(vals):.4f}-{max(vals):.4f}"
                )

            old = [get(lad[t]) for lad in legacy.values() if t in lad]
            oldtxt = (
                f"{min(old):.4f}-{max(old):.4f}" if len(old) > 1 else
                (f"{old[0]:.4f}" if old else "-")
            )
            print(f"{t:>8d} {span('postfloor'):>22s} {span('prefloor'):>22s} {oldtxt:>18s}")


if __name__ == "__main__":
    main()
