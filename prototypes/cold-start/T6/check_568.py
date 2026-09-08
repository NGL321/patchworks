"""T6 (#568 / B19): the checks the state-vs-emitted instrument has to pass.

Cheap, and run before the arms are, because the whole reading turns on the emitted
stream really being `F_c h_c` for the same `h_c` the state stream records.

1. **The gather is the engine's own.** For every predicting cell and every incident
   edge, the row `Emission.gather` produces equals `maps[pair_index(e, side)] @ h_c`
   computed by hand off `sheaf.stalks`, and equals `sheaf.broadcast` when the sheaf
   is asked for a broadcast off the *same* stalks.
2. **The pad is inert.** A cell of below-maximum degree has exact zeros in its
   padded columns, so masking-by-multiplying leaves the participation ratio alone.
3. **The interior mask lines up.** Zeroing the non-interior ends leaves exactly the
   interior edges' widths' worth of non-zero columns.
4. **The rank ceiling holds.** Emitted participation ratio never exceeds the number
   of lane dimensions the cell has (`Σ_e m_e`), and state never exceeds `n`.
5. **The readable projector is a projector** inside the exposed block, and the
   readable stream's rank never exceeds its rank.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/check_568.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0 = _HERE.parent / "T0"
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
b19 = _load("b19", _HERE / "b19_state_emitted.py")
arms_mod = b19.arms_mod

from excitation import blocks  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402


def main() -> None:
    env, agent = arms_mod.build_arm("reserve", 42)
    try:
        # A few ticks so nothing is at its zero initialisation.
        for _ in t0.run_ticks(agent, 40, seed=42):
            pass
        dome, sheaf = agent.dome, agent.sheaf
        emission = b19.Emission(dome)
        rec = b19.Recorder(agent, emission)
        for _ in t0.run_ticks(agent, 12, seed=7):
            rec.observe()
        h, y = rec.streams()
        y3 = torch.stack(rec.y, dim=0).transpose(0, 1)  # [cells, T, deg_max, m_max]

        maps = sheaf.maps.maps.detach()
        n = dome.shape.n

        # 1. The gather is the engine's own, on the last recorded frame.
        with torch.no_grad():
            gathered = emission.gather(sheaf)
            engine = sheaf.maps.restrict(sheaf.stalks[sheaf.layout.pair_positions])
        scale = float(engine.abs().max())
        worst = 0.0
        for row, cid in enumerate(emission.cells):
            stalk = sheaf.stalks[sheaf.layout.slice(cid)]
            padded_stalk = torch.zeros(sheaf.maps.stalk_width)
            padded_stalk[: stalk.numel()] = stalk
            for d, eid in enumerate(dome.incident[cid]):
                edge = dome.edges[eid]
                side = 0 if edge.u == cid else 1
                p = pair_index(edge.id, side)
                by_hand = maps[p] @ padded_stalk
                worst = max(worst, float((gathered[row, d] - by_hand).abs().max()))
                worst = max(worst, float((gathered[row, d] - engine[p]).abs().max()))
        # float32 bmm: compare against the scale of the values, not against zero.
        assert worst < 1e-5 * max(scale, 1.0), (
            f"gather disagrees with the engine by {worst} at scale {scale}"
        )
        print(
            f"1. gather == maps.restrict == by-hand F_e h_c   max |diff| {worst:.2e} "
            f"at scale {scale:.2e}  OK"
        )

        # 2. The pad is inert.
        bad = 0
        for row, cid in enumerate(emission.cells):
            deg = len(dome.incident[cid])
            if deg < emission.degree_max:
                bad += int((y3[row, :, deg:] != 0).sum())
        assert bad == 0, f"{bad} non-zero padded entries"
        print(f"2. padded ends exactly zero                     {bad} non-zero  OK")

        # 3. The interior mask keeps exactly the interior ends.
        keep = emission.interior
        for row, cid in enumerate(emission.cells):
            want = [
                dome.edges[eid].kind.value == "interior" for eid in dome.incident[cid]
            ]
            got = keep[row, : len(want)].tolist()
            assert got == want, f"cell {cid}: interior mask {got} != {want}"
        print(f"3. interior mask matches edge kinds              {keep.sum()} ends  OK")

        # 4. The rank ceilings hold.
        from excitation import participation_ratio

        pr_state = participation_ratio(h, centred=True)
        pr_emit = participation_ratio(y, centred=True)
        ticks = h.shape[1]
        over_state = int((pr_state > min(n, ticks) + 1e-6).sum())
        over_emit = 0
        for row, cid in enumerate(emission.cells):
            cap = min(sum(dome.edges[e].m for e in dome.incident[cid]), ticks)
            over_emit += int(pr_emit[row] > cap + 1e-6)
        assert over_state == 0 and over_emit == 0, (over_state, over_emit)
        print(
            f"4. rank ceilings: state <= min(n,T)={min(n, ticks)}, "
            f"emitted <= min(sum m_e,T)          OK"
        )

        # 5. The readable projector is a projector, inside the exposed block.
        bl = blocks(agent)
        p = bl.interior_rowspace.double()
        idem = float((p @ p - p).abs().max())
        sym = float((p - p.transpose(1, 2)).abs().max())
        private = bl.private.double()
        leak = float((torch.einsum("cnm,cm->cn", p, private)).abs().max())
        assert idem < 1e-6 and sym < 1e-9 and leak < 1e-6, (idem, sym, leak)
        print(
            f"5. interior_rowspace idempotent {idem:.1e}, symmetric {sym:.1e}, "
            f"no private leak {leak:.1e}  OK"
        )

        readable = torch.einsum("ctn,cnm->ctm", h.double(), p)
        pr_read = participation_ratio(readable, centred=True)
        ranks = torch.linalg.matrix_rank(p).double()
        over = int((pr_read > ranks + 1e-6).sum())
        assert over == 0, over
        print(f"6. readable rank <= rank(projector)              {over} over  OK")
        print("\nall checks pass")
    finally:
        env.close()


if __name__ == "__main__":
    main()
