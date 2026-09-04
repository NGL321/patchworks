"""Mask attainability, pre-registered by #411 §8 and read on #415.

For each structural mask on the built dome: does it contain a scaled
co-isometry, and how far from flat is the projection's fixed point?
"""

from collections import Counter

import torch

from patchworks.graph import DEFAULT_SPEC, build_graph
from patchworks.restriction import RestrictionMaps, pair_index

dome = build_graph(DEFAULT_SPEC)

rows = []
for edge in dome.edges:
    for side, cell_id in enumerate((edge.u, edge.v)):
        k = int(dome.restriction_mask(edge.id, cell_id).sum())
        cell = dome.cells[cell_id]
        rows.append(
            dict(
                pair=pair_index(edge.id, side),
                edge=edge.id,
                m=edge.m,
                k=k,
                cell=cell_id,
                kind=cell.kind,
                boundary=cell.is_boundary,
            )
        )

short = [r for r in rows if r["k"] < r["m"]]
print(f"endpoints: {len(rows)}   boundary-held: {sum(r['boundary'] for r in rows)}")
print(f"(m, k) census: {Counter((r['m'], r['k']) for r in rows).most_common()}")
print(f"\nmasks with k < m -- no scaled co-isometry exists: {len(short)}")
for r in short:
    floor = (r["k"] / r["m"]) ** 0.5
    print(
        f"  pair {r['pair']:5d} edge {r['edge']:4d} cell {r['cell']:4d} "
        f"{r['kind']:>12} m={r['m']} k={r['k']}  "
        f"attainable sigma_min/flat = {floor:.4f}  "
        f"(rank deficit {r['m'] - r['k']} of {r['m']})"
    )

# The projection's fixed point, read at the draw. Flat is measured over all m
# singular values of the m x k active block -- the m - k structural zeros
# included, because those are exactly what an unattainable mask leaves dead.
g = torch.Generator().manual_seed(42)
maps = RestrictionMaps(dome, generator=g)


def spectrum(pair: int, m: int, k: int) -> torch.Tensor:
    block = maps.maps[pair, :m, :k].detach().double()
    s = torch.linalg.svdvals(block)
    return torch.cat([s, s.new_zeros(m - s.numel())])


before, after, norms = [], [], []
for r in rows:
    m, k = r["m"], r["k"]
    s = spectrum(r["pair"], m, k)
    flat = s.square().sum().sqrt() / m**0.5
    before.append(float((s - flat).abs().max() / flat))

    block = maps.maps[r["pair"], :m, :k].detach().double()
    u, _, vh = torch.linalg.svd(block, full_matrices=False)
    projected = flat * (u @ vh)
    sp = torch.linalg.svdvals(projected)
    sp = torch.cat([sp, sp.new_zeros(m - sp.numel())])
    flat_p = sp.square().sum().sqrt() / m**0.5
    after.append(float((sp - flat_p).abs().max() / flat_p))
    norms.append(float((projected.norm() - block.norm()).abs() / block.norm()))

attainable = [a for a, r in zip(after, rows, strict=True) if r["k"] >= r["m"]]
deficient = [a for a, r in zip(after, rows, strict=True) if r["k"] < r["m"]]

print(
    f"\nat the draw:   max departure from flat {max(before):.4f}, "
    f"median {sorted(before)[len(before) // 2]:.4f}"
)
print(
    f"after one projection, attainable masks ({len(attainable)}): "
    f"max departure {max(attainable):.2e}"
)
print(
    f"after one projection, deficient masks ({len(deficient)}): "
    f"max departure {max(deficient):.4f}, min {min(deficient):.4f}"
)
ok = [n for n, r in zip(norms, rows, strict=True) if r["k"] >= r["m"]]
bad = [n for n, r in zip(norms, rows, strict=True) if r["k"] < r["m"]]
print(f"\nnorm move, attainable masks: max {max(ok):.2e}")
print(f"norm move, deficient masks:  max {max(bad):.4f}, min {min(bad):.4f}")
print(f"all 9 deficient endpoints boundary-held: {all(r['boundary'] for r in short)}")
print(
    "deficient endpoints among the banded (unpinned) maps: "
    f"{sum(1 for r in rows if r['k'] < r['m'] and not r['boundary'])} "
    f"of {sum(1 for r in rows if not r['boundary'])}"
)
