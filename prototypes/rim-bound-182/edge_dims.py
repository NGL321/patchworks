"""What `m` is, per level, and how the two arguments of the `max` compare."""

from collections import Counter, defaultdict

import torch

from patchworks.graph import build_graph
from patchworks.tick import Sheaf

dome = build_graph()
sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(0))
rho = sheaf.maps.rho
owner = sheaf.maps.owner
print("rho", rho, "rho^2", rho**2)

per_level_m = defaultdict(Counter)
rows = []
for cell in dome.cells:
    if cell.is_boundary:
        continue
    mine = (owner == cell.id).nonzero().flatten().tolist()
    if not mine:
        continue
    ms = [dome.edges[i // 2].m for i in mine]
    per_level_m[cell.index.level].update(ms)
    rows.append((cell.index.level, len(mine), sum(ms), rho**2 * len(mine), cell.stalk))

for level in sorted(per_level_m):
    here = [r for r in rows if r[0] == level]
    ties = sum(1 for r in here if r[2] == r[3])
    sm = sum(1 for r in here if r[2] > r[3])
    rd = sum(1 for r in here if r[2] < r[3])
    print(
        f"L{level}: {len(here):>3} cells, m histogram {dict(per_level_m[level])}, "
        f"stalk {sorted({r[4] for r in here})}, "
        f"sum_m>rho2deg {sm}, tie {ties}, rho2deg>sum_m {rd}"
    )
