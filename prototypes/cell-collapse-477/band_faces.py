"""Separate ADR-0015's two band faces at the apex: does the projection rescale
K *down* (upper face, #335's mechanism) or *up* (lower face, its opposite)?

`CellOperators.project` clamps norms into [1/rho_k, 1] at BOTH ends, and the
mask it returns (`target != norms`) does not say which face fired.
`benchmarks/projection_firing.py` therefore reports a firing rate that conflates
"retention was cut" with "retention was restored".
"""
import sys, pathlib, collections
import numpy as np
import torch

root = pathlib.Path.cwd()
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "tools"))
sys.path.insert(0, str(root / "benchmarks"))
sys.path.insert(0, str(root / "tests"))

import projection_firing as pf
from patchworks.agent import run
from patchworks.learning import PredictionRule, TransportRule
from patchworks.graph import DEFAULT_SPEC, build_graph

TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

env, agent = pf.build(DEFAULT_SPEC, "train", SEED)
ops = agent.sheaf.operators
rho_k = float(ops.rho_k)
print("band = [%.4f, 1.0]   ticks=%d  seed=%d" % (1.0 / rho_k, TICKS, SEED))

up_hits, down_hits = [], []
original = ops.project


def watched():
    norms = ops.norms.detach().clone()
    down = (norms > 1.0)
    up = (norms < 1.0 / rho_k)
    down_hits.append(down.clone())
    up_hits.append(up.clone())
    return original()


ops.project = watched
bias = PredictionRule(agent.sheaf)
transport = TransportRule(agent.sheaf)
try:
    for _o in run(agent, TICKS, seed=SEED):
        bias.step()
        if agent.sheaf.ticks > 1:
            transport.step()
finally:
    del ops.project
    env.close()

skip = pf.burn_in(DEFAULT_SPEC)
down = torch.stack(down_hits[skip:]).to(torch.float64).mean(dim=0).numpy()
up = torch.stack(up_hits[skip:]).to(torch.float64).mean(dim=0).numpy()

dome = agent.dome
pred = list(dome.predicting)
lvl = [int(dome.cells[c].index.level) for c in pred]
col = [str(dome.cells[c].index.column) for c in pred]

print()
print("=== firing rate by level, split by band face ===")
print("  level  cells   DOWN (upper face, shortens)   UP (lower face, lengthens)")
for L in sorted(set(lvl)):
    m = [i for i in range(len(pred)) if lvl[i] == L]
    print("   L%d     %3d          %8.4f                    %8.4f"
          % (L, len(m), float(np.median(down[m])), float(np.median(up[m]))))

print()
print("=== the two collapsing classes ===")
groups = {
    'vision L1': [i for i in range(len(pred)) if col[i] == 'vision' and lvl[i] == 1],
    'soma L1': [i for i in range(len(pred)) if col[i] == 'somatomotor' and lvl[i] == 1],
    'apex L7': [i for i in range(len(pred)) if lvl[i] == 7],
}
for g, m in groups.items():
    print("  %-10s DOWN %8.4f    UP %8.4f" % (g, float(np.median(down[m])), float(np.median(up[m]))))

radii = ops.radii().detach().numpy()
print()
print("=== rho(K) at the end, by group ===")
for g, m in groups.items():
    print("  %-10s median rho %.4f" % (g, float(np.median(radii[m]))))
