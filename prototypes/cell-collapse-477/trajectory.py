import sys, json
sys.path.insert(0, 'src')
import numpy as np
from patchworks.graph import DEFAULT_SPEC, build_graph

dome = build_graph(DEFAULT_SPEC)
pred = list(dome.predicting)
n = len(pred)
lvl = [int(dome.cells[c].index.level) for c in pred]
col = [str(dome.cells[c].index.column) for c in pred]

data = {s: json.load(open(
    'prototypes/chart-per-domain-132/132-postfloor-real-train-seed%d-100000.json' % s))
    for s in (42, 43, 44)}
ticks = [ck['ticks'] for ck in data[42]['checkpoints']]

groups = {
    'vision L1  (64)': [i for i in range(n) if col[i] == 'vision' and lvl[i] == 1],
    'vision L2  (16)': [i for i in range(n) if col[i] == 'vision' and lvl[i] == 2],
    'soma  L1   (6)': [i for i in range(n) if col[i] == 'somatomotor' and lvl[i] == 1],
    'soma  L2   (4)': [i for i in range(n) if col[i] == 'somatomotor' and lvl[i] == 2],
    'core  L3-6 (52)': [i for i in range(n) if 3 <= lvl[i] <= 6],
    'apex  L7   (8)': [i for i in range(n) if lvl[i] == 7],
}

print("=== median rho(K) trajectory by construction group, mean over seeds 42/43/44 ===")
print("  %-16s" % "group", "  ".join("%7d" % t for t in ticks))
for gname, m in groups.items():
    rows = []
    for s in (42, 43, 44):
        rho = np.array([ck['per_cell']['rho_K'] for ck in data[s]['checkpoints']])
        rows.append(np.median(rho[:, m], axis=1))
    med = np.mean(rows, axis=0)
    print("  %-16s" % gname, "  ".join("%7.4f" % v for v in med))

print()
print("=== the same, as fraction of the t=100 value (the knee) ===")
print("  %-16s" % "group", "  ".join("%7d" % t for t in ticks))
for gname, m in groups.items():
    rows = []
    for s in (42, 43, 44):
        rho = np.array([ck['per_cell']['rho_K'] for ck in data[s]['checkpoints']])
        rows.append(np.median(rho[:, m], axis=1))
    med = np.mean(rows, axis=0)
    print("  %-16s" % gname, "  ".join("%7.3f" % (v / med[0]) for v in med))

print()
print("=== median nonnormality (shape; the band cannot move it) ===")
print("  %-16s" % "group", "  ".join("%7d" % t for t in ticks))
for gname, m in groups.items():
    rows = []
    for s in (42, 43, 44):
        nn = np.array([ck['per_cell']['nonnormality'] for ck in data[s]['checkpoints']])
        rows.append(np.median(nn[:, m], axis=1))
    med = np.mean(rows, axis=0)
    print("  %-16s" % gname, "  ".join("%7.4f" % v for v in med))

print()
print("=== median stable_rank (shape; the band cannot move it) ===")
print("  %-16s" % "group", "  ".join("%7d" % t for t in ticks))
for gname, m in groups.items():
    rows = []
    for s in (42, 43, 44):
        sr = np.array([ck['per_cell']['stable_rank'] for ck in data[s]['checkpoints']])
        rows.append(np.median(sr[:, m], axis=1))
    med = np.mean(rows, axis=0)
    print("  %-16s" % gname, "  ".join("%7.3f" % v for v in med))
