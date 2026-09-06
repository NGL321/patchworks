import sys, json, collections, os
sys.path.insert(0, 'src')
import numpy as np
from patchworks.graph import DEFAULT_SPEC, build_graph, CellKind

dome = build_graph(DEFAULT_SPEC)
pred = list(dome.predicting)
n = len(pred)
print("predicting cells:", n)

print("pred ids contiguous:", list(pred) == list(range(min(pred), max(pred) + 1)))
lvl = [int(dome.cells[c].index.level) for c in pred]
deg = [int(dome.degrees[c]) for c in pred]
drive_cells = [cid for cid, c in enumerate(dome.cells) if c.kind == CellKind.DRIVE]
boundary = set(dome.boundary)
print("drive cells:", drive_cells, "boundary:", len(boundary))

nbr = {c: set(dome.neighbours(c)) for c in range(len(dome.cells))}
drive_adj = [1 if any(x in nbr[c] for x in drive_cells) else 0 for c in pred]
bnd_adj = [sum(1 for x in nbr[c] if x in boundary) for c in pred]

dist = {}
dq = collections.deque()
for b in boundary:
    dist[b] = 0
    dq.append(b)
while dq:
    u = dq.popleft()
    for v in nbr[u]:
        if v not in dist:
            dist[v] = dist[u] + 1
            dq.append(v)
d_rim = [dist[c] for c in pred]

try:
    pv = [int(dome.private_dimensions[i]) for i in range(n)]
except Exception as e:
    print("pv fail:", e)
    pv = [0] * n

print("levels present:", sorted(collections.Counter(lvl).items()))
print("d_rim distribution:", sorted(collections.Counter(d_rim).items()))
print("p_v distribution:", sorted(collections.Counter(pv).items()))
print("drive-adjacent predicting cells:", [i for i, x in enumerate(drive_adj) if x])

data = {}
for s in (42, 43, 44):
    p = 'prototypes/chart-per-domain-132/132-postfloor-real-train-seed%d-100000.json' % s
    data[s] = json.load(open(p))

ticks = [ck['ticks'] for ck in data[42]['checkpoints']]
print()
print("=== dead cells (modes_retaining == 0) per checkpoint ===")
print("ticks:      ", ticks)
dead_hist = {}
for s in (42, 43, 44):
    rows = []
    for ck in data[s]['checkpoints']:
        mr = np.array(ck['per_cell']['modes_retaining'])
        rows.append(set(np.where(mr == 0)[0].tolist()))
    dead_hist[s] = rows
    print("seed %d:" % s, [len(r) for r in rows])

print()
print("=== trajectory ===")
for s in (42, 43, 44):
    rows = dead_hist[s]
    ever = set().union(*rows)
    final = rows[-1]
    print("seed %d: ever-dead %d, dead at 100k %d, recovered-by-100k %s"
          % (s, len(ever), len(final), sorted(ever - final)))
    nonmono = 0
    for c in sorted(ever):
        seq = [1 if c in r else 0 for r in rows]
        first = seq.index(1)
        if 0 in seq[first:]:
            nonmono += 1
    print("        died-then-revived at least once: %d / %d" % (nonmono, len(ever)))

print()
print("=== structural identity of the dead set at 100k ===")
for s in (42, 43, 44):
    final = sorted(dead_hist[s][-1])
    print("seed %d dead idx: %s" % (s, final))
    print("   levels   :", [lvl[i] for i in final])
    print("   d_rim    :", [d_rim[i] for i in final])
    print("   degree   :", [deg[i] for i in final])
    print("   bnd_adj  :", [bnd_adj[i] for i in final])
    print("   p_v      :", [pv[i] for i in final])
    print("   driveadj :", [drive_adj[i] for i in final])

print()
print("=== base rates ===")
print("level counts   :", sorted(collections.Counter(lvl).items()))
print("d_rim counts   :", sorted(collections.Counter(d_rim).items()))
print("degree counts  :", sorted(collections.Counter(deg).items()))

allfinal = set().union(*[dead_hist[s][-1] for s in (42, 43, 44)])
inter = set.intersection(*[dead_hist[s][-1] for s in (42, 43, 44)])
print("union of dead across 3 seeds:", len(allfinal), sorted(allfinal))
print("intersection                :", len(inter), sorted(inter))
