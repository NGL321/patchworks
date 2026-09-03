"""Where does the residue sit -- on the open weight, or on a masked-closed one?"""
import copy, torch, sys
sys.path.insert(0, "/app/tests")
from conftest import SMALL
from patchworks.graph import build_graph, CellKind
from patchworks.tick import Sheaf
from patchworks.learning import SparsityAnneal, TransportRule

NUDGE = 0.25
dome = build_graph(SMALL)
s = Sheaf(dome, generator=torch.Generator().manual_seed(0))
g = torch.Generator().manual_seed(7)
with torch.no_grad():
    s.stalks[: s.layout.total] = torch.randn(s.layout.total, generator=g)
    s.charts.normal_(0.0, 1.0, generator=g)
    for edge in dome.edges:
        for side in (0, 1):
            s.broadcast[2*edge.id+side, :edge.m].normal_(0.0, 1.0, generator=g)
for _ in range(3):
    s.tick()

D = next(c.id for c in s.dome.cells if c.kind is CellKind.DRIVE)
eps = sorted((s.maps.owner == D).nonzero().flatten().tolist())
base, pert = copy.deepcopy(s), copy.deepcopy(s)
g2 = torch.Generator().manual_seed(11)
with torch.no_grad():
    draw = torch.empty(len(eps), *s.maps.maps.shape[1:]).normal_(0.0, NUDGE, generator=g2)
    pert.maps.maps[eps] += draw * pert.maps.support[eps]

def upd(sh):
    r = TransportRule(sh, anneal=SparsityAnneal(horizon=1)); r.steps = 1
    return r.gradient()

before, after = upd(base), upd(pert)
for e in eps:
    sup = s.maps.support[e]
    d = (after[e] - before[e])
    idx = (d != 0).nonzero()
    print(f"ep{e}: support open at {sup.nonzero().tolist()}   map shape {tuple(sup.shape)}")
    print(f"   before nonzero at {(before[e]!=0).nonzero().tolist()}")
    print(f"   after  nonzero at {(after[e]!=0).nonzero().tolist()}")
    for i in idx.tolist():
        r, c = i
        print(f"   delta at [{r},{c}] = {d[r, c].item():.6e}   support={int(sup[r, c])}"
              f"   before={before[e][r,c].item():.6e} after={after[e][r,c].item():.6e}")
