"""Split each drive endpoint's gradient into disagreement and sparsity parts."""
import copy, torch, sys
sys.path.insert(0, "/app/tests")
from conftest import SMALL
from patchworks.graph import build_graph, CellKind
from patchworks.tick import Sheaf
from patchworks.learning import (
    SparsityAnneal, TransportRule, TransportPath, relative_disagreement,
    normalised_l1, MAPS_PARAMETER,
)
from torch.func import functional_call, grad

NUDGE = 0.25

def running():
    dome = build_graph(SMALL)
    s = Sheaf(dome, generator=torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        s.stalks[: s.layout.total] = torch.randn(s.layout.total, generator=g)
        s.charts.normal_(0.0, 1.0, generator=g)
        for edge in dome.edges:
            for side in (0, 1):
                s.broadcast[2 * edge.id + side, : edge.m].normal_(0.0, 1.0, generator=g)
    for _ in range(3):
        s.tick()
    return s

def owned(s, cell):
    return sorted((s.maps.owner == cell).nonzero().flatten().tolist())

def perturb(s, cell, seed=11):
    eps = owned(s, cell)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        draw = torch.empty(len(eps), *s.maps.maps.shape[1:]).normal_(0.0, NUDGE, generator=g)
        s.maps.maps[eps] += draw * s.maps.support[eps]

def split(s, eps, label):
    r = TransportRule(s, anneal=SparsityAnneal(horizon=1)); r.steps = 1
    gathered, neighbour = r.inputs()
    path = TransportPath(s.maps)
    P = r.permitted
    pressure = r.pressure

    def dis_only(mp):
        out = functional_call(path, mp, (gathered,))
        return relative_disagreement(out, neighbour).sum()

    def pen_only(mp):
        return (pressure * normalised_l1(mp[MAPS_PARAMETER], P)).sum()

    gd = grad(dis_only)({MAPS_PARAMETER: s.maps.maps})[MAPS_PARAMETER]
    gp = grad(pen_only)({MAPS_PARAMETER: s.maps.maps})[MAPS_PARAMETER]
    print(f"  [{label}] pressure={pressure:.4f}")
    for e in eps:
        F = s.maps.maps[e]
        p = P[e].item()
        l1 = F.abs().sum().item(); l2 = F.norm().item()
        h = l1 / ((p ** 0.5) * l2) if l2 > 0 else float("nan")
        print(f"   ep{e}: p={p:.0f} h={h:.8f} |gdis|={gd[e].norm():.3e} "
              f"|gpen|={gp[e].norm():.3e} |F|1={l1:.4e} |F|F={l2:.4e}")

s = running()
D = next(c.id for c in s.dome.cells if c.kind is CellKind.DRIVE)
eps = owned(s, D)
base, pert = copy.deepcopy(s), copy.deepcopy(s)
perturb(pert, D)
split(base, eps, "before")
split(pert, eps, "after")
