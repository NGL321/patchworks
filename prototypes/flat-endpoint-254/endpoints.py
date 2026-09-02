"""The real fixture, the real perturbation: before/after at each drive endpoint."""
import copy, torch, sys
sys.path.insert(0, "/app/tests")
from conftest import SMALL
from patchworks.graph import build_graph, CellKind
from patchworks.tick import Sheaf
from patchworks.learning import (
    SparsityAnneal, TransportRule, TransportPath, relative_disagreement, MAPS_PARAMETER,
)
from torch.func import functional_call

NUDGE = 0.25

def running():
    dome = build_graph(SMALL)
    sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        sheaf.stalks[: sheaf.layout.total] = torch.randn(sheaf.layout.total, generator=g)
        sheaf.charts.normal_(0.0, 1.0, generator=g)
        for edge in dome.edges:
            for side in (0, 1):
                sheaf.broadcast[2 * edge.id + side, : edge.m].normal_(0.0, 1.0, generator=g)
    for _ in range(3):
        sheaf.tick()
    return sheaf

def owned_endpoints(sheaf, cell):
    return frozenset((sheaf.maps.owner == cell).nonzero().flatten().tolist())

def perturb(sheaf, cell, seed=11):
    endpoints = sorted(owned_endpoints(sheaf, cell))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        draw = torch.empty(len(endpoints), *sheaf.maps.maps.shape[1:]).normal_(0.0, NUDGE, generator=g)
        sheaf.maps.maps[endpoints] += draw * sheaf.maps.support[endpoints]

def map_update(sheaf):
    r = TransportRule(sheaf, anneal=SparsityAnneal(horizon=1)); r.steps = 1
    return r.gradient()

def report(sheaf, label):
    r = TransportRule(sheaf, anneal=SparsityAnneal(horizon=1)); r.steps = 1
    gathered, neighbour = r.inputs()
    out = functional_call(TransportPath(sheaf.maps), {MAPS_PARAMETER: sheaf.maps.maps}, (gathered,))
    ratio = relative_disagreement(out, neighbour)
    return out, neighbour, ratio

sheaf = running()
DRIVE = next(c.id for c in sheaf.dome.cells if c.kind is CellKind.DRIVE)
owned = sorted(owned_endpoints(sheaf, DRIVE))
baseline, perturbed = copy.deepcopy(sheaf), copy.deepcopy(sheaf)
perturb(perturbed, DRIVE)
before, after = map_update(baseline), map_update(perturbed)
ob, nb, rb = report(baseline, "b")
oa, na, ra = report(perturbed, "a")

print("drive", DRIVE, "endpoints", owned, " edge widths m:",
      [sheaf.dome.edges[e // 2].m for e in owned])
for e in owned:
    moved = not torch.equal(before[e], after[e])
    print(f" endpoint {e}: moved={moved}")
    print(f"   before ratio={rb[e]:.6f} <Fx,y>={float((ob[e].flatten()*nb[e].flatten()).sum()):+.4e}"
          f" ||Fx||={ob[e].norm():.4e} grad0={bool((before[e]==0).all())}")
    print(f"   after  ratio={ra[e]:.6f} <Fx,y>={float((oa[e].flatten()*na[e].flatten()).sum()):+.4e}"
          f" ||Fx||={oa[e].norm():.4e} grad0={bool((after[e]==0).all())}")
unmoved = frozenset(owned) - frozenset(e for e in range(before.shape[0]) if not torch.equal(before[e], after[e]))
print("UNMOVED:", sorted(unmoved))
