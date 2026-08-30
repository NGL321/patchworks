"""The wedge's counts, computed rather than asserted (patchworks#163).

Every table in `docs/spec/11-the-language-graph.md` -- cell counts, edge counts, cut
capacities, and the guaranteed private dimension gradient -- comes out of this script,
using `06-graph-topology.md`'s own arithmetic: a cut is the sum of `m_e` over the edges
crossing it, and `dim H^0 >= n - sum_e m_e` per cell.

It is arithmetic over the construction parameters, not a build: the wedge has no builder
yet, and `graph.py` builds only the dome. What it is for is that the spec's numbers agree
with the rule the spec states, and that a later builder has a target to reproduce. The
dome's figures it is checked against come from `patchworks dome`, not from here.
"""

n = 32
m_int, m_bnd, m_drive = 4, 8, 1
B = 128
fan = 4
core = [16, 14, 12, 10, 8]
core_deg, apex_deg = 6, 4

edges = []


def add(label, nedges, m):
    edges.append((label, nedges, m))


add("L0->L1 (boundary)", 2 * B, m_bnd)
add("L1->L2", 2 * (B // fan), m_int)
add("L1 lateral", 2 * (B // fan - 1), m_int)
add("L2 lateral", 2 * (B // fan // fan - 1), m_int)
add("L2->L3 (merge)", 2 * (B // fan // fan), m_int)
add("core vertical (fan 2)", sum(2 * s for s in core[:-1]), m_int)
add("drive", core[-1], m_drive)

print("edges and cut capacities, numbers per tick")
for label, ne, m in edges:
    print(f"  {label:26s} {ne:4d} edges x m={m}  = {ne * m}")

print()
print("per-cell degree and guaranteed private dimension")


def row(name, deg, sm):
    print(f"  {name:28s} deg {deg:2d}   sum m_e {sm:3d}   dim H0 >= {max(0, n - sm)}")


for lat, tag in ((2, "interior"), (1, "end")):
    row(f"L1 ({tag})", fan + 1 + lat, fan * m_bnd + m_int + lat * m_int)
for lat, tag in ((2, "interior"), (1, "end")):
    row(f"L2 ({tag})", fan + 1 + lat, fan * m_int + m_int + lat * m_int)
row("L3-L6 core", core_deg, core_deg * m_int)
row("L7 apex (with drive edge)", apex_deg + 1, apex_deg * m_int + m_drive)

pred = 2 * (B // fan) + 2 * (B // fan // fan) + sum(core)
bnd = 2 * B + 1
tot_edges = sum(ne for _, ne, _ in edges)
ends = 2 * tot_edges - 2 * B - core[-1]
print()
print(f"predicting {pred}   boundary {bnd}   edges {tot_edges}")
print(f"mean degree over predicting cells: {ends / pred:.2f}")
print(f"L0->L1 cut: {2 * B * m_bnd}   (dome's is 2120 over 265 boundary edges)")
