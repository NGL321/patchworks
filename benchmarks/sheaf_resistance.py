"""The sheaf Laplacian's effective resistance, rim to apex, along the channel (#237).

[#150](https://github.com/NGL321/patchworks/issues/150) computed the **graph**
Laplacian's effective resistance and said plainly that it had not computed the
**sheaf** Laplacian's: *"the sheaf resistance can be arbitrarily worse than the
graph resistance in specific directions, and which directions those are is a real
question this document does not answer."*
[#230](https://github.com/NGL321/patchworks/issues/230) closed the structural
remedy family on #150's graph-side 1.82 unit-resistance edges, and named this the
one condition under which that closure reverses. This is that computation::

    python benchmarks/sheaf_resistance.py control     # ~30 s, no run
    python benchmarks/sheaf_resistance.py read        # untrained surface
    python benchmarks/sheaf_resistance.py read --learn 30000

**The object.** The sheaf coboundary `δ: C⁰ → C¹` sends a node-stalk assignment
to its disagreements, `(δx)_e = F_{e,u} x_u − F_{e,v} x_v`, and the sheaf
Laplacian is `L = δᵀδ` (Hansen & Ghrist, arXiv:1808.01513). Effective resistance
between a **direction** `a` in cell `u`'s stalk and a direction `b` in cell `v`'s
is `R = χᵀ L⁺ χ` for `χ = a@u − b@v` — the sheaf's generalisation of the graph's
`(e_u − e_v)ᵀ L⁺ (e_u − e_v)`, and equal to it exactly when the sheaf is trivial.
#150's own words are why it needs *a pair of stalk directions, not a pair of
cells*, before it has a referent.

**Read as a current, which is what makes it computable at this size.** `C⁰` is
17,104 dimensions here, so `L⁺` is not formed. Since `L⁺ = δ⁺(δ⁺)ᵀ`,

    R  =  min { ‖y‖² : δᵀ y = χ }

— the energy of the least-norm unit current that supplies `χ`. One sparse
least-squares solve per pair returns it, and :func:`control` checks that this
route reproduces #150's published graph-side table to machine precision on the
trivial sheaf.

**The one thing the graph case does not have: `χ` may not be suppliable at all.**
The graph is connected, so `e_u − e_v` is orthogonal to `ker L` and `R` is always
finite. The sheaf's kernel is `H⁰`, and here `dim C¹ = 3,764` against
`dim C⁰ = 17,104`, so `dim H⁰ ≥ 13,340` **by construction** — that is what
private features are. A `χ` with a component in `H⁰` has no finite-energy current
and its resistance is `+∞`. So each pair is reported as two numbers, not one:

- `leak` — the fraction of `‖χ‖²` no current can supply, `‖χ − δᵀy‖²/‖χ‖²`. The
  infinite part, and a statement about the **rank** of the maps.
- `R` — the resistance of the part that can be, `‖y‖²`, in the same units as
  #150's table so the two are directly comparable.

The split matters for what turns on this ticket. #230's closed family is
**topological** — widen the funnel, parallel the L2→L3 cut, add relay cells or a
virtual node — and `R` is the term those remedies move. `leak` is not a
topological quantity, and no rewiring in that family touches it.

**The directions are the channel's, per [ADR-0022](../docs/adr/0022-a-hop-is-an-operator-norm-along-a-learned-channel.md).**
An isotropic probe here would repeat the error that cost this map a 1e14 phantom
deficit, and the correction is a standing ADR rather than a preference. The
channel between `u` and `v` is read off the composed chain operator along the
path, exactly the composition :mod:`alignment_read` uses per hop::

    C  =  F_{v,e_k}ᵀ · Π_i [ F_{c_i,e_{i+1}} F_{c_i,e_i}ᵀ ] · F_{u,e_1}

`a` is its top right singular vector, in `u`'s stalk; `b` its top left, in `v`'s,
signed so that `C a = σ₁ b` with `σ₁ > 0` — the direction the chain actually
delivers, which is the analogue of the graph's `e_u − e_v`. The reconciliation
gains are omitted from `C` deliberately: they are positive scalars per cell, so
they scale the composition and cannot move a singular vector.

Two baselines run beside it, per pair and never pooled:

- **isotropic** — `a`, `b` drawn uniformly on their stalk spheres. What the probe
  ADR-0022 rejects would have reported.
- **public** — drawn uniformly inside the span of the incident maps' row spaces,
  which is the largest subspace that is not private by construction. It separates
  what the structural mask costs from what the *learned* rank-deficiency costs.

Nothing here decides anything on its own. It returns a measurement against
#150's per-kind table and #230's ruling.
"""

import argparse
import dataclasses
import sys
import time
import types
from collections import defaultdict, deque

import numpy as np
import scipy.sparse as sp
import torch

from patchworks.graph import CellKind, Dome, build_graph
from patchworks.restriction import RestrictionMaps, pair_index

sys.path.insert(0, "benchmarks")

#: Eigenvalues of `δδᵀ` below this share of the largest are the factorisation's
#: own noise rather than directions the graph carries, and are dropped from the
#: pseudo-inverse. `δ` is rank-deficient here by construction, so the cut is not
#: optional and where it falls is what separates `H⁰` from arithmetic.
EIGEN_FLOOR = 1e-12

#: Below this share of `‖χ‖²`, a residual is the solver's floor rather than a
#: kernel component. Set from the control run, where the true leak is zero and
#: the measured one lands ~1e-26.
LEAK_FLOOR = 1e-12


# -- the complex -------------------------------------------------------------


def offsets(sizes) -> tuple[list[int], int]:
    out, total = [], 0
    for n in sizes:
        out.append(total)
        total += n
    return out, total


def edge_gram_inverse(delta) -> np.ndarray:
    """`(δδᵀ)⁺`, by eigendecomposition with :data:`EIGEN_FLOOR` as the cut.

    Factored out of :attr:`Complex.solver` so a test can drive the same
    arithmetic on a graph whose resistances are known in closed form.
    """
    gram = (delta @ delta.T).toarray()
    values, vectors = np.linalg.eigh(gram)
    keep = values > values.max() * EIGEN_FLOOR
    inverse = np.zeros_like(values)
    inverse[keep] = 1.0 / values[keep]
    return (vectors * inverse) @ vectors.T


class Complex:
    """`δ` for one surface, plus where each cell's and edge's block lives."""

    def __init__(self, dome: Dome, maps: RestrictionMaps) -> None:
        self.dome = dome
        self.cell_at, self.c0 = offsets(c.stalk for c in dome.cells)
        self.edge_at, self.c1 = offsets(e.m for e in dome.edges)
        blocks = maps.maps.detach().double().numpy()
        self.blocks = {}
        rows, cols, vals = [], [], []
        for e in dome.edges:
            for side, cid in enumerate((e.u, e.v)):
                F = blocks[pair_index(e.id, side)][: e.m, : dome.cells[cid].stalk]
                self.blocks[(e.id, cid)] = F
                r, c = np.nonzero(F)
                rows.append(self.edge_at[e.id] + r)
                cols.append(self.cell_at[cid] + c)
                vals.append((1.0 if side == 0 else -1.0) * F[r, c])
        self.delta = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.c1, self.c0),
        )
        self.deltaT = self.delta.T.tocsc()
        self._solver: np.ndarray | None = None
        self._spectrum: tuple[np.ndarray, np.ndarray] | None = None

    def chi(self, u: int, a: np.ndarray, v: int, b: np.ndarray) -> np.ndarray:
        x = np.zeros(self.c0)
        x[self.cell_at[u] : self.cell_at[u] + a.size] = a
        x[self.cell_at[v] : self.cell_at[v] + b.size] = -b
        return x

    def regularised(self, chi: np.ndarray, epsilon: float) -> float:
        """`χᵀ(L + εI)⁻¹χ` — the resistance with the kernel damped rather than cut.

        The plain resistance cannot compare two graphs whose *suppliable* demands
        differ, because each is then answering a different question: delete an
        edge and `χ̃` is a smaller and differently-directed vector, so `R` can
        fall while transmission gets worse. This form has no such hole. It is
        finite for every `χ`, it tends to `R` from below as `ε → 0` when the leak
        is zero, and it is **monotone in the PSD order**: deleting edges only
        lowers `L`, which can only raise `χᵀ(L + εI)⁻¹χ`. So a ratio between two
        surfaces is a real comparison and is guaranteed to be at least 1.

        Woodbury is what makes it cheap: `L = δᵀδ` is 17,104-square, but

            (δᵀδ + εI)⁻¹ = (1/ε)[ I − δᵀ(εI + δδᵀ)⁻¹ δ ]

        and `δδᵀ` is 3,764-square, whose eigendecomposition is already taken once
        for :attr:`solver`.
        """
        if self._spectrum is None:
            gram = (self.delta @ self.deltaT).toarray()
            self._spectrum = np.linalg.eigh(gram)
        values, vectors = self._spectrum
        w = vectors.T @ (self.delta @ chi)
        return float(chi @ chi - (w**2 / (values + epsilon)).sum()) / epsilon

    @property
    def scale(self) -> float:
        """`λ_max(δδᵀ)`, so an `ε` can be quoted relative to the surface."""
        if self._spectrum is None:
            self.regularised(np.zeros(self.c0), 1.0)
        return float(self._spectrum[0].max())

    @property
    def solver(self) -> np.ndarray:
        """`(δδᵀ)⁺`, factored once and reused by every pair.

        The least-norm current is `y* = (δᵀ)⁺χ`, and `(δᵀ)⁺ = (δδᵀ)⁺δ`. `C¹` is
        3,764 dimensions against `C⁰`'s 17,104, so the Gram matrix on the *edge*
        side is small enough to pseudo-invert densely once — after which every
        resistance is two matrix-vector products rather than an iterative solve.
        On a trained surface the iterative route is slow enough to matter; this
        one is exact and the same object, and
        :meth:`tests.test_sheaf_resistance` checks both against the closed forms.
        """
        if self._solver is None:
            self._solver = edge_gram_inverse(self.delta)
        return self._solver

    def resistance(self, chi: np.ndarray) -> tuple[float, float]:
        """`(R, leak)`: the least-norm current's energy, and what it cannot supply."""
        y = self.solver @ (self.delta @ chi)
        residual = self.deltaT @ y - chi
        return float(y @ y), float(residual @ residual) / float(chi @ chi)

    def public_basis(self, cell_id: int) -> np.ndarray:
        """An orthonormal basis of `span_e row(F_{e,v})` — the not-private subspace."""
        stack = np.concatenate(
            [self.blocks[(e, cell_id)] for e in self.dome.incident[cell_id]], axis=0
        )
        u, s, vt = np.linalg.svd(stack, full_matrices=False)
        keep = s > s.max() * 1e-10 if s.size and s.max() > 0 else np.zeros(0, bool)
        return vt[keep]


# -- paths and the channel ---------------------------------------------------


def apex_cells(dome: Dome) -> list[int]:
    """The deepest predicting level. On the real dome that is level 7, which is
    the literal `graph_transmission.resistance_section` reads; taken as the
    deepest rather than as 7 so a test dome has an apex too."""
    predicting = [c for c in dome.cells if c.kind is CellKind.PREDICTING]
    deepest = max(c.index.level for c in predicting)
    return [c.id for c in predicting if c.index.level == deepest]


def stratum(dome: Dome, cell_id: int) -> str:
    cell = dome.cells[cell_id]
    if cell.kind is CellKind.PREDICTING:
        return f"L{cell.index.level}"
    return cell.kind.value


def shortest_path(dome: Dome, source: int, targets: set[int]) -> list[int] | None:
    """Cell ids from `source` to the nearest of `targets`, `None` if unreached."""
    prev = {source: -1}
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        if cell in targets and cell != source:
            path = [cell]
            while prev[path[-1]] != -1:
                path.append(prev[path[-1]])
            return path[::-1]
        for e in dome.incident[cell]:
            nxt = dome.edges[e].other(cell)
            if nxt not in prev:
                prev[nxt] = cell
                queue.append(nxt)
    return None


def edges_of(dome: Dome, path: list[int]) -> list[int]:
    out = []
    for a, b in zip(path, path[1:]):
        out.append(next(e for e in dome.incident[a] if dome.edges[e].other(a) == b))
    return out


def chain(complex_: Complex, path: list[int]) -> np.ndarray:
    """`C`, the composed chain operator from `path[0]`'s stalk to `path[-1]`'s.

    The per-hop factor is :mod:`alignment_read`'s `M(e_in → v → e_out) =
    F_{v,e_out} · gain_v · F_{v,e_in}ᵀ` with the gain dropped: it is a positive
    scalar and cannot move a singular vector, and the directions are all this
    function is read for.
    """
    ids = edges_of(complex_.dome, path)
    C = complex_.blocks[(ids[0], path[0])]
    for cell, e_in, e_out in zip(path[1:-1], ids, ids[1:]):
        C = complex_.blocks[(e_out, cell)] @ complex_.blocks[(e_in, cell)].T @ C
        C = renormalise(C)
    return renormalise(complex_.blocks[(ids[-1], path[-1])].T @ C)


def renormalise(C: np.ndarray) -> np.ndarray:
    """`C / ‖C‖`, applied after every hop, and it is not cosmetic.

    On a trained surface the maps are effective rank ~1.001 and consecutive ones
    are close to mutually orthogonal in their dominant directions, so the raw
    seven-hop product lands at `σ₁ ≈ 4e-17` — at float64's noise floor against
    intermediates of order 1, which makes its singular vectors numerically
    meaningless and the phrase *along the channel* empty. Scaling cannot move a
    singular vector, so normalising after each hop returns the same directions
    with the intermediates held at order 1. It also keeps this function honest
    about what it is for: `chain` yields the channel's **direction** and never
    its gain, which is measured rather than chained (#214, and #142's cost).
    """
    norm = np.linalg.norm(C)
    return C / norm if norm > 0 else C


def channel(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """`(a, b, σ₁)`: the chain's top right and left singular vectors, `C a = σ₁ b`."""
    u, s, vt = np.linalg.svd(C, full_matrices=False)
    return vt[0], u[:, 0], float(s[0])


def unit(rng, n: int) -> np.ndarray:
    x = rng.standard_normal(n)
    return x / np.linalg.norm(x)


# -- the sections ------------------------------------------------------------


def control(dome: Dome) -> None:
    """The instrument against #150's published graph-side table.

    The trivial sheaf — one dimension per cell, every restriction map `[1]` — has
    `L` equal to the graph Laplacian, so `min ‖y‖²` must return the numbers
    `benchmarks/graph_transmission.py` prints and `docs/research/150` publishes.
    Nothing below is believable if this section is not exact.
    """
    import graph_transmission as gt

    print("\n## control: the trivial sheaf is the graph\n")
    apex = apex_cells(dome)
    patch = [c.id for c in dome.cells if c.kind is CellKind.PATCH]
    dense = gt.effective_resistance(gt.weighted_adjacency(dome, lambda e: 1.0))
    print(
        f"  #150's patch->apex block, dense pseudoinverse: "
        f"mean {dense[np.ix_(patch, apex)].mean():.4f} "
        f"max {dense[np.ix_(patch, apex)].max():.4f}   "
        f"(published: mean 1.8163 max 2.0524)"
    )

    # The same `Complex` the read runs, on the trivial sheaf: one dimension per
    # cell, every restriction map `[1.0]`. `L = δᵀδ` is then the graph Laplacian
    # and the two must agree, which is what makes the sheaf numbers readable in
    # #150's units at all.
    trivial = dataclasses.replace(
        dome,
        cells=tuple(dataclasses.replace(c, stalk=1) for c in dome.cells),
        edges=tuple(dataclasses.replace(e, m=1) for e in dome.edges),
    )
    blocks = torch.zeros((2 * len(dome.edges), 1, 1), dtype=torch.float64)
    blocks[:, 0, 0] = 1.0
    cx = Complex(trivial, types.SimpleNamespace(maps=blocks))
    one = np.ones(1)

    worst_err, worst_leak = 0.0, 0.0
    for u in patch[:8]:
        for v in apex:
            R, leak = cx.resistance(cx.chi(u, one, v, one))
            worst_err = max(worst_err, abs(R - dense[u, v]))
            worst_leak = max(worst_leak, leak)
    print(f"  least-norm current vs pseudoinverse, 64 pairs: max error {worst_err:.2e}")
    print(f"  leak on a connected graph (must be 0): max {worst_leak:.2e}")
    print(f"  the leak floor this sets: {LEAK_FLOOR:.0e}")


def surface_section(cx: Complex, maps: RestrictionMaps) -> None:
    """What the maps are, before what they cost: the gauge, the rank, the cut.

    The whole difference between an untrained reading and a trained one is here,
    so it is printed beside them rather than left to be inferred. The rank line
    discharges `docs/research/053`'s standing note — *nothing has ever measured
    whether today's maps have drifted toward rank-1* — and the gauge line is what
    says the answer is not a scale collapse. The `EIGEN_FLOOR` sweep is the
    reading's own robustness: the trained `δδᵀ` is ill-conditioned enough that
    where the pseudo-inverse cuts has to be checked rather than assumed.
    """
    print("\n### the surface these resistances are of\n")
    norms = maps.norms().detach().numpy()
    pinned = maps.pinned.numpy()
    print(
        f"  Frobenius norm  pinned {norms[pinned].min():.4f}-{norms[pinned].max():.4f} "
        f"(gauge: exactly 1)   banded {norms[~pinned].min():.4f}-"
        f"{norms[~pinned].max():.4f} (gauge: [{1 / maps.rho:.2f}, {maps.rho:.2f}])"
    )
    ranks = []
    for F in cx.blocks.values():
        s = np.linalg.svd(F, compute_uv=False) ** 2
        ranks.append(float(s.sum() / s.max()) if s.max() > 0 else 0.0)
    ranks = np.array(ranks)
    print(
        f"  effective rank of the maps (participation ratio): "
        f"median {np.median(ranks):.4f}  min {ranks.min():.4f}  max {ranks.max():.4f}"
    )
    values = cx._spectrum[0] if cx._spectrum else np.linalg.eigvalsh(
        (cx.delta @ cx.deltaT).toarray()
    )
    positive = values[values > 0]
    print(
        f"  spectrum of delta.deltaT: max {values.max():.4e}  "
        f"min>0 {positive.min():.4e}  "
        f"condition {values.max() / positive.min():.3e}"
    )
    kept = [(f, int((values > values.max() * f).sum())) for f in
            (1e-8, 1e-10, 1e-12, 1e-14)]
    print(
        "  directions kept at EIGEN_FLOOR "
        + ", ".join(f"{f:.0e}: {k}/{len(values)}" for f, k in kept)
    )


def read(dome: Dome, maps: RestrictionMaps, label: str, trials: int, seed: int) -> None:
    print(f"\n## rim to apex on the {label} surface\n")
    started = time.time()
    cx = Complex(dome, maps)
    print(
        f"  dim C0 {cx.c0}, dim C1 {cx.c1}; "
        f"dim H0 >= {cx.c0 - cx.c1} by construction, "
        f"{(cx.c0 - cx.c1) / cx.c0:.1%} of the node stalks"
    )
    surface_section(cx, maps)
    apex = set(apex_cells(dome))
    rng = np.random.default_rng(seed)

    sources: list[int] = []
    by_kind: dict[str, list[int]] = defaultdict(list)
    for c in dome.cells:
        if c.id in apex or c.kind is CellKind.PREDICTING:
            continue
        by_kind[c.kind.value].append(c.id)
    for kind, ids in sorted(by_kind.items()):
        take = ids if len(ids) <= trials else list(rng.choice(ids, trials, replace=False))
        sources.extend(int(i) for i in take)

    graph_R = None
    try:
        import graph_transmission as gt

        graph_R = gt.effective_resistance(gt.weighted_adjacency(dome, lambda e: 1.0))
    except Exception as exc:  # pragma: no cover - reporting only
        print(f"  (graph-side comparison unavailable: {exc})")

    rows = []
    for n, u in enumerate(sources, 1):
        path = shortest_path(dome, u, apex)
        if path is None:
            continue
        v = path[-1]
        C = chain(cx, path)
        a, b, _ = channel(C)
        probes = {"channel": (a, b)}
        pu, pv = cx.public_basis(u), cx.public_basis(v)
        probes["public"] = (
            pu.T @ unit(rng, pu.shape[0]),
            pv.T @ unit(rng, pv.shape[0]),
        )
        probes["isotropic"] = (
            unit(rng, dome.cells[u].stalk),
            unit(rng, dome.cells[v].stalk),
        )
        row = dict(
            source=u,
            target=v,
            kind=stratum(dome, u),
            hops=len(path) - 1,
            graph=float(graph_R[u, v]) if graph_R is not None else float("nan"),
        )
        for name, (pa, pb) in probes.items():
            R, leak = cx.resistance(cx.chi(u, pa, v, pb))
            row[name] = R
            row[f"{name}_leak"] = leak
        # The rim cell's own edge, along *that* edge's channel. #150's finding is
        # that ~1.0 of the graph's 1.82 is this leaf and no rewiring removes it,
        # so the climb is only worth reading in units of it.
        leaf = path[:2]
        la, lb, _ = channel(chain(cx, leaf))
        row["leaf"], row["leaf_leak"] = cx.resistance(cx.chi(u, la, leaf[1], lb))
        row["leaf_graph"] = (
            float(graph_R[u, leaf[1]]) if graph_R is not None else float("nan")
        )
        rows.append(row)
        print(f"  {n}/{len(sources)} {row['kind']:>15} -> apex  "
              f"{len(path) - 1} hops  R_channel {row['channel']:10.3f}  "
              f"leak {row['channel_leak']:.3f}", flush=True)

    report(rows, time.time() - started)
    routes_section(cx, rows, graph_R)
    edge_section(cx, graph_R)


def spanning_tree(dome: Dome, roots: list[int]) -> set[int]:
    """Edge ids of a BFS tree from `roots`. It contains every shortest path."""
    seen, keep = set(roots), set()
    queue = deque(roots)
    while queue:
        cell = queue.popleft()
        for e in dome.incident[cell]:
            nxt = dome.edges[e].other(cell)
            if nxt not in seen:
                seen.add(nxt)
                keep.add(e)
                queue.append(nxt)
    return keep


def routes_section(cx: Complex, rows: list[dict], graph_R) -> None:
    """What the dome's parallel routes are still worth, on *this* surface.

    #150's closure rests on a topological claim: rim-to-apex is seven hops but
    only 1.82 unit-resistance edges, *"because each level is a lattice with many
    parallel routes into the next"*, so the depth is already bought back and
    rewiring has nothing left to buy. That claim is graph-side. Whether it holds
    of the sheaf is not an argument — it is the difference between the dome and a
    spanning tree of the dome, which keeps every shortest path and deletes every
    alternative route, measured on the same maps and along the same directions.

    Graph-side the deletion must hurt: the climb becomes a single path. If it
    hurts the sheaf too, the routes are still carrying and the topological lever
    is real. If it costs nothing, the routes were already worth nothing, and
    adding more of them is not a remedy — which is the scope question #230's
    closed family turns on.
    """
    dome = cx.dome
    keep = spanning_tree(dome, apex_cells(dome))
    print(f"\n### what the parallel routes are worth ({len(keep)} of "
          f"{len(dome.edges)} edges span the dome)\n")
    print(
        "  The same maps and the same channel directions, with every edge outside\n"
        "  a BFS tree deleted. Graph-side this must hurt, because the climb loses\n"
        "  its alternatives. What it does sheaf-side is the reading.\n"
    )
    mask = np.zeros(cx.c1, dtype=bool)
    for e in keep:
        mask[cx.edge_at[e] : cx.edge_at[e] + dome.edges[e].m] = True
    tree = Complex.__new__(Complex)
    tree.dome, tree.blocks = dome, cx.blocks
    tree.cell_at, tree.c0, tree.edge_at, tree.c1 = (
        cx.cell_at, cx.c0, cx.edge_at, cx.c1
    )
    tree.delta = cx.delta.multiply(mask[:, None]).tocsr()
    tree.deltaT = tree.delta.T.tocsc()
    tree._solver = None
    tree._spectrum = None

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
    # The plain `R` cannot carry this comparison: the tree's suppliable demand is
    # a smaller and differently-directed vector, so `R` can *fall* when an edge
    # is deleted. `regularised` has no such hole — it is monotone in the PSD
    # order, so `tree / dome` is a real ratio and is bounded below by 1.
    epsilons = [e * cx.scale for e in (1e-6, 1e-9, 1e-12)]
    tree_graph = None
    if graph_R is not None:
        import graph_transmission as gt

        sub = dataclasses.replace(
            dome, edges=tuple(e for e in dome.edges if e.id in keep)
        )
        tree_graph = gt.effective_resistance(gt.weighted_adjacency(sub, lambda e: 1.0))
    header = "  ".join(f"eps={e / cx.scale:.0e}" for e in epsilons)
    print(
        f"  {'from':>15} {'n':>4} | routes buy, tree/dome:  {header} | "
        f"{'graph dome':>10} {'graph tree':>10} {'buy':>7}"
    )
    for kind in sorted(groups, key=lambda s: (s[0] != "L", s)):
        block = groups[kind]
        ratios: list[list[float]] = [[] for _ in epsilons]
        g_dome, g_tree = [], []
        for r in block:
            path = shortest_path(dome, r["source"], {r["target"]})
            a, b, _ = channel(chain(cx, path))
            chi = cx.chi(r["source"], a, r["target"], b)
            for i, eps in enumerate(epsilons):
                ratios[i].append(tree.regularised(chi, eps) / cx.regularised(chi, eps))
            g_dome.append(r["graph"])
            if tree_graph is not None:
                g_tree.append(float(tree_graph[r["source"], r["target"]]))
        line = f"  {kind:>15} {len(block):>4} |                         " + "  ".join(
            f"{np.median(v):8.2f}x" for v in ratios
        )
        if g_tree:
            line += (
                f" | {np.median(g_dome):10.3f} {np.median(g_tree):10.3f} "
                f"{np.median(np.array(g_tree) / np.array(g_dome)):6.2f}x"
            )
        print(line)


def edge_section(cx: Complex, graph_R) -> None:
    """Every edge's own resistance, along that edge's own channel.

    The ticket asks for this per edge and never as a graph-wide average, and it
    is the sheaf's answer to #150's cut-share table `w_e R(u,v)` — which is 1
    exactly for a bridge and low where parallel routes exist. Here the two
    endpoints' maps set the direction, so a resistive edge is one whose maps
    disagree about which direction they carry, not one the topology isolates.
    """
    dome = cx.dome
    print("\n### per edge, along that edge's own channel\n")
    rows = []
    for e in dome.edges:
        C = cx.blocks[(e.id, e.v)].T @ cx.blocks[(e.id, e.u)]
        a, b, sigma = channel(C)
        R, leak = cx.resistance(cx.chi(e.u, a, e.v, b))
        rows.append(
            dict(
                edge=e.id,
                kind=f"{e.kind.value} (m={e.m})",
                R=R / (1 - leak) if leak < 1 else float("inf"),
                leak=leak,
                sigma=sigma,
                graph=float(graph_R[e.u, e.v]) * e.m if graph_R is not None else np.nan,
            )
        )
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    print(
        f"  {'edges':>18} {'count':>6} | {'R median':>9} {'min':>9} {'max':>9} | "
        f"{'leak median':>12} | {'graph w_e R':>12}"
    )
    for kind in sorted(by_kind):
        block = by_kind[kind]
        R = np.array([r["R"] for r in block])
        print(
            f"  {kind:>18} {len(block):>6} | {np.median(R):9.3f} {R.min():9.3f} "
            f"{R.max():9.3f} | {np.median([r['leak'] for r in block]):12.4f} | "
            f"{np.median([r['graph'] for r in block]):12.4f}"
        )
    print("\n  the eight most resistive edges, which is where a channel would die")
    for r in sorted(rows, key=lambda r: -r["R"])[:8]:
        e = dome.edges[r["edge"]]
        print(
            f"    {str(dome.cells[e.u].index):>18} -> {str(dome.cells[e.v].index):<18} "
            f"{r['kind']:<16} R {r['R']:9.3f}  leak {r['leak']:.4f}  "
            f"sigma1 {r['sigma']:.4f}"
        )


def spread(values: np.ndarray) -> str:
    return (
        f"{np.median(values):10.3f} {values.mean():10.3f} "
        f"{values.min():10.3f} {values.max():10.3f}"
    )


def report(rows: list[dict], wall: float) -> None:
    if not rows:
        print("  nothing measured")
        return
    print(f"\n### per stratum, against #150's graph-side R  ({wall:.0f} s)\n")
    print(
        "  `R` is the least-norm current's energy for the suppliable part of the\n"
        "  demand; `R/(1-leak)` rescales it to a full unit of demand and is the\n"
        "  column comparable with `graph R`, where the leak is zero.\n"
    )
    print(
        f"  {'from':>15} {'n':>4} {'hops':>5} {'graph R':>9} | "
        f"{'probe':>10} {'R median':>10} {'mean':>10} {'min':>10} {'max':>10} | "
        f"{'R/(1-leak)':>11} {'leak':>7} {'inf':>7}"
    )
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
    for kind in sorted(groups, key=lambda s: (s[0] != "L", s)):
        block = groups[kind]
        graph = np.array([r["graph"] for r in block])
        hops = np.array([r["hops"] for r in block])
        first = True
        for probe in ("channel", "public", "isotropic"):
            R = np.array([r[probe] for r in block])
            leak = np.array([r[f"{probe}_leak"] for r in block])
            head = (
                f"  {kind:>15} {len(block):>4} {hops.mean():5.1f} {graph.mean():9.4f} | "
                if first
                else f"  {'':>15} {'':>4} {'':>5} {'':>9} | "
            )
            print(
                head
                + f"{probe:>10} "
                + spread(R)
                + f" | {np.median(R / (1 - leak)):11.3f} {np.median(leak):7.4f} "
                f"{int((leak > LEAK_FLOOR).sum()):>3}/{len(block):<3}"
            )
            first = False
    print("\n### the ratio the ticket asks for: sheaf R / graph R, per pair\n")
    print(
        "  Read on `R/(1-leak)`. `arbitrarily worse` is what #150 left open; a\n"
        "  bounded constant here is what leaves #230's structural closure standing.\n"
    )
    print(f"  {'from':>15} {'probe':>10} {'median':>10} {'min':>10} {'max':>10}")
    for kind in sorted(groups, key=lambda s: (s[0] != "L", s)):
        block = groups[kind]
        for probe in ("channel", "public", "isotropic"):
            ratio = np.array(
                [r[probe] / (1 - r[f"{probe}_leak"]) / r["graph"] for r in block]
            )
            print(
                f"  {kind:>15} {probe:>10} {np.median(ratio):10.2f} "
                f"{ratio.min():10.2f} {ratio.max():10.2f}"
            )

    print("\n### the climb, in the rim cell's own leaf edge\n")
    print(
        "  #150: of the graph's 1.82, ~1.0 is the patch's own leaf edge, which no\n"
        "  rewiring short of moving the sensor removes. The same split, sheaf-side\n"
        "  and along each edge's own channel.\n"
    )
    print(
        f"  {'from':>15} {'n':>4} | {'leaf R':>9} {'leaf leak':>10} | "
        f"{'rim->apex R':>12} | {'in leaf edges':>14} {'graph-side':>11}"
    )
    for kind in sorted(groups, key=lambda s: (s[0] != "L", s)):
        block = groups[kind]
        leaf = np.array([r["leaf"] / (1 - r["leaf_leak"]) for r in block])
        full = np.array([r["channel"] / (1 - r["channel_leak"]) for r in block])
        gleaf = np.array([r["leaf_graph"] for r in block])
        ggraph = np.array([r["graph"] for r in block])
        print(
            f"  {kind:>15} {len(block):>4} | {np.median(leaf):9.3f} "
            f"{np.median([r['leaf_leak'] for r in block]):10.4f} | "
            f"{np.median(full):12.3f} | {np.median(full / leaf):14.2f} "
            f"{np.median(ggraph / gleaf):11.2f}"
        )

    channel_leak = np.array([r["channel_leak"] for r in rows])
    print(
        f"\n  channel-direction leak over all {len(rows)} pairs: "
        f"median {np.median(channel_leak):.4f}, "
        f"{int((channel_leak > LEAK_FLOOR).sum())} of {len(rows)} pairs carry one"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=("control", "read"))
    parser.add_argument("--learn", type=int, default=0, help="ticks of both rules")
    parser.add_argument("--trials", type=int, default=16, help="rim cells per stratum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="dome")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--save",
        default=None,
        help="write the trained maps here, so a re-read costs no training run",
    )
    parser.add_argument("--load", default=None, help="read maps written by --save")
    args = parser.parse_args(argv)

    if args.section == "control":
        control(build_graph())
        return

    if args.load:
        dome = build_graph()
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(args.seed))
        loaded = torch.load(args.load, weights_only=True)
        with torch.no_grad():
            maps.maps.copy_(loaded["maps"])
        read(dome, maps, loaded["label"], args.trials, args.seed)
    elif args.learn:
        import untrained_fixed_point as ufp

        env, agent = ufp.build(args.name, args.split, args.seed)
        print(f"training {args.learn} ticks with both rules...", flush=True)
        ufp.taught(agent, args.learn, args.seed)
        label = f"taught {args.learn}-tick (seed {args.seed})"
        if args.save:
            torch.save(
                {"maps": agent.sheaf.maps.maps.detach().clone(), "label": label},
                args.save,
            )
            print(f"saved the trained maps to {args.save}", flush=True)
        read(agent.dome, agent.sheaf.maps, label, args.trials, args.seed)
    else:
        dome = build_graph()
        maps = RestrictionMaps(
            dome, generator=torch.Generator().manual_seed(args.seed)
        )
        read(dome, maps, f"untrained (seed {args.seed})", args.trials, args.seed)


if __name__ == "__main__":
    main()
