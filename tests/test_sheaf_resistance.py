"""`benchmarks/sheaf_resistance.py`: the estimator, and that the script runs (#237).

#237 is a measurement and nothing here pins its verdict — the rim-to-apex
resistances and the leak are readings of a surface later tickets are expected to
move, and a test holding today's numbers would have to be deleted by whoever
moves them. That is `tests/test_graph_transmission.py`'s reasoning, and #150's
before it, and it applies unchanged.

What is worth holding is that the estimator is **right**. The whole ticket is
arithmetic on a 17,104-dimensional object nobody can eyeball, and an off-by-one
in the coboundary's signs or in which block a cell's stalk occupies would
produce a plausible, wrong verdict on whether #230's structural closure
survives. So the estimator is checked against closed forms rather than against
itself:

* **The trivial sheaf is the graph.** With one dimension per cell and every
  restriction map `[1.0]`, `L = δᵀδ` is the graph Laplacian, so the least-norm
  current's energy must equal ordinary effective resistance — checked against
  `graph_transmission.effective_resistance`, which Foster's theorem already
  pins, and against series and parallel in closed form.
* **A connected graph leaks nothing**, because `e_u − e_v ⊥ ker L` exactly.
* **Scaling a sheaf's maps by `α` scales its resistances by `α^-2`**, which fixes
  the convention that makes the sheaf and graph numbers comparable at all.
* **A private direction is infinitely resistive**: a direction in the kernel of
  every incident map leaks all of itself, which is the statement that the
  measurement is *for*.
* **The channel is the chain's top singular pair**, `C a = σ₁ b` with `σ₁ > 0`,
  the sign convention the graph's `e_u − e_v` fixes and the SVD does not.
"""

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import graph_transmission as transmission
import sheaf_resistance as sheaf
from patchworks.graph import build_graph
from patchworks.restriction import RestrictionMaps

from conftest import SMALL


class Trivial:
    """A stand-in for `RestrictionMaps` carrying one prescribed block per pair."""

    def __init__(self, blocks: torch.Tensor) -> None:
        self.maps = blocks


def trivial_maps(dome) -> Trivial:
    """Every map `[1.0]` on a one-dimensional stalk: `L` is the graph Laplacian."""
    blocks = torch.zeros((2 * len(dome.edges), 1, 1), dtype=torch.float64)
    blocks[:, 0, 0] = 1.0
    return Trivial(blocks)


def one_dimensional(dome):
    """`dome` with every stalk and every edge width forced to 1."""
    return dataclasses.replace(
        dome,
        cells=tuple(dataclasses.replace(c, stalk=1) for c in dome.cells),
        edges=tuple(dataclasses.replace(e, m=1) for e in dome.edges),
    )


class TestTrivialSheafIsTheGraph:
    """The estimator's calibration: at `d = 1, F = 1` it must be the graph's."""

    def dome(self):
        return one_dimensional(build_graph(SMALL))

    def test_matches_effective_resistance(self):
        dome = self.dome()
        cx = sheaf.Complex(dome, trivial_maps(dome))
        dense = transmission.effective_resistance(
            transmission.weighted_adjacency(dome, lambda e: 1.0)
        )
        one = np.ones(1)
        for u, v in ((0, len(dome.cells) - 1), (1, 5), (3, len(dome.cells) - 2)):
            R, leak = cx.resistance(cx.chi(u, one, v, one))
            assert R == pytest.approx(dense[u, v], abs=1e-9)
            assert leak == pytest.approx(0.0, abs=sheaf.LEAK_FLOOR)

    def test_series_and_parallel(self):
        """Two unit edges in series are 2; two in parallel are 1/2."""
        for edges, expected in ((((0, 1), (1, 2)), 2.0), (((0, 1), (0, 1)), 0.5)):
            rows, cols, vals = [], [], []
            for i, (u, v) in enumerate(edges):
                rows += [i, i]
                cols += [u, v]
                vals += [1.0, -1.0]
            size = max(max(e) for e in edges) + 1
            delta = sp.csr_matrix((vals, (rows, cols)), shape=(len(edges), size))
            chi = np.zeros(size)
            chi[0], chi[edges[-1][1] if expected == 2.0 else 1] = 1.0, -1.0
            y = sheaf.edge_gram_inverse(delta) @ (delta @ chi)
            assert float(y @ y) == pytest.approx(expected, abs=1e-9)

    def test_a_connected_graph_leaks_nothing(self):
        dome = self.dome()
        cx = sheaf.Complex(dome, trivial_maps(dome))
        one = np.ones(1)
        leaks = [
            cx.resistance(cx.chi(u, one, len(dome.cells) - 1, one))[1]
            for u in range(0, len(dome.cells) - 1, 7)
        ]
        assert max(leaks) < sheaf.LEAK_FLOOR


class TestConvention:
    def test_scaling_the_maps_scales_resistance_inversely_squared(self):
        """`F -> alpha F` sends `L -> alpha^2 L`, so `R -> alpha^-2 R`.

        This is what puts the sheaf's numbers in the graph's units: at
        `alpha = 1` on a trivial sheaf the two coincide, and the exponent is
        what would silently break a comparison against #150's table.
        """
        dome = one_dimensional(build_graph(SMALL))
        one, alpha = np.ones(1), 3.0
        base = sheaf.Complex(dome, trivial_maps(dome))
        scaled_blocks = trivial_maps(dome).maps * alpha
        scaled = sheaf.Complex(dome, Trivial(scaled_blocks))
        u, v = 0, len(dome.cells) - 1
        assert scaled.resistance(scaled.chi(u, one, v, one))[0] == pytest.approx(
            base.resistance(base.chi(u, one, v, one))[0] / alpha**2, rel=1e-7
        )


class TestRegularisedResistance:
    """`χᵀ(L + εI)⁻¹χ`, which is what carries the parallel-routes comparison."""

    def test_it_agrees_with_the_plain_resistance_when_nothing_leaks(self):
        """With no kernel component it must tend to `R` from below as `ε → 0`."""
        dome = one_dimensional(build_graph(SMALL))
        cx = sheaf.Complex(dome, trivial_maps(dome))
        one = np.ones(1)
        u, v = 0, len(dome.cells) - 1
        chi = cx.chi(u, one, v, one)
        exact = cx.resistance(chi)[0]
        previous = 0.0
        for eps in (1e-4, 1e-6, 1e-8, 1e-10):
            got = cx.regularised(chi, eps * cx.scale)
            # `≤ exact` holds in exact arithmetic. The float kernel residual is
            # divided by `ε`, so the bound is stated with room for it.
            assert got <= exact * (1 + 1e-4)
            assert got >= previous
            previous = got
        assert previous == pytest.approx(exact, rel=1e-4)

    def test_deleting_edges_can_only_raise_it(self):
        """Monotone in the PSD order — the property the comparison rests on.

        The plain `R` has no such guarantee once the leak moves, which is
        precisely why the routes section does not use it: deleting an edge
        shrinks the suppliable demand, so `R` can fall while transmission gets
        worse.
        """
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        cx = sheaf.Complex(dome, maps)
        keep = sheaf.spanning_tree(dome, sheaf.apex_cells(dome))
        assert 0 < len(keep) < len(dome.edges)

        mask = np.zeros(cx.c1, dtype=bool)
        for e in keep:
            mask[cx.edge_at[e] : cx.edge_at[e] + dome.edges[e].m] = True
        tree = sheaf.Complex.__new__(sheaf.Complex)
        tree.delta = cx.delta.multiply(mask[:, None]).tocsr()
        tree.deltaT = tree.delta.T.tocsc()
        tree.c0, tree._solver, tree._spectrum = cx.c0, None, None

        rng = np.random.default_rng(0)
        apex = sheaf.apex_cells(dome)[0]
        for cell in dome.cells:
            if cell.id == apex:
                continue
            chi = cx.chi(
                cell.id,
                sheaf.unit(rng, cell.stalk),
                apex,
                sheaf.unit(rng, dome.cells[apex].stalk),
            )
            eps = 1e-8 * cx.scale
            assert tree.regularised(chi, eps) >= cx.regularised(chi, eps) - 1e-9


class TestPrivateDirections:
    def test_a_private_direction_leaks_all_of_itself(self):
        """The measurement's whole point: rank deficiency is infinite resistance.

        A direction in the kernel of every map incident on a cell is in `ker L`,
        so no finite current supplies it and `leak` is 1 for the half of `chi`
        that carries it. The structural mask makes such directions by
        construction, which is what `Dome.private_mask` is.
        """
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        cx = sheaf.Complex(dome, maps)
        cell = next(
            c
            for c in dome.cells
            if not c.is_boundary and c.stalk > dome._permitted[c.id]
        )
        private = np.zeros(cell.stalk)
        private[dome._permitted[cell.id]] = 1.0  # the first masked direction
        for e in dome.incident[cell.id]:
            assert np.allclose(cx.blocks[(e, cell.id)] @ private, 0.0)
        apex = sheaf.apex_cells(dome)[0]
        target = np.zeros(dome.cells[apex].stalk)
        target[0] = 1.0
        _R, leak = cx.resistance(cx.chi(cell.id, private, apex, target))
        assert leak > 0.4, "half of chi is unsuppliable, so at least ~0.5 must leak"

    def test_the_public_basis_excludes_the_mask(self):
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        cx = sheaf.Complex(dome, maps)
        for cell in dome.cells:
            if cell.is_boundary:
                continue
            basis = cx.public_basis(cell.id)
            assert basis.shape[0] <= dome._permitted[cell.id]
            if basis.shape[0]:
                assert np.allclose(basis[:, dome._permitted[cell.id] :], 0.0, atol=1e-12)


class TestChannel:
    def test_the_chain_delivers_its_top_pair_with_a_positive_sign(self):
        """`C a = sigma_1 b`, `sigma_1 > 0` — the orientation `e_u - e_v` fixes."""
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        cx = sheaf.Complex(dome, maps)
        apex = set(sheaf.apex_cells(dome))
        source = next(c.id for c in dome.cells if c.kind.value == "patch")
        path = sheaf.shortest_path(dome, source, apex)
        assert path is not None and len(path) > 2
        C = sheaf.chain(cx, path)
        a, b, sigma = sheaf.channel(C)
        assert sigma > 0
        assert np.allclose(C @ a, sigma * b, atol=1e-10)
        assert np.linalg.norm(a) == pytest.approx(1.0)
        assert np.linalg.norm(b) == pytest.approx(1.0)
        # It is the *top* pair, so nothing else on the sphere beats it.
        rng = np.random.default_rng(0)
        for _ in range(32):
            assert np.linalg.norm(C @ sheaf.unit(rng, C.shape[1])) <= sigma + 1e-9

    def test_renormalising_each_hop_does_not_move_the_directions(self):
        """`chain` normalises after every hop, and that must be free.

        It is there because a trained surface's raw seven-hop product lands at
        `σ₁ ≈ 4e-17` — float64 noise against intermediates of order 1, which
        would make *along the channel* a random direction. Scaling cannot move a
        singular vector, so on a well-conditioned surface the normalised chain
        must return exactly the directions the raw product does. This is the
        check that the fix is a fix and not a change of quantity.
        """
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        cx = sheaf.Complex(dome, maps)
        apex = set(sheaf.apex_cells(dome))
        source = next(c.id for c in dome.cells if c.kind.value == "patch")
        path = sheaf.shortest_path(dome, source, apex)

        ids = sheaf.edges_of(dome, path)
        raw = cx.blocks[(ids[0], path[0])]
        for cell, e_in, e_out in zip(path[1:-1], ids, ids[1:]):
            raw = cx.blocks[(e_out, cell)] @ cx.blocks[(e_in, cell)].T @ raw
        raw = cx.blocks[(ids[-1], path[-1])].T @ raw

        a_raw, b_raw, _ = sheaf.channel(raw)
        a, b, _ = sheaf.channel(sheaf.chain(cx, path))
        # Singular vectors are fixed only up to a shared sign flip.
        sign = np.sign(a_raw @ a) or 1.0
        assert np.allclose(a, sign * a_raw, atol=1e-8)
        assert np.allclose(b, sign * b_raw, atol=1e-8)

    def test_the_path_walks_real_edges(self):
        dome = build_graph(SMALL)
        apex = set(sheaf.apex_cells(dome))
        source = next(c.id for c in dome.cells if c.kind.value == "patch")
        path = sheaf.shortest_path(dome, source, apex)
        ids = sheaf.edges_of(dome, path)
        assert len(ids) == len(path) - 1
        for e, (a, b) in zip(ids, zip(path, path[1:])):
            assert {dome.edges[e].u, dome.edges[e].v} == {a, b}


class TestBenchmark:
    """That the script still runs against the API, on the small dome."""

    def test_read_runs(self, capsys):
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        sheaf.read(dome, maps, "test", trials=2, seed=0)
        printed = capsys.readouterr().out
        assert "per stratum" in printed
        assert "in leaf edges" in printed

    def test_control_runs(self, capsys):
        sheaf.control(build_graph(SMALL))
        assert "trivial sheaf is the graph" in capsys.readouterr().out
