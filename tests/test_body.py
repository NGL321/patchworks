"""The cell body (tickets #84 and #138, docs/spec/01-cell-and-sheaf.md).

What these tests hold down is the construction, not the behaviour: the width,
the freeze, the shape invariant `k < n`, the operator band, and that the
population is evaluated in one batched pass rather than a loop over cells.
Nothing here trains anything -- nothing in the frozen half of the body ever
will.

After the Koopman conversion the body is `encode` (nonlinear, frozen), the
per-cell operator `K` (linear, learned), and `decode` (linear, frozen as a
gauge). `encode` is the only nonlinearity, so it is the only map with a hidden
layer, a fold margin, or an activation region.
"""

import pytest
import torch

from patchworks.body import (
    DEFAULT_RHO_K,
    BodyShape,
    CellBiases,
    CellBody,
    CellOperators,
    hidden_width,
)

# The proof of concept's values (docs/spec/06-graph-topology.md).
N, K = 32, 12
CELLS = 150


@pytest.fixture
def shape():
    return BodyShape(n=N, k=K)


@pytest.fixture
def body(shape):
    return CellBody(shape, generator=torch.Generator().manual_seed(0))


# The body's buffers are not all weights: `fold_gradient_norms` is a derived
# constant (#197), non-persistent and recomputed after any load. The three
# construction tests below are about the *weights*, so they iterate these.
WEIGHTS = ("encode_hidden_weight", "encode_output_weight", "decode_weight")


def weights(body):
    return [(name, getattr(body, name)) for name in WEIGHTS]


@pytest.fixture
def biases(shape):
    return CellBiases(shape, CELLS, generator=torch.Generator().manual_seed(1))


@pytest.fixture
def operators(shape):
    return CellOperators(shape, CELLS)


class TestWidths:
    def test_the_rule_is_the_floor(self):
        assert hidden_width(44, 12) == 45
        assert hidden_width(12, 12) == 13
        assert hidden_width(12, 32) == 32

    def test_poc_width(self, shape):
        # encode: R^12 x R^32 -> R^12. It is the only map with a hidden layer:
        # `K` and `decode` are linear (#138).
        assert shape.encode_width == 45

    def test_the_width_rederives_from_the_rule_when_n_and_k_move(self):
        # n/k and k are both rungs on the flex ladder: pull one and the width
        # re-derives itself rather than staying a constant.
        assert BodyShape(n=64, k=8).encode_width == 73

    def test_the_linear_maps_have_no_hidden_width(self, shape):
        # The conversion deleted them, and reaching for one should say so
        # rather than quietly return something plausible.
        assert not hasattr(shape, "step_width")
        assert not hasattr(shape, "decode_width")

    def test_weight_shapes_follow_the_width(self, body):
        assert body.encode_hidden_weight.shape == (45, K + N)
        assert body.encode_output_weight.shape == (K, 45)
        # `D`, one matrix and no hidden layer.
        assert body.decode_weight.shape == (N, K)

    def test_the_frozen_half_is_encode_and_decode_and_nothing_else(self, body):
        held = [name for name, _ in body.named_buffers()]
        assert sorted(held) == [
            "decode_weight",
            "encode_hidden_weight",
            "encode_output_weight",
            # Not a fourth weight: the row norms of the first, cached because
            # they are the denominator of every fold-margin read and the
            # numerator is the only per-cell half (#197, ADR-0019).
            "fold_gradient_norms",
        ]

    def test_the_derived_constant_is_not_a_second_declaration(self, body, shape):
        # The failure #185 exists to kill: two places holding one quantity with
        # nothing between them. Non-persistent, so it is never loaded, and the
        # load hook recomputes it, so a swapped-in body cannot leave it stale.
        assert "fold_gradient_norms" not in body.state_dict()
        other = CellBody(shape, generator=torch.Generator().manual_seed(11))
        body.load_state_dict(other.state_dict())
        assert torch.equal(
            body.fold_gradient_norms,
            torch.linalg.vector_norm(other.encode_hidden_weight[:, shape.k :], dim=-1),
        )


class TestShapeInvariant:
    def test_k_is_below_n(self, shape):
        # The low-dimensional requirement: a shape invariant no training story
        # may violate.
        assert shape.k < shape.n

    @pytest.mark.parametrize("n, k", [(32, 32), (12, 32)])
    def test_k_at_or_above_n_is_refused(self, n, k):
        with pytest.raises(ValueError, match="k < n"):
            BodyShape(n=n, k=k)


class TestTheFreeze:
    def test_body_weights_require_no_gradient(self, body):
        for name, weight in weights(body):
            assert weight.requires_grad is False, name

    def test_the_body_exposes_no_parameters_at_all(self, body):
        # Registered as buffers, so no optimiser can reach them: the freeze is
        # enforced, not a convention.
        assert list(body.parameters()) == []

    def test_the_surface_is_the_biases_and_the_operators(self, body, biases, operators):
        # Buffers are the frozen body; parameters are the adapting surface.
        trainable = [p for p in body.parameters() if p.requires_grad]
        trainable += [p for p in biases.parameters() if p.requires_grad]
        trainable += [p for p in operators.parameters() if p.requires_grad]
        assert len(trainable) == 4  # three bias vectors and K
        assert all(p.shape[0] == CELLS for p in trainable)

    def test_the_surface_is_233_numbers_per_cell(self, biases, operators):
        # #138's ledger: 146 became 89 + K's 144. The conversion deletes 25
        # learned per-cell numbers with `step` and 13 more with `decode`'s
        # hidden layer, before it adds anything.
        per_cell = sum(p[0].numel() for p in biases.parameters())
        assert per_cell == 89
        assert operators.K[0].numel() == 144
        assert per_cell + operators.K[0].numel() == 233

    def test_biases_carry_a_leading_cell_dimension(self, biases):
        assert biases.encode_hidden_bias.shape == (CELLS, 45)
        assert biases.encode_output_bias.shape == (CELLS, K)
        assert biases.decode_output_bias.shape == (CELLS, N)

    def test_the_deleted_bias_vectors_are_gone(self, biases):
        # `step`'s pair went with the map, and `decode`'s hidden bias with the
        # hidden layer. Only `decode`'s output bias survives: the constant
        # observable.
        names = sorted(name for name, _ in biases.named_parameters())
        assert names == [
            "decode_output_bias",
            "encode_hidden_bias",
            "encode_output_bias",
        ]

    def test_gradient_reaches_the_surface_through_the_frozen_path(
        self, body, biases, operators
    ):
        # The prediction rule is a local gradient step *through* the frozen
        # half of the path, so the freeze must not block autograd -- only
        # ownership. `K` is on that path and takes a gradient with the biases,
        # in one backward pass (#139).
        chart = torch.zeros(CELLS, K)
        stalk = torch.randn(CELLS, N)
        _, prediction = body(chart, stalk, biases, operators)
        prediction.pow(2).sum().backward()
        for name, parameter in list(biases.named_parameters()) + list(
            operators.named_parameters()
        ):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    def test_the_body_is_shared_across_cells(self, body):
        # One set of weights for the whole population: nothing about the body
        # carries a cell index.
        for _, weight in weights(body):
            assert weight.ndim == 2
            assert CELLS not in weight.shape


class TestInitialisation:
    def test_the_draw_is_non_degenerate(self, body):
        for name, weight in weights(body):
            rank = torch.linalg.matrix_rank(weight)
            assert rank == min(weight.shape), name

    def test_the_draw_is_reproducible_from_a_generator(self, shape):
        one = CellBody(shape, generator=torch.Generator().manual_seed(7))
        two = CellBody(shape, generator=torch.Generator().manual_seed(7))
        assert torch.equal(one.encode_hidden_weight, two.encode_hidden_weight)
        assert torch.equal(one.decode_weight, two.decode_weight)

    def test_cells_differ_only_by_their_biases(self, shape, body):
        drawn = CellBiases(shape, CELLS, generator=torch.Generator().manual_seed(3))
        chart = torch.randn(CELLS, K)
        stalk = torch.randn(1, N).expand(CELLS, N)
        # Same body, same evidence, same prior -- the biases alone separate the
        # cells, by translating the folds of the shared map.
        constant_chart = chart[:1].expand(CELLS, K)
        fused = body.encode(constant_chart, stalk, drawn)
        assert not torch.allclose(fused[0], fused[1])


class TestForwardPath:
    def test_the_three_maps_have_the_specified_signatures(
        self, body, biases, operators
    ):
        chart = torch.randn(CELLS, K)
        stalk = torch.randn(CELLS, N)
        fused = body.encode(chart, stalk, biases)
        assert fused.shape == (CELLS, K)
        assert operators.advance(fused).shape == (CELLS, K)
        assert body.decode(fused, biases).shape == (CELLS, N)

    def test_encode_fuses_prior_belief_with_new_evidence(self, body, biases):
        # Both arguments move the fused chart: it is a fusion, not a re-read of
        # the node stalk.
        chart = torch.randn(CELLS, K)
        stalk = torch.randn(CELLS, N)
        fused = body.encode(chart, stalk, biases)
        assert not torch.allclose(fused, body.encode(torch.randn(CELLS, K), stalk, biases))
        assert not torch.allclose(fused, body.encode(chart, torch.randn(CELLS, N), biases))

    def test_forward_returns_the_advanced_chart_and_the_prediction(
        self, body, biases, operators
    ):
        chart = torch.randn(CELLS, K)
        stalk = torch.randn(CELLS, N)
        advanced, prediction = body(chart, stalk, biases, operators)
        assert advanced.shape == (CELLS, K)
        assert prediction.shape == (CELLS, N)
        expected_advanced, expected_prediction = body.advance(
            body.encode(chart, stalk, biases), biases, operators
        )
        assert torch.equal(advanced, expected_advanced)
        assert torch.equal(prediction, expected_prediction)

    def test_decode_is_linear_in_the_chart(self, body, biases):
        # The readout gauge's claim, read straight off the map: `D z + b` is
        # affine, so a doubled step in the chart doubles the step in the
        # prediction, everywhere and not only inside a region.
        chart = torch.randn(CELLS, K, dtype=torch.float64)
        nudge = torch.randn(CELLS, K, dtype=torch.float64)
        double = CellBody(body.shape, dtype=torch.float64)
        with torch.no_grad():
            double.decode_weight.copy_(body.decode_weight.double())
        wide = CellBiases(body.shape, CELLS, dtype=torch.float64)
        with torch.no_grad():
            wide.decode_output_bias.copy_(biases.decode_output_bias.double())
        one = double.decode(chart, wide)
        two = double.decode(chart + nudge, wide)
        three = double.decode(chart + 2 * nudge, wide)
        assert torch.allclose(three - two, two - one, atol=1e-12)

    def test_the_activation_is_relu(self, shape):
        # Piecewise-linear is what the fold vocabulary is made of, and ReLU is
        # the instance: drive `encode`'s hidden layer far negative and the map
        # is exactly its output bias. `encode` is the only map this can be
        # asked of now -- the other two are linear.
        body = CellBody(shape, generator=torch.Generator().manual_seed(11))
        biases = CellBiases(shape, CELLS, bias_variance=0.0)
        with torch.no_grad():
            biases.encode_hidden_bias -= 1e6
        fused = body.encode(torch.randn(CELLS, K), torch.randn(CELLS, N), biases)
        assert torch.allclose(fused, torch.zeros(CELLS, K))

    def test_encode_is_exactly_affine_inside_one_activation_region(self, shape):
        # The partition into convex polytopes on each of which the map is
        # exactly affine is the object the timescale mechanism is made of.
        # Read in float64: the finite differences below cancel away most of a
        # float32 mantissa, and the claim is exactness, not closeness.
        #
        # The tolerance was measured, not guessed. At this seed the smallest
        # |preactivation| over all 1950 hidden units is 1.7e-3, so a 1e-6 nudge
        # cannot carry any cell across a fold, and the observed float64 slack
        # was ~3.5e-15. torch is pinned exactly, so this is deterministic --
        # but a torch bump or a BLAS swap could shift the slack, and the fix
        # then is to re-measure the margin rather than to loosen atol.
        generator = torch.Generator().manual_seed(23)
        body = CellBody(shape, generator=generator, dtype=torch.float64)
        biases = CellBiases(shape, CELLS, generator=generator, dtype=torch.float64)
        chart = torch.randn(CELLS, K, generator=generator, dtype=torch.float64)
        stalk = torch.randn(CELLS, N, generator=generator, dtype=torch.float64)
        nudge = 1e-6 * torch.randn(CELLS, K, generator=generator, dtype=torch.float64)
        one = body.encode(chart, stalk, biases)
        two = body.encode(chart + nudge, stalk, biases)
        three = body.encode(chart + 2 * nudge, stalk, biases)
        assert torch.allclose(three - two, two - one, atol=1e-15)

    @pytest.mark.parametrize("bad", [torch.zeros(CELLS, K + 1), torch.zeros(K)])
    def test_a_misshaped_chart_is_refused(self, body, biases, operators, bad):
        with pytest.raises(ValueError, match="chart"):
            body.decode(bad, biases)
        with pytest.raises(ValueError, match="chart"):
            operators.advance(bad)


class TestBatchedExecution:
    def test_the_population_is_one_evaluation_not_a_loop(
        self, body, biases, operators
    ):
        # A per-cell loop must agree with the batched pass exactly -- that
        # equality is what licenses only ever running the batched one. `K` is
        # `[cells, k, k]` and one `bmm`, so it batches over a leading dimension
        # like everything else rather than being many small models.
        chart = torch.randn(CELLS, K)
        stalk = torch.randn(CELLS, N)
        with torch.no_grad():
            operators.K.normal_(0.0, 0.2, generator=torch.Generator().manual_seed(5))
        advanced, prediction = body(chart, stalk, biases, operators)

        for cell in range(0, CELLS, 37):
            one = CellBiases(body.shape, 1, bias_variance=0.0)
            alone = CellOperators(body.shape, 1)
            with torch.no_grad():
                for name, bias in biases.named_parameters():
                    getattr(one, name).copy_(bias[cell : cell + 1])
                alone.K.copy_(operators.K[cell : cell + 1])
            alone_advanced, alone_prediction = body(
                chart[cell : cell + 1], stalk[cell : cell + 1], one, alone
            )
            assert torch.allclose(alone_advanced, advanced[cell : cell + 1], atol=1e-6)
            assert torch.allclose(alone_prediction, prediction[cell : cell + 1], atol=1e-6)

    def test_the_population_size_is_not_baked_into_the_body(self, shape, body):
        # One body, any number of cells: the cell count lives on the per-cell
        # surface.
        for cells in (1, 7, CELLS):
            biases = CellBiases(shape, cells)
            operators = CellOperators(shape, cells)
            out, prediction = body(
                torch.randn(cells, K), torch.randn(cells, N), biases, operators
            )
            assert out.shape == (cells, K)
            assert prediction.shape == (cells, N)


class TestMismatchedSurfaces:
    def test_biases_from_a_different_shape_are_refused(self, body):
        other = CellBiases(BodyShape(n=64, k=8), CELLS)
        with pytest.raises(ValueError, match="this body"):
            body.encode(torch.randn(CELLS, K), torch.randn(CELLS, N), other)

    def test_one_bias_vector_is_not_broadcast_over_a_population(self, shape, body):
        single = CellBiases(shape, 1)
        with pytest.raises(ValueError, match="cells"):
            body.encode(torch.randn(CELLS, K), torch.randn(CELLS, N), single)

    def test_one_operator_is_not_broadcast_over_a_population(self, shape):
        single = CellOperators(shape, 1)
        with pytest.raises(ValueError, match="cells"):
            single.advance(torch.randn(CELLS, K))


class TestSubset:
    """What selection keeps (ticket #85): a population holding a subset of rows."""

    def test_the_kept_rows_are_the_ones_asked_for_in_that_order(self, shape, biases):
        index = torch.tensor([7, 3, 3, 0])
        kept = biases.subset(index)
        assert kept.cells == 4
        for name, parameter in biases.named_parameters():
            assert torch.equal(getattr(kept, name), parameter[index])

    def test_keeping_nothing_is_representable(self, biases):
        # A band no draw reached keeps nothing, which is a result rather than a
        # misuse -- even though a population of zero cells cannot be drawn.
        kept = biases.subset(torch.tensor([], dtype=torch.long))
        assert kept.cells == 0

    def test_the_kept_set_is_detached_from_the_draw_it_came_from(self, biases):
        kept = biases.subset(torch.tensor([1]))
        with torch.no_grad():
            kept.encode_hidden_bias.add_(1.0)
        assert not torch.equal(kept.encode_hidden_bias[0], biases.encode_hidden_bias[1])

    def test_keeping_a_set_does_not_advance_the_global_rng(self, biases):
        # Selection keeps however many candidates a band happened to offer, so a
        # draw made on the global generator here would shift every later draw by
        # an amount that depends on the result -- inside a construction whose
        # contract is that it reproduces from its seed.
        torch.manual_seed(0)
        before = torch.rand(1)
        torch.manual_seed(0)
        biases.subset(torch.tensor([0, 2, 5]))
        assert torch.equal(torch.rand(1), before)

    def test_an_index_that_is_not_one_dimensional_is_refused(self, biases):
        with pytest.raises(ValueError, match=r"index must be \[cells\]"):
            biases.subset(torch.tensor([[0, 1]]))

    def test_a_mask_is_refused_rather_than_half_understood(self, biases):
        # Selection works in masks, so passing one here is the natural slip; it
        # would give a population whose `cells` was the mask's length rather
        # than its count, and only a downstream shape check would notice.
        mask = torch.zeros(CELLS, dtype=torch.bool)
        mask[:3] = True
        with pytest.raises(ValueError, match="not mask them"):
            biases.subset(mask)
        assert biases.subset(mask.nonzero(as_tuple=False).flatten()).cells == 3


class TestTheOperators:
    """`K`, the per-cell operator the conversion put in `step`'s place (#138)."""

    def test_k_is_dense_per_cell_and_square_in_the_chart(self, operators):
        assert operators.K.shape == (CELLS, K, K)

    def test_k_is_the_identity_scaled_at_construction(self, shape):
        # `a.I` rather than a random draw, so an untrained graph is quiescent
        # rather than noisy: a cell not yet predicting anything should not be
        # emitting.
        built = CellOperators(shape, 4, scale=0.5)
        assert torch.equal(built.K, 0.5 * torch.eye(K).expand(4, K, K))

    def test_k_carries_no_bias(self, operators):
        # An affine `K` makes the *dynamics* affine -- a drift that compounds
        # every tick, which is the persistent offset ADR-0004 refuses to let a
        # linear map launder away. `decode`'s output bias is the permitted
        # kind: a static readout offset that never accumulates.
        assert [name for name, _ in operators.named_parameters()] == ["K"]

    def test_advancing_is_exactly_the_matrix_product(self, operators):
        # Of the *used* operator since #433, which is the whole of the change:
        # the raw `K` is the learned parameter and is not what the cell
        # computes with.
        with torch.no_grad():
            operators.K.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(2))
        chart = torch.randn(CELLS, K)
        used = operators.used().detach()
        expected = torch.stack([used[c] @ chart[c] for c in range(CELLS)])
        assert torch.allclose(operators.advance(chart), expected, atol=1e-6)

    def test_a_scale_outside_the_band_is_refused(self, shape):
        # The band is `[1/rho_K, 1]`, and `a` is inside it or it is not `a`.
        with pytest.raises(ValueError, match=r"\[1/rho_k, 1\]"):
            CellOperators(shape, 4, scale=1.5)
        with pytest.raises(ValueError, match=r"\[1/rho_k, 1\]"):
            CellOperators(shape, 4, scale=0.4, rho_k=2.0)


class TestTheOperatorBand:
    """#140: the band is on the norm, not the radius, and it is spectral.

    #433 moved the enforcement into the forward path, so what these hold is
    that `sigma_max(used)` is in band **identically** rather than that a
    projection restores it. The band, both faces and the norm choice are
    unchanged; only the mechanism is.
    """

    def test_the_band_holds_from_above(self, operators):
        # The upper face is exactly 1: what it forbids is amplification, and a
        # cell sitting at 1 is non-expansive rather than divergent. No step
        # restores this -- the used operator never leaves.
        with torch.no_grad():
            operators.K.mul_(50.0)
        assert torch.allclose(operators.norms.detach(), torch.ones(CELLS), atol=1e-5)

    def test_the_band_holds_from_below(self, operators):
        # Two-sided: the lower face is kept, so an operator that has shrunk
        # away is scaled back *up* into the band rather than left there.
        with torch.no_grad():
            operators.K.mul_(1e-4)
        assert torch.allclose(
            operators.norms.detach(),
            torch.full((CELLS,), 1.0 / DEFAULT_RHO_K),
            atol=1e-5,
        )

    def test_an_operator_already_inside_the_band_is_used_unchanged(self, operators):
        # In band, `used` is the identity on the value as well as the referent:
        # the factor is exactly 1 and nothing is rescaled.
        with torch.no_grad():
            operators.K.mul_(0.75)
        assert torch.allclose(
            operators.used().detach(), operators.K.detach(), atol=1e-6
        )

    def test_the_learned_parameter_is_left_where_the_gradient_put_it(self, operators):
        # The counterpart of the above and the point of the move: enforcement
        # no longer writes to `K`. Nothing between steps rescales the thing the
        # prediction rule is descending.
        with torch.no_grad():
            operators.K.mul_(50.0)
        before = operators.K.detach().clone()
        _ = operators.advance(torch.randn(CELLS, K))
        assert torch.allclose(operators.K.detach(), before, atol=1e-6)

    def test_the_constrained_quantity_is_the_norm_not_the_radius(self, shape):
        # The whole of #140's correction, as a construction: a matrix whose
        # spectral radius is small and whose spectral norm is enormous. A band
        # written on the radius would pass this untouched and leave `body` an
        # unbounded factor in the transmission budget.
        operators = CellOperators(shape, 1)
        with torch.no_grad():
            operators.K.zero_()
            operators.K[0, 0, 1] = 50.0
        assert float(operators.raw_radii()[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(operators.raw_norms.detach()[0]) == pytest.approx(50.0, rel=1e-4)
        assert float(operators.norms.detach()[0]) == pytest.approx(1.0, rel=1e-4)

    def test_the_normalisation_is_a_rescale_and_keeps_the_direction(self, operators):
        # ADR-0010's mechanism, unchanged by the move: the whole operator is
        # rescaled, so what the band restores is magnitude and never structure.
        # The rescale is still *radial*, which is #335's complaint and is what
        # #433 did not claim to fix -- it fixed *when*, not *what*.
        with torch.no_grad():
            operators.K.normal_(0.0, 2.0, generator=torch.Generator().manual_seed(9))
        ratio = (operators.used().detach() / operators.K.detach()).flatten(1)
        assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-5)

    def test_the_gradient_sees_the_constraint(self, operators):
        # What the move bought. At the upper face the normalisation's Jacobian
        # removes the purely radial component: scaling `K` up cannot scale the
        # output up, so the gradient of any loss with respect to that direction
        # is zero rather than an uncorrelated shove landing after the step.
        with torch.no_grad():
            operators.K.normal_(0.0, 2.0, generator=torch.Generator().manual_seed(4))
        chart = torch.randn(CELLS, K)
        before = operators.advance(chart).detach().clone()
        with torch.no_grad():
            operators.K.mul_(3.0)
        assert torch.allclose(operators.advance(chart).detach(), before, atol=1e-5)

    def test_a_used_operator_is_in_band_for_any_parameter(self, shape):
        # The band as an identity rather than an invariant something restores.
        operators = CellOperators(shape, 8)
        with torch.no_grad():
            operators.K.normal_(0.0, 5.0, generator=torch.Generator().manual_seed(11))
        norms = operators.norms.detach()
        assert bool((norms <= 1.0 + 1e-5).all())
        assert bool((norms >= 1.0 / DEFAULT_RHO_K - 1e-5).all())

    def test_a_dead_operator_is_reported_dead_rather_than_lifted(self, shape):
        # Below `NORM_FLOOR` there is no direction to rescale, so `used` leaves
        # it and `norms` says so. Claiming the floor here would report a
        # retention the cell does not have.
        operators = CellOperators(shape, 1)
        with torch.no_grad():
            operators.K.zero_()
        assert float(operators.norms.detach()[0]) == 0.0
        assert float(operators.used().detach().abs().max()) == 0.0

    def test_a_band_below_one_is_refused(self, shape):
        with pytest.raises(ValueError, match="rho_k >= 1"):
            CellOperators(shape, 4, rho_k=0.5)

    def test_the_kept_operators_are_the_ones_asked_for_in_that_order(self, operators):
        with torch.no_grad():
            operators.K.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(4))
        index = torch.tensor([7, 3, 3, 0])
        kept = operators.subset(index)
        assert kept.cells == 4
        assert torch.equal(kept.K, operators.K[index])


class TestBenchmark:
    """`benchmarks/body_forward.py` is the reported wall time's provenance.

    Timings are not asserted -- a wall-clock threshold in CI measures the
    runner, not the body. What is asserted is that the script still runs
    against the current API, so the number in `09-the-build-stack.md` keeps a
    reproduction.
    """

    def test_the_benchmark_still_runs(self):
        import body_forward

        samples = body_forward.time_forward(8, BodyShape(n=8, k=3), repeats=3)
        assert len(samples) == 3
        assert all(sample > 0 for sample in samples)
