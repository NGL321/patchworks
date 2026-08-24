"""The shared frozen cell body (ticket #84, docs/spec/01-cell-and-sheaf.md).

What these tests hold down is the construction, not the behaviour: the widths,
the freeze, the shape invariant `k < n`, and that the population is evaluated
in one batched pass rather than a loop over cells. Nothing here trains
anything -- nothing in the body ever will.
"""

import pytest
import torch

from patchworks.body import BodyShape, CellBiases, CellBody, hidden_width

# The proof of concept's values (docs/spec/06-graph-topology.md).
N, K = 32, 12
CELLS = 150


@pytest.fixture
def shape():
    return BodyShape(n=N, k=K)


@pytest.fixture
def body(shape):
    return CellBody(shape, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def biases(shape):
    return CellBiases(shape, CELLS, generator=torch.Generator().manual_seed(1))


class TestWidths:
    def test_the_rule_is_the_floor(self):
        assert hidden_width(44, 12) == 45
        assert hidden_width(12, 12) == 13
        assert hidden_width(12, 32) == 32

    def test_poc_widths(self, shape):
        # encode: R^32 x R^12 -> R^12, step: R^12 -> R^12, decode: R^12 -> R^32.
        #
        # ESCALATED (ticket #84): the record prints 45 / 13 / 33 here, but
        # `max{d_x + 1, d_y}` at decode is max(13, 32) = 32. The rule is
        # implemented, not the printed constant -- see patchworks.body.hidden_width
        # for why, and revisit this assertion when the record is settled.
        assert (shape.encode_width, shape.step_width, shape.decode_width) == (45, 13, 32)

    def test_widths_rederive_from_the_rule_when_n_and_k_move(self):
        # n/k and k are both rungs on the flex ladder: pull one and the widths
        # re-derive themselves rather than staying at three constants.
        other = BodyShape(n=64, k=8)
        assert other.encode_width == 73
        assert other.step_width == 9
        assert other.decode_width == 64

    def test_weight_shapes_follow_the_widths(self, body):
        assert body.encode_hidden_weight.shape == (45, K + N)
        assert body.encode_output_weight.shape == (K, 45)
        assert body.step_hidden_weight.shape == (13, K)
        assert body.step_output_weight.shape == (K, 13)
        assert body.decode_hidden_weight.shape == (32, K)
        assert body.decode_output_weight.shape == (N, 32)

    def test_one_hidden_layer_per_map(self, body):
        # Two weight matrices per map is one hidden layer. A second layer is
        # measured expensive and over-subscribes the bias vector further.
        weights = [name for name, _ in body.named_buffers()]
        assert sorted(weights) == sorted(
            f"{m}_{role}_weight"
            for m in ("encode", "step", "decode")
            for role in ("hidden", "output")
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
        for name, weight in body.named_buffers():
            assert weight.requires_grad is False, name

    def test_the_body_exposes_no_parameters_at_all(self, body):
        # Registered as buffers, so no optimiser can reach them: the freeze is
        # enforced, not a convention.
        assert list(body.parameters()) == []

    def test_biases_are_the_only_trainable_surface(self, body, biases):
        trainable = [p for p in body.parameters() if p.requires_grad]
        trainable += [p for p in biases.parameters() if p.requires_grad]
        assert len(trainable) == 6  # hidden and output, for each of three maps
        assert all(p.shape[0] == CELLS for p in trainable)

    def test_biases_carry_a_leading_cell_dimension(self, biases):
        assert biases.encode_hidden_bias.shape == (CELLS, 45)
        assert biases.encode_output_bias.shape == (CELLS, K)
        assert biases.step_hidden_bias.shape == (CELLS, 13)
        assert biases.step_output_bias.shape == (CELLS, K)
        assert biases.decode_hidden_bias.shape == (CELLS, 32)
        assert biases.decode_output_bias.shape == (CELLS, N)

    def test_gradient_reaches_the_biases_through_the_frozen_path(self, body, biases):
        # The bias rule is a local gradient step *through* the frozen forward
        # path, so the freeze must not block autograd -- only ownership.
        chart = torch.zeros(CELLS, K)
        stalk = torch.randn(CELLS, N)
        _, prediction = body(chart, stalk, biases)
        prediction.pow(2).sum().backward()
        for name, bias in biases.named_parameters():
            assert bias.grad is not None, name
            assert torch.isfinite(bias.grad).all(), name

    def test_the_body_is_shared_across_cells(self, body, biases):
        # One set of weights for the whole population: nothing about the body
        # carries a cell index.
        for _, weight in body.named_buffers():
            assert weight.ndim == 2
            assert CELLS not in weight.shape


class TestInitialisation:
    def test_the_draw_is_non_degenerate(self, body):
        for name, weight in body.named_buffers():
            rank = torch.linalg.matrix_rank(weight)
            assert rank == min(weight.shape), name

    def test_the_draw_is_reproducible_from_a_generator(self, shape):
        one = CellBody(shape, generator=torch.Generator().manual_seed(7))
        two = CellBody(shape, generator=torch.Generator().manual_seed(7))
        assert torch.equal(one.step_hidden_weight, two.step_hidden_weight)

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
    def test_the_three_maps_have_the_specified_signatures(self, body, biases):
        chart = torch.randn(CELLS, K)
        stalk = torch.randn(CELLS, N)
        fused = body.encode(chart, stalk, biases)
        assert fused.shape == (CELLS, K)
        assert body.step(fused, biases).shape == (CELLS, K)
        assert body.decode(fused, biases).shape == (CELLS, N)

    def test_encode_fuses_prior_belief_with_new_evidence(self, body, biases):
        # Both arguments move the fused chart: it is a fusion, not a re-read of
        # the node stalk.
        chart = torch.randn(CELLS, K)
        stalk = torch.randn(CELLS, N)
        fused = body.encode(chart, stalk, biases)
        assert not torch.allclose(fused, body.encode(torch.randn(CELLS, K), stalk, biases))
        assert not torch.allclose(fused, body.encode(chart, torch.randn(CELLS, N), biases))

    def test_forward_returns_the_advanced_chart_and_the_prediction(self, body, biases):
        chart = torch.randn(CELLS, K)
        stalk = torch.randn(CELLS, N)
        advanced, prediction = body(chart, stalk, biases)
        assert advanced.shape == (CELLS, K)
        assert prediction.shape == (CELLS, N)
        expected_advanced, expected_prediction = body.advance(
            body.encode(chart, stalk, biases), biases
        )
        assert torch.equal(advanced, expected_advanced)
        assert torch.equal(prediction, expected_prediction)

    def test_the_activation_is_relu(self, shape):
        # Piecewise-linear is what the timescale mechanism is made of, and ReLU
        # is the instance: drive the hidden layer far negative and the map is
        # exactly its output bias.
        body = CellBody(shape, generator=torch.Generator().manual_seed(11))
        biases = CellBiases(shape, CELLS, bias_variance=0.0)
        with torch.no_grad():
            biases.step_hidden_bias -= 1e6
        chart = torch.randn(CELLS, K)
        assert torch.allclose(body.step(chart, biases), torch.zeros(CELLS, K))

    def test_the_map_is_exactly_affine_inside_one_activation_region(self, shape):
        # The partition into convex polytopes on each of which the map is
        # exactly affine is the object the timescale mechanism is made of.
        # Read in float64: the finite differences below cancel away most of a
        # float32 mantissa, and the claim is exactness, not closeness.
        generator = torch.Generator().manual_seed(23)
        body = CellBody(shape, generator=generator, dtype=torch.float64)
        biases = CellBiases(shape, CELLS, generator=generator, dtype=torch.float64)
        chart = torch.randn(CELLS, K, generator=generator, dtype=torch.float64)
        nudge = 1e-6 * torch.randn(CELLS, K, generator=generator, dtype=torch.float64)
        one = body.step(chart, biases)
        two = body.step(chart + nudge, biases)
        three = body.step(chart + 2 * nudge, biases)
        assert torch.allclose(three - two, two - one, atol=1e-15)

    @pytest.mark.parametrize("bad", [torch.zeros(CELLS, K + 1), torch.zeros(K)])
    def test_a_misshaped_chart_is_refused(self, body, biases, bad):
        with pytest.raises(ValueError, match="chart"):
            body.step(bad, biases)


class TestBatchedExecution:
    def test_the_population_is_one_evaluation_not_a_loop(self, body, biases):
        # A per-cell loop must agree with the batched pass exactly -- that
        # equality is what licenses only ever running the batched one.
        chart = torch.randn(CELLS, K)
        stalk = torch.randn(CELLS, N)
        advanced, prediction = body(chart, stalk, biases)

        for cell in range(0, CELLS, 37):
            one = CellBiases(body.shape, 1, bias_variance=0.0)
            with torch.no_grad():
                for name, bias in biases.named_parameters():
                    getattr(one, name).copy_(bias[cell : cell + 1])
            alone_advanced, alone_prediction = body(
                chart[cell : cell + 1], stalk[cell : cell + 1], one
            )
            assert torch.allclose(alone_advanced, advanced[cell : cell + 1], atol=1e-6)
            assert torch.allclose(alone_prediction, prediction[cell : cell + 1], atol=1e-6)

    def test_the_population_size_is_not_baked_into_the_body(self, shape, body):
        # One body, any number of cells: the cell count lives on the biases.
        for cells in (1, 7, CELLS):
            biases = CellBiases(shape, cells)
            out, prediction = body(torch.randn(cells, K), torch.randn(cells, N), biases)
            assert out.shape == (cells, K)
            assert prediction.shape == (cells, N)


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
