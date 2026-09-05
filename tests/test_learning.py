"""The prediction rule: prediction error, live in the cell's own inference
parameters, over detached state (tickets #88 and #139).

`docs/spec/07-local-learning-rule.md`, *The prediction rule*, is what these hold
down, together with `docs/spec/09-the-build-stack.md`, *The locality guard*,
parts 1 and 2.

Two of them are the ones that catch a leak, and both are written as real
assertions rather than smoke checks: the batched gradient equals the per-cell
local gradient, and the body's weights receive no gradient. They are the
`torch.func` idiom's whole justification — the batched identity is what makes
`vmap` unnecessary, and the frozen body is what makes the identity exact.

Since #139 the rule is **widened**: it trains the biases *and* the per-cell
operator `K`, in the same local gradient step, on the same signal. The target is
not an allowlist -- it is `named_parameters()` whole -- so what these hold down
is that registering `K` as a parameter is the entire widening.
"""

import contextlib
import copy
from unittest import mock

import pytest
import torch

from patchworks.body import CellBiases, CellOperators
from patchworks.graph import build_graph
from patchworks.learning import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_OPERATOR_RATE_RATIO,
    ForwardPath,
    PredictionRule,
    prediction_error,
    prediction_gradient,
)
from patchworks.tick import Sheaf

from conftest import SMALL

#: What the widened rule trains: the three surviving bias vectors and `K`. The
#: names are `functional_call`'s, so each carries the module's attribute name on
#: the path.
TRAINED_NAMES = {
    "biases.encode_hidden_bias",
    "biases.encode_output_bias",
    "biases.decode_output_bias",
    "operators.K",
}


@pytest.fixture
def dome():
    return build_graph(SMALL)


@pytest.fixture
def sheaf(dome):
    return Sheaf(dome, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def running(sheaf):
    """A sheaf a few ticks into a run, with something in every buffer.

    A freshly built sheaf is all zeros and a fresh dome is unstimulated, so
    every quantity below would be trivially zero and half these assertions
    would pass on nothing.
    """
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        sheaf.stalks[: sheaf.layout.total] = torch.randn(
            sheaf.layout.total, generator=generator
        )
        sheaf.charts.normal_(0.0, 1.0, generator=generator)
        for edge in sheaf.dome.edges:
            for side in (0, 1):
                sheaf.broadcast[2 * edge.id + side, : edge.m].normal_(
                    0.0, 1.0, generator=generator
                )
    for _ in range(3):
        sheaf.tick()
    return sheaf


def biases_of(sheaf):
    return {name: p.detach().clone() for name, p in sheaf.biases.named_parameters()}


def surface_of(sheaf):
    """The whole per-cell surface -- the biases and `K` -- as detached copies."""
    kept = biases_of(sheaf)
    kept["K"] = sheaf.operators.K.detach().clone()
    return kept


def rule_inputs(sheaf, dtype=torch.float32):
    """The three detached arrays the rule reads, in the asked-for precision."""
    return tuple(
        tensor.to(dtype)
        for tensor in (sheaf.prior_charts, sheaf.prior_evidence, sheaf.evidence())
    )


def population(sheaf, dtype=torch.float32):
    """A copy of the body and the whole per-cell surface, in one precision.

    Copied rather than converted in place, so a double-precision reading of the
    identity below cannot change what the sheaf itself is running.
    """
    return (
        copy.deepcopy(sheaf.body).to(dtype),
        copy.deepcopy(sheaf.biases).to(dtype),
        copy.deepcopy(sheaf.operators).to(dtype),
    )


def one_cell(sheaf, biases, operators, cell):
    """The per-cell surface narrowed to cell `cell`, as a population of one.

    Deliberately not a slice of the batched call. A one-cell surface means
    nothing in the computation *could* span cells, so what it produces is the
    local gradient the batched one has to match. `K` is narrowed alongside the
    biases, since #139 made it part of what is differentiated.
    """
    dtype = biases.encode_hidden_bias.dtype
    alone = CellBiases(sheaf.dome.shape, 1).to(dtype)
    alone_operators = CellOperators(sheaf.dome.shape, 1).to(dtype)
    with torch.no_grad():
        for name, parameter in alone.named_parameters():
            parameter.copy_(getattr(biases, name)[cell : cell + 1])
        alone_operators.K.copy_(operators.K[cell : cell + 1])
    return alone, alone_operators


def gradient_of(body, biases, operators, inputs):
    """The rule's gradient for one population, straight off the transform."""
    path = ForwardPath(body, biases, operators)
    return prediction_gradient(path.trained_parameters(), path, *inputs)


def _live_scalar():
    """A one-element tensor that puts whatever multiplies it onto the tape."""
    return torch.ones(1, requires_grad=True)


class TestAPhaseSeparateFromTheTick:
    def test_the_tick_alone_trains_nothing(self, running):
        before = biases_of(running)
        running.tick()
        for name, parameter in running.biases.named_parameters():
            assert torch.equal(parameter, before[name])

    def test_the_rule_alone_moves_no_tick_state(self, running):
        before = {
            name: getattr(running, name).clone()
            for name in ("stalks", "charts", "prediction", "broadcast", "incoming")
        }
        PredictionRule(running).step()
        for name, value in before.items():
            assert torch.equal(getattr(running, name), value)

    def test_the_rule_runs_over_state_the_tick_left_detached(self, running):
        for tensor in (
            running.prior_charts,
            running.prior_evidence,
            running.evidence(),
        ):
            assert tensor.grad_fn is None and not tensor.requires_grad

    def test_the_recorded_inputs_are_what_the_inference_phase_read(self, sheaf):
        # The record is worth nothing if it is not the pair the forward path
        # actually ran on: re-running the body on it has to reproduce the tick's
        # own prediction exactly.
        charts_before = sheaf.charts.clone()
        evidence_before = sheaf.evidence()
        sheaf.inference_phase()
        assert torch.equal(sheaf.prior_charts, charts_before)
        assert torch.equal(sheaf.prior_evidence, evidence_before)

    def test_the_record_survives_the_message_passing_phase(self, running):
        # `prior_evidence` is a gather off the flat stalk buffer and
        # reconciliation edits that buffer in place. A view rather than a copy
        # would leave the rule re-running the forward path on the *post*-
        # reconciliation stalk, which is the target, not the input.
        recorded = running.prior_evidence.clone()
        running.message_passing_phase()
        assert torch.equal(running.prior_evidence, recorded)


class TestPredictionErrorIsRecomputedLive:
    def test_re_running_the_path_reproduces_the_ticks_prediction(self, running):
        path = ForwardPath(running.body, running.biases, running.operators)
        live = path(running.prior_charts, running.prior_evidence)
        assert torch.allclose(live, running.prediction, atol=1e-6)

    def test_moving_the_surface_moves_the_recomputed_prediction(self, running):
        # The other half of the test above: the path is re-run in the *current*
        # parameters, so a bias that has moved since the tick changes what it
        # predicts. A number carried over from the tick would not move at all.
        path = ForwardPath(running.body, running.biases, running.operators)
        with torch.no_grad():
            running.biases.decode_output_bias.add_(0.5)
        live = path(running.prior_charts, running.prior_evidence)
        assert not torch.allclose(live, running.prediction, atol=1e-3)

    def test_moving_k_moves_the_recomputed_prediction_too(self, running):
        # The widening's own version of the test above: `K` is on the same
        # re-run path, so moving it has to move the prediction. If it did not,
        # the rule would be descending on something `K` cannot reach.
        path = ForwardPath(running.body, running.biases, running.operators)
        with torch.no_grad():
            running.operators.K.mul_(0.5)
        live = path(running.prior_charts, running.prior_evidence)
        assert not torch.allclose(live, running.prediction, atol=1e-3)

    def test_the_target_is_the_stalk_reconciliation_left_behind(self, running):
        # Prediction error is measured against evidence, and reconciliation
        # edits that stalk between the two ticks. That edit is the whole of how
        # the neighbours' disagreement reaches the rule -- so if the target were
        # the prediction the tick emitted, the signal would be identically zero.
        assert not torch.allclose(running.evidence(), running.prediction, atol=1e-6)

    def test_the_gradient_is_the_gradient_of_the_live_objective(self, running):
        # Central differences on the objective itself. This is what pins the
        # rule to prediction error rather than to something that merely moves
        # when the biases do.
        body, biases, operators = population(running, torch.float64)
        # **`K` is moved off the construction point first, and #433 is why.**
        # The used operator is `K` normalised into the band, so the objective
        # has two non-smooth sets: the band's faces, and the matrices whose top
        # singular value is repeated, where `sigma_max` is not differentiable at
        # all. Construction sits on *both* -- `K = a.I` has every singular value
        # equal, and the default `a` is 1.0, the upper face exactly. A central
        # difference straddles a kink and matches neither one-sided derivative
        # there, which is a fact about the probe rather than about the rule.
        # Away from those sets the two agree exactly, which is what this holds.
        # The kink itself is held by
        # `TestTheUsedOperatorIsNotSmoothAtTheBandFace` below.
        with torch.no_grad():
            operators.K.normal_(0.0, 0.4, generator=torch.Generator().manual_seed(3))
        path = ForwardPath(body, biases, operators)
        arguments = (path, *rule_inputs(running, torch.float64))
        parameters = path.trained_parameters()
        taken = prediction_gradient(parameters, *arguments)

        step = 1e-6
        # `operators.K` is probed with the biases: the widened rule's whole
        # claim is that `K` descends the same objective, so the same finite
        # difference has to land on it.
        for name in (
            "biases.encode_hidden_bias",
            "biases.decode_output_bias",
            "operators.K",
        ):
            for cell, index in ((0, 0), (2, 1)):
                probe = parameters[name].clone()
                shifted = dict(parameters, **{name: probe})
                # `K` is `[cells, k, k]`, so the probe is a matrix entry there
                # and a vector entry on a bias.
                at = (cell, index, index) if probe.ndim == 3 else (cell, index)
                probe[at] += step
                up = prediction_error(shifted, *arguments)
                probe[at] -= 2 * step
                down = prediction_error(shifted, *arguments)
                # Relative rather than absolute since #433: the used operator
                # puts a spectral norm on the forward path, so evaluating the
                # objective carries the SVD's roundoff and the central
                # difference inherits it divided by `step`. Measured at ~2e-8
                # relative on `K` here, against an exact analytic gradient --
                # the old purely-linear path could afford `abs=1e-8` and this
                # one cannot. `rel=1e-6` is still four orders tighter than any
                # difference a wrong gradient would produce.
                assert (up - down).item() / (2 * step) == pytest.approx(
                    taken[name][at].item(), rel=1e-6
                )

    def test_the_gradient_is_finite_at_the_construction_point(self, running):
        """#433's one sharp edge, held rather than left to be rediscovered.

        `K = a.I` has every singular value equal, so `sigma_max` is not
        differentiable there, and the default `a = 1.0` puts it on the band's
        upper face as well. Autograd returns *a* subgradient at such a point
        rather than raising, and what matters for a run starting there is that
        it is **finite and non-zero** — a NaN would poison `K` on the first
        step and a zero would leave it unable to move at all. The spectrum
        splits under the first update and the objective is smooth from then on,
        which is why nothing downstream special-cases this.
        """
        gradients = PredictionRule(running).gradient()
        taken = gradients["operators.K"]
        assert torch.isfinite(taken).all()
        assert taken.abs().sum() > 0

    def test_the_whole_forward_path_carries_gradient(self, running):
        # The missing-gradient failure the transform is chosen to make loud:
        # a bias left out of the parameter dict, or an `argnums` naming the
        # wrong argument, shows up here as a group that never learns.
        gradients = PredictionRule(running).gradient()
        assert set(gradients) == TRAINED_NAMES
        for name, value in gradients.items():
            assert value.abs().sum() > 0, name


class TestTheBatchedGradientEqualsThePerCellLocalGradient:
    # The identity is exact arithmetic, so it is read in double: two float32
    # BLAS kernels -- a `[cells, ·]` matmul and a `[1, ·]` one -- differ in the
    # last bits, which is a fact about accumulation order and not about the
    # claim. The float32 case is checked below at the precision it can hold, so
    # the shipped rule is not taken on trust from the double one.
    #
    # **The disagreement is read against each group's own scale since #139**,
    # and the reason is magnitude rather than looseness. Measured on this
    # fixture, the worst disagreement *as a fraction of the group's largest
    # entry* is uniform across all four trained groups -- 3.1e-5, 3.5e-5 and
    # 4.5e-5 on the three bias vectors, 4.5e-5 on `K` (float64: 6.8e-14 to
    # 1.5e-13). The widened target is therefore no less exact than the old one.
    # What changed is scale: `K`'s gradient rows run ~4x the largest bias row,
    # so a flat elementwise `atol` calibrated on biases reads a proportionally
    # identical error as a failure, while a flat `rtol` would compare a
    # near-zero entry against itself and demand more of it than float32 can
    # give. Both are avoided by stating the claim the way it was measured: the
    # worst absolute slack is a small fraction of what the gradient's largest
    # entry is worth.
    @pytest.mark.parametrize(
        "dtype, tolerance", [(torch.float64, 1e-11), (torch.float32, 1e-3)]
    )
    def test_every_cell(self, running, dtype, tolerance):
        body, biases, operators = population(running, dtype)
        inputs = rule_inputs(running, dtype)
        batched = gradient_of(body, biases, operators, inputs)
        for cell in range(len(running.dome.predicting)):
            alone, alone_operators = one_cell(running, biases, operators, cell)
            local = gradient_of(
                body,
                alone,
                alone_operators,
                tuple(tensor[cell : cell + 1] for tensor in inputs),
            )
            for name, value in local.items():
                reference = batched[name][cell : cell + 1]
                slack = float((value - reference).abs().max())
                scale = float(reference.abs().max())
                assert slack <= tolerance * scale, (
                    f"{name}, cell {cell}: {slack:.3e} against a scale of "
                    f"{scale:.3e}"
                )

    def test_the_shipped_rule_is_that_batched_gradient(self, running):
        # The identity is only worth anything if `PredictionRule` is what takes it.
        body, biases, operators = population(running, torch.float32)
        expected = gradient_of(body, biases, operators, rule_inputs(running))
        taken = PredictionRule(running).gradient()
        for name, value in expected.items():
            assert torch.equal(taken[name], value), name

    def test_one_cells_gradient_does_not_move_when_another_cells_biases_do(
        self, running
    ):
        # The identity's teeth. If anything trainable spanned the batched
        # dimension the sum would give an average rather than a stack, and cell
        # 0's row would follow cell 1's own parameters. (#90 makes this standing, over
        # both rules and the whole adapting surface; here it is the batching
        # claim's own check.)
        before = PredictionRule(running).gradient()
        with torch.no_grad():
            running.biases.encode_hidden_bias[1].add_(1.0)
            running.biases.decode_output_bias[1].add_(1.0)
            running.operators.K[1].add_(0.1)
        after = PredictionRule(running).gradient()
        for name in TRAINED_NAMES:
            assert torch.equal(before[name][0], after[name][0]), name
        assert not torch.equal(
            before["biases.decode_output_bias"][1],
            after["biases.decode_output_bias"][1],
        )


class TestTheBodyReceivesNoGradient:
    def test_the_bodys_weights_receive_no_gradient(self, running):
        # Made as reachable as they could ever be first: the freeze is a
        # registration as a buffer, so a body whose weights require grad is the
        # nearest thing to the failure this excludes. Under ambient autograd
        # and a `.backward()` these would all be populated.
        for _, weight in running.body.named_buffers():
            weight.requires_grad_(True)
        before = {
            name: weight.detach().clone()
            for name, weight in running.body.named_buffers()
        }
        PredictionRule(running).step()
        for name, weight in running.body.named_buffers():
            assert weight.grad is None, name
            assert torch.equal(weight.detach(), before[name]), name

    def test_the_body_is_absent_from_what_is_differentiated(self, running):
        path = ForwardPath(running.body, running.biases, running.operators)
        assert set(path.trained_parameters()) == TRAINED_NAMES
        assert set(PredictionRule(running).gradient()) == TRAINED_NAMES

    def test_the_body_is_not_a_parameter_of_anything(self, running):
        # What makes the batched identity exact is that nothing trainable spans
        # the cell dimension. The body is shared, so this is the check that it
        # is also frozen.
        assert {name for name, _ in running.body.named_parameters()} == set()


class TestOneGlobalLearningRate:
    @pytest.mark.parametrize("learning_rate", [1e-3, 0.03, 0.5])
    def test_the_step_is_the_learning_rate_times_the_gradient(
        self, running, learning_rate
    ):
        # One scalar, the same for every cell and every one of the six bias
        # groups, and nothing else in the step. There is no optimiser here.
        rule = PredictionRule(running, learning_rate=learning_rate)
        before = biases_of(running)
        gradients = rule.step()
        for name, parameter in running.biases.named_parameters():
            expected = before[name] - learning_rate * gradients[f"biases.{name}"]
            assert torch.equal(parameter.detach(), expected), name

    def test_the_rule_carries_nothing_between_steps(self, running):
        # No momentum, no running average, nothing per-cell and nothing
        # per-edge: two rules stepped from the same state agree, and a second
        # step of one rule is the plain gradient of the state it now sees.
        first = PredictionRule(running, learning_rate=0.03)
        before = surface_of(running)
        first.step()
        gradients = first.step()
        after_two = biases_of(running)

        # The whole surface is restored, not only the biases: `K` moved too,
        # and a rule stepped from a half-restored state would differ for a
        # reason that has nothing to do with carried state.
        with torch.no_grad():
            for name, parameter in running.biases.named_parameters():
                parameter.copy_(before[name])
            running.operators.K.copy_(before["K"])
        second = PredictionRule(running, learning_rate=0.03)
        second.step()
        expected = second.step()
        for name in gradients:
            assert torch.allclose(gradients[name], expected[name], atol=1e-7)
        for name, parameter in running.biases.named_parameters():
            assert torch.allclose(parameter.detach(), after_two[name], atol=1e-7)

    def test_the_step_actually_moves_every_bias(self, running):
        # The formula above is satisfied by a zero gradient too.
        before = biases_of(running)
        PredictionRule(running).step()
        for name, parameter in running.biases.named_parameters():
            assert not torch.equal(parameter.detach(), before[name]), name

    def test_the_default_is_positive_and_small(self):
        assert 0 < DEFAULT_LEARNING_RATE < 1

    @pytest.mark.parametrize(
        "learning_rate", [0.0, -1e-3, float("nan"), float("inf"), float("-inf")]
    )
    def test_a_non_positive_learning_rate_is_refused(self, sheaf, learning_rate):
        # nan and inf among them: `learning_rate <= 0` admits both, and a nan
        # rate silently poisons every bias on the first step. The bound is
        # written as one comparison that nan and inf both fail, rather than as
        # a pair of tests for them.
        with pytest.raises(ValueError, match="global scalar"):
            PredictionRule(sheaf, learning_rate=learning_rate)


class TestTheRuleRefusesARecordThatNeverHappened:
    """A fresh Sheaf's prior chart and evidence are zeros, which are a
    well-formed record of nothing. Descending on them reports a gradient
    rather than a mistake, so the rule refuses a sheaf that has not ticked."""

    def test_a_step_before_the_first_tick_is_refused(self, sheaf):
        with pytest.raises(ValueError, match="needs a tick to learn from"):
            PredictionRule(sheaf).gradient()

    def test_the_wrong_order_is_refused_rather_than_silently_wrong(self, sheaf):
        # `rule.step(); agent.tick()` is the loop this exists to catch.
        with pytest.raises(ValueError, match="needs a tick to learn from"):
            PredictionRule(sheaf).step()

    def test_one_tick_is_enough(self, sheaf):
        sheaf.tick()
        PredictionRule(sheaf).gradient()


class TestTheParametersHandedToTheTransformAreACopy:
    def test_trained_parameters_do_not_share_storage_with_the_live_surface(
        self, running
    ):
        # `Tensor.detach` shares storage, so without the clone an in-place
        # write through this dict reaches the running adapting surface while
        # leaving a clean tape -- the leak class #90 exists to catch and the
        # tape assertion cannot see.
        path = PredictionRule(running).path
        handed = path.trained_parameters()
        live = dict(path.named_parameters())
        for name, value in handed.items():
            before = live[name].detach().clone()
            value.add_(1.0)
            assert torch.equal(live[name].detach(), before), name


class TestTheGuard:
    def test_the_rule_leaves_the_ticks_state_off_the_tape(self, running):
        PredictionRule(running).step()
        running.assert_no_tape()
        for _, parameter in running.biases.named_parameters():
            assert parameter.grad_fn is None and parameter.is_leaf

    def test_the_gradient_carries_no_tape(self, running):
        for value in PredictionRule(running).gradient().values():
            assert value.grad_fn is None and not value.requires_grad

    def test_the_biases_accumulate_nothing_on_dot_grad(self, running):
        # The transform returns gradients as a pytree. Anything landing on
        # `.grad` would mean an ambient `.backward()` had run somewhere, which
        # is the idiom this rule is written to avoid.
        PredictionRule(running).step()
        for _, parameter in running.biases.named_parameters():
            assert parameter.grad is None

    def test_the_step_needs_its_no_grad(self, running):
        # The mutation test #86 established the pattern for: `no_grad` is a
        # context manager rather than a decorator precisely so this can reach
        # it. With the guard removed the in-place descent on a leaf that
        # requires grad is refused outright -- loud, which is the point.
        with mock.patch("torch.no_grad", contextlib.nullcontext):
            with pytest.raises(RuntimeError, match="leaf Variable"):
                PredictionRule(running).step()

    @pytest.mark.parametrize("record", ["prior_charts", "prior_evidence", "stalks"])
    def test_the_rule_refuses_an_input_that_is_not_detached(self, running, record):
        setattr(running, record, getattr(running, record) * _live_scalar())
        with pytest.raises(AssertionError, match="autograd tape"):
            PredictionRule(running).gradient()

    @pytest.mark.parametrize("record", ["prior_charts", "prior_evidence"])
    def test_the_records_the_rule_reads_are_covered_by_the_guard(self, running, record):
        # The rule's inputs are only detached if something says so. They are
        # what the tick hands the learning phase, so they are part of what
        # `assert_no_tape` inspects.
        setattr(running, record, getattr(running, record) * _live_scalar())
        with pytest.raises(AssertionError, match=record):
            running.assert_no_tape()

    def test_an_unguarded_tick_really_does_tape_them(self, running):
        # The other side of it: with `no_grad` removed the records are on a
        # tape by the second inference phase -- the first one leaves the chart
        # and the stalk buffer taped, and the second reads them in.
        with mock.patch("torch.no_grad", contextlib.nullcontext):
            for _ in range(2):
                with contextlib.suppress(AssertionError):
                    running.inference_phase()
        assert running.prior_charts.grad_fn is not None
        assert running.prior_evidence.grad_fn is not None
