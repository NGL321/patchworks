"""The learning phase: the local learning rule, over the tick's detached state.

`docs/spec/07-local-learning-rule.md` fixes what is computed and
`docs/spec/09-the-build-stack.md`, *The locality guard*, fixes how it is
written. The rule is **two rules**, split by parameter group (ADR-0008); this
module holds the first of them.

**The bias rule** trains biases on **prediction error**: the difference between
what `decode` predicted last tick and the node stalk the cell reads in as
evidence this tick. Reconciliation edits that node stalk in between, so the
signal already carries whatever the neighbours' disagreement did to the cell's
belief — without the rule ever reading a neighbour, a disagreement, or an edge.
It is a closed backprop through the cell's own frozen forward path, `encode` /
`step` / `decode`, stopped at the per-cell biases the shared body doesn't own.

Four commitments are structural here rather than configurable.

* **It is a phase, not part of the tick.** The tick is a rollout and carries no
  tape (:mod:`patchworks.tick`); this runs afterwards, as a function of that
  cell's own parameters plus arrays the tick left behind.
* **What is detached is the state, not the quantity being descended on.** The
  rule *re-runs* the forward path so that prediction error is live in the
  biases, against a detached target. :attr:`~patchworks.tick.Sheaf.prediction`
  is a number carried over dead from the tick and has no gradient in anything;
  descending on it would be a bug that never announces itself.
* **Written as a function transform.** `torch.func`: the objective is a pure
  function whose parameters arrive as an explicit argument through
  `functional_call`, and the gradient is taken with `grad(·)` scoped by
  `argnums` to the biases alone. The reason is the **direction in which each
  idiom fails**. Under ambient autograd a deleted `.detach()` produces an
  *extra* gradient — the silent, flattering failure that makes the agent look
  better and the thesis look right. Under the transform, naming the wrong
  `argnums` produces a *missing* gradient: a cell that stops learning, which is
  loud and immediate.
* **This still batches, and no `vmap` is needed.** Every coupling term enters
  as a detached constant, so the population's local graphs compose into one
  batched graph with no cross-cell edges and the gradient of the sum over cells
  is exactly the per-cell local gradient. The identity holds because the
  batched items share no *trainable* parameter: the body is shared but
  **frozen**, and its weights are buffers rather than parameters, so the shared
  half of the path is structurally absent from what is differentiated rather
  than filtered out of it.

The one permitted global signal is :data:`DEFAULT_LEARNING_RATE`, a single
scalar mirroring reconciliation's `γ`. Nothing else broadcasts, and the rule
holds no per-cell or per-edge state of its own — the step is
`bias ← bias − η · ∇bias`, with nothing to carry between ticks.
"""

from __future__ import annotations

import torch
from torch.func import functional_call, grad

from .body import CellBiases, CellBody
from .tick import Sheaf

__all__ = [
    "DEFAULT_LEARNING_RATE",
    "BiasRule",
    "ForwardPath",
    "bias_gradient",
    "prediction_error",
]

#: `η`, the single global learning-rate scalar — one of exactly two permitted
#: global signals, and the only one this rule uses
#: (`docs/spec/07-local-learning-rule.md`, *Permitted global signals*). It is
#: schedule-shaped rather than information-shaped: it carries no cell's error
#: anywhere, which is what keeps it from being a global loss wearing a
#: different name.
#:
#: **The value is chosen here, not recorded.** The spec fixes that there is one
#: scalar and that it mirrors `γ`; it fixes no number, and unlike `γ` there is
#: no bound in the record that picks one out. 1e-2 is small enough that a step
#: is a drift in the body's operating point rather than a jump between
#: activation regions — the settling floor's failure mode
#: (`docs/spec/05-timescales.md`, and *Stability under simultaneous, cell-local
#: learning*) — and it is the thing to retune first once #90 and #91 can
#: measure what a run actually does.
DEFAULT_LEARNING_RATE = 1e-2


class ForwardPath(torch.nn.Module):
    """The cell's own frozen forward path, as one callable module.

    `functional_call` replaces a *module's* ambient parameters, so the thing it
    is called on has to **be** the forward path. The body deliberately takes
    its biases as an argument rather than owning them — that is what makes one
    shared body many different cells — so this holds the two together for the
    length of a gradient step and owns nothing itself.

    It is not state: build one per rule, or one per call, and it makes no
    difference to anything.
    """

    def __init__(self, body: CellBody, biases: CellBiases) -> None:
        super().__init__()
        self.body = body
        self.biases = biases

    def forward(self, chart: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        """`[cells, n]`: what `decode` predicts from that chart and that evidence.

        The advanced chart the body also returns is dropped. The bias rule's
        objective is about the prediction alone; the chart's own round trip is
        the fold margin's quantity, not this one.
        """
        _advanced, prediction = self.body(chart, evidence, self.biases)
        return prediction

    def bias_parameters(self) -> dict[str, torch.Tensor]:
        """The bias tensors, keyed as `functional_call` names them.

        Exactly the six bias vectors and nothing else, without a filter: the
        body's weights are registered as **buffers**, so the shared frozen half
        of the path cannot appear in a parameter dict at all. That is the same
        construction that makes the batched gradient equal the per-cell one.

        Detached, so what the transform hands back is a plain array rather than
        a node on the ambient tape. Nothing is lost by it: `grad` differentiates
        what `argnums` **names**, not what happens to carry `requires_grad`,
        which is the property this whole idiom is chosen for.
        """
        return {name: value.detach() for name, value in self.named_parameters()}


def prediction_error(
    bias_parameters: dict[str, torch.Tensor],
    path: ForwardPath,
    chart: torch.Tensor,
    evidence: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Half the squared prediction error of a whole population, as one scalar.

    A pure function. The biases arrive as ``bias_parameters`` — an explicit
    argument, not the module's ambient state — and everything else is a
    detached array the tick left behind: the chart the inference phase advanced
    from, the node stalk it read as evidence, and the node stalk reconciliation
    has since left in its place.

    The prediction is **recomputed** here rather than read off the tick, which
    is the whole point: it is what makes prediction error live in the biases.

    The objective is a plain **sum** over cells, and the sum is what batches.
    Each cell's term is a function of its own row of every argument, so the
    graph has no cross-cell edge in it and `∂/∂bias_c` of the sum is cell `c`'s
    own local gradient — exactly, not approximately.
    """
    prediction = functional_call(path, bias_parameters, (chart, evidence))
    return 0.5 * (prediction - target).pow(2).sum()


#: `∂ prediction_error / ∂ biases`, and nothing else.
#:
#: `argnums=0` names the bias dict alone. The path, the chart, the evidence and
#: the target are ordinary arguments of the function and are not differentiated
#: — not because they were severed, but because differentiation only ever
#: traverses what was named. Naming the wrong argument here produces a
#: **missing** gradient rather than an extra one, which is the failure
#: direction the transform was chosen for.
bias_gradient = grad(prediction_error, argnums=0)


class BiasRule:
    """The bias rule as a phase: one local gradient step per predicting cell.

    Built against a :class:`~patchworks.tick.Sheaf` and run after its tick.
    Holds a learning rate, a forward path, and no state whatsoever — no
    momentum, no running average, nothing per-cell and nothing per-edge. Two
    calls with the same sheaf state produce the same step.
    """

    def __init__(
        self, sheaf: Sheaf, *, learning_rate: float = DEFAULT_LEARNING_RATE
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(
                "the learning rate is a single positive global scalar "
                "(docs/spec/07-local-learning-rule.md, Permitted global signals); "
                f"got {learning_rate}"
            )
        self.sheaf = sheaf
        self.learning_rate = learning_rate
        self.path = ForwardPath(sheaf.body, sheaf.biases)

    def gradient(self) -> dict[str, torch.Tensor]:
        """The batched gradient of prediction error, keyed by bias name.

        Each value keeps the biases' own `[cells, ·]` leading dimension, and
        row `c` is cell `c`'s local gradient — which is why no `vmap` is
        needed. Taken without applying it, because #90's perturbation test
        wants to compare updates rather than trajectories.

        The tick's own guard is re-run on the way **in**, not only on the way
        out. "Over detached inputs" is a claim about what this phase reads, and
        a claim nothing checks at the boundary it is made about is a convention
        rather than a guarantee.
        """
        sheaf = self.sheaf
        sheaf.assert_no_tape()
        return bias_gradient(
            self.path.bias_parameters(),
            self.path,
            sheaf.prior_charts,
            sheaf.prior_evidence,
            sheaf.evidence(),
        )

    def step(self) -> dict[str, torch.Tensor]:
        """Take the gradient and descend it. Returns the gradient it applied.

        `bias ← bias − η · ∇bias`, under one global `η`. There is no optimiser
        here and nothing to give up by receiving gradients as a pytree instead
        of on `.grad`.

        The `no_grad` is a context manager rather than a decorator, for the
        reason :mod:`patchworks.tick` gives: a guard a test cannot remove is a
        guard nobody knows is load-bearing.
        """
        gradients = self.gradient()
        with torch.no_grad():
            for name, parameter in self.sheaf.biases.named_parameters():
                parameter.sub_(self.learning_rate * gradients[f"biases.{name}"])
        # The rule reads the tick's state and writes only the biases, so this
        # should be untouched -- which is exactly why it is cheap to say so.
        self.sheaf.assert_no_tape()
        return gradients
