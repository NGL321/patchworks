"""The learning phase: the local learning rule, over the tick's detached state.

`docs/spec/07-local-learning-rule.md` fixes what is computed and
`docs/spec/09-the-build-stack.md`, *The locality guard*, fixes how it is
written. The rule is **two rules**, split by parameter group (ADR-0008), and
this module holds both.

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

import math

import torch
from torch.func import functional_call, grad

from .body import CellBiases, CellBody
from .restriction import RestrictionMaps
from .tick import Sheaf

__all__ = [
    "DEFAULT_ANNEAL_HORIZON",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_SPARSITY_PRESSURE",
    "MAPS_PARAMETER",
    "NORM_FLOOR",
    "BiasRule",
    "ForwardPath",
    "SparsityAnneal",
    "TransportPath",
    "TransportRule",
    "bias_gradient",
    "checked_learning_rate",
    "normalised_l1",
    "prediction_error",
    "relative_disagreement",
    "transport_gradient",
    "transport_objective",
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


def checked_learning_rate(learning_rate: float) -> float:
    """`η` back, or a refusal. Both rules run off the same one scalar.

    Written as a comparison rather than as `learning_rate <= 0`, so that `nan`
    and `inf` are refused by the same expression: `nan` fails every comparison,
    and a `nan` rate poisons the whole adapting surface on its first step
    without anything announcing it.
    """
    if not 0.0 < learning_rate < math.inf:
        raise ValueError(
            "the learning rate is a single positive global scalar "
            "(docs/spec/07-local-learning-rule.md, Permitted global signals); "
            f"got {learning_rate}"
        )
    return learning_rate


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

        Cloned as well as detached. `Tensor.detach` returns a tensor **sharing
        storage** with the original, so an in-place write through what this
        hands back would reach the live `nn.Parameter` -- the running adapting
        surface -- while leaving a perfectly clean tape. That is precisely the
        leak class #90's perturbation test exists to catch and the tape
        assertion cannot see, so this function does not manufacture one.
        """
        return {
            name: value.detach().clone() for name, value in self.named_parameters()
        }


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
        self.sheaf = sheaf
        self.learning_rate = checked_learning_rate(learning_rate)
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

        A sheaf that has never ticked is refused for the same reason. Its
        `prior_charts` and `prior_evidence` are zeros, which are a perfectly
        well-formed record of nothing -- so `rule.step(); agent.tick()`, or a
        step straight after `observe()`, would descend on a fabricated pair
        and report a gradient rather than a mistake.
        """
        sheaf = self.sheaf
        if sheaf.ticks == 0:
            raise ValueError(
                "the bias rule needs a tick to learn from: `prior_charts` and "
                "`prior_evidence` are still the zero placeholders a fresh Sheaf "
                "is built with, so descending on them would train against a "
                "record that never happened. Run a tick first -- note the order "
                "is `agent.tick(); rule.step()`, not the reverse"
            )
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


# -- the transport rule ----------------------------------------------------
#
# **The transport rule** trains restriction maps on **disagreement**, already
# computed during the message-passing phase and, per ADR-0007, never aimed at a
# zero target. Of that ADR's two permitted objectives it takes the **relative**
# one, and the choice is load-bearing rather than stylistic: learning on
# *change* in disagreement does not exclude the trivial solution, because
# shrinking the maps produces a negative change every step, which a
# change-descending rule reads as progress. Learning on disagreement relative to
# the restricted beliefs' own current magnitudes, `‖F_u x_u‖ + ‖F_v x_v‖`, does.
# The ratio is unchanged when both of an edge's maps scale together, so
# shrinking the edge buys nothing, and sending one map to zero sends the ratio
# to `1`, its worst value (ADR-0010).
#
# The normaliser is **locally stateless** and deliberately not a running average
# of the edge's recent scale: that would be a per-edge auxiliary variable with a
# hand-set time constant, the object ADR-0007 rejects under *A per-edge learned
# baseline*.
#
# **The one cross-cell parameter.** Disagreement on an edge is a function of
# *both* its maps and each belongs to a different cell's adapting surface, so a
# cell's transport objective contains a neighbour's *trainable* parameter, which
# a layer's loss in a feedforward network never does. That neighbour's map
# enters this rule exactly once, inside
# :attr:`~patchworks.tick.Sheaf.incoming` — the belief the neighbour restricted
# onto the shared edge stalk one tick ago, which is the *only* thing the rule
# ever learns about a neighbour. `07-local-learning-rule.md` is explicit that
# the rule never reads a neighbour's raw node stalk, "only the disagreement
# already derived from it during reconciliation", and `09-the-build-stack.md`
# names the phase's detached inputs as the cell's own chart, its own node stalk,
# and "the neighbour contribution to each of its incident edges". So the
# neighbour's map arrives already applied, as an ordinary argument of the
# objective, and is simply not among the `argnums` — not because it was severed,
# but because differentiation only ever traverses what was named. Every term of
# the summed objective is a function of one edge endpoint's own map row and
# nothing else, which is what makes the batched gradient the per-endpoint local
# gradient exactly.
#
# ADR-0010's gauge projection runs **after** the step and **outside** the
# transform: it is not part of the objective, it has no gradient, and it is as
# local as the rest of the rule, since a cell owns its own incident maps and
# needs nothing from a neighbour to project them.

#: The key `functional_call` names the padded map tensor by, inside a
#: :class:`TransportPath`. One tensor holds every edge endpoint's map, so the
#: whole population's transport rule is one parameter and one gradient.
MAPS_PARAMETER = "maps.maps"

#: The additive floor inside every norm this rule takes, in squared units.
#: `‖v‖ = √(vᵀv + floor)` rather than `√(vᵀv)`, because the plain norm's
#: gradient at `v = 0` is `0/0` and would put `nan` through the whole adapting
#: surface on the first all-zero edge — and one `nan` reaches every map in the
#: graph within a step, through the projection's rescale. At `1e-24` the floor
#: is `1e-12` in the norm itself, far below anything the maps or the stalks
#: carry, and it makes the gradient at the zero vector finite and `0` rather
#: than undefined.
#:
#: **That is all it does, and the narrow claim is deliberate.** It does not
#: give the rule a fixed point at agreement. The numerator is a norm and not a
#: squared norm — which is what ADR-0010's `[0, 1]` triangle-inequality reading
#: requires — so `‖∂J/∂F x‖ → 1/(‖F x‖ + ‖y‖)` as `F x → y` rather than to
#: zero: an endpoint that has converged still takes a full `η · ∇` step every
#: tick and the maps wander at amplitude `~η` instead of settling. That is the
#: objective the record specifies behaving as specified, and it sits with
#: ADR-0007 (the rule may never take zero disagreement as its target, and the
#: floor means it is never reached anyway) and ADR-0001 (the adapting surface
#: never freezes) rather than against them. It is recorded here because the
#: sentence this comment used to carry — "no step is taken where there is
#: nothing to learn from" — reads like a settling guarantee and is not one.
NORM_FLOOR = 1e-24

#: `λ_max`, the ceiling the sparsity pressure anneals up to — the second and
#: last permitted global signal alongside `η`
#: (`docs/spec/07-local-learning-rule.md`, *Permitted global signals*), and
#: schedule-shaped rather than information-shaped in exactly the same way.
#:
#: **The value is chosen here, not recorded.** `06-graph-topology.md` fixes that
#: sparsity annealing is "a schedule on the sparsity pressure, not a structural
#: process" and ADR-0010 fixes that the pressure is an L1 on the normalised map;
#: neither fixes a number. It is set by **measuring** the balance rather than by
#: arguing it: the relative disagreement lies in `[0, 1]`, and the pressure
#: term's gradient is `√(1 − h²)/‖F‖_F` (see :func:`normalised_l1`), so the two
#: are the same order but not the same size. Measured on the default dome at
#: `λ = 0.4`, the pressure term's per-map gradient sits at a median **0.12** of
#: the transport term's (0.114–0.127 across three seeds), which is what "prunes
#: *within* the mask" asks of a secondary pressure. It is a thing to retune once
#: #91 can read effective rank, and `tests/test_transport_rule.py` holds the
#: figure to the dome it names.
#:
#: The value moved from `0.03` to `0.4` when the `1/√p` normalisation went into
#: :func:`normalised_l1` (#89): the term shrank by roughly `√p`, and the ceiling
#: rose by roughly the same factor to hold the balance where it was.
DEFAULT_SPARSITY_PRESSURE = 0.4

#: How many steps the pressure takes to reach that ceiling.
#:
#: **Chosen here, not recorded**, and the *direction* is the choice: the
#: pressure anneals **up**, from nothing, rather than down. A map is drawn
#: random and dense, and until transport has organised it there is no shape for
#: an L1 to prune *within* — pruning a map before it carries anything is the
#: local-neuroplasticity analogue run backwards. **That direction was escalated
#: from #89 and ruled on rather than left to default**, so it is settled and not
#: merely a first guess; the horizon and the ramp's shape were not, and remain
#: this module's. The horizon is set an
#: order of magnitude above the slowest cell's own time constant
#: (`docs/spec/05-timescales.md` reaches `τ ≥ 100 ticks`), so that on the
#: timescale any one cell adapts over the pressure is a constant rather than
#: something it is chasing.
DEFAULT_ANNEAL_HORIZON = 1000


class SparsityAnneal:
    """The schedule on the sparsity pressure: one global scalar, per step.

    A pure schedule — a linear ramp from zero to :attr:`pressure` over
    :attr:`horizon` steps, flat thereafter. It reads nothing about any cell,
    any edge, or any disagreement, which is what keeps the second permitted
    global signal schedule-shaped rather than information-shaped. It holds no
    state either: the step index is handed in.
    """

    def __init__(
        self,
        *,
        pressure: float = DEFAULT_SPARSITY_PRESSURE,
        horizon: int = DEFAULT_ANNEAL_HORIZON,
    ) -> None:
        if not 0.0 <= pressure < math.inf:
            raise ValueError(
                "the sparsity pressure is a single non-negative global scalar "
                "(docs/spec/07-local-learning-rule.md, Permitted global signals); "
                f"got {pressure}"
            )
        # Written as a two-sided comparison for the reason
        # `checked_learning_rate` is: `horizon < 1` admits `nan` and `inf`, and
        # both are silently wrong rather than loud. A `nan` horizon puts
        # `min(1.0, nan)` at `1.0`, so the full ceiling applies from step zero
        # -- the anneal run backwards -- and an infinite one holds the pressure
        # at zero forever, switching the second global signal off.
        if not 1 <= horizon < math.inf:
            raise ValueError(
                f"the anneal horizon is a positive step count, got {horizon}"
            )
        self.pressure = pressure
        self.horizon = horizon

    def at(self, step: int) -> float:
        """`λ(step)`: the pressure this step's objective composes.

        Zero at the first step and the full ceiling from :attr:`horizon`
        onwards. The bound is two-sided for the reason the horizon's is:
        `step < 0` admits `nan`, and `min(1.0, nan)` is `1.0`, so a `nan`
        position would report the full pressure rather than a mistake.
        """
        if not 0 <= step < math.inf:
            raise ValueError(f"the schedule starts at step 0, got {step}")
        return self.pressure * min(1.0, step / self.horizon)

    def __repr__(self) -> str:
        return f"SparsityAnneal(pressure={self.pressure}, horizon={self.horizon})"


class TransportPath(torch.nn.Module):
    """The restriction of every cell's own node stalk, as one callable module.

    The counterpart to :class:`ForwardPath`, and it exists for the same reason:
    `functional_call` replaces a *module's* ambient parameters, so the thing it
    is called on has to **be** what the objective runs. It owns nothing.
    """

    def __init__(self, maps: RestrictionMaps) -> None:
        super().__init__()
        self.maps = maps

    def forward(self, gathered: torch.Tensor) -> torch.Tensor:
        """`[pairs, m_max]`: what each cell restricts onto each incident edge.

        `gathered` is `[pairs, stalk_max]` — each edge endpoint's **owning
        cell's** node stalk, padded. No row of it is a neighbour's stalk, which
        is what makes the rule's locality a property of the argument rather
        than of the arithmetic.
        """
        return self.maps.restrict(gathered)

    def map_parameters(self) -> dict[str, torch.Tensor]:
        """The padded map tensor, keyed as `functional_call` names it.

        Detached and **cloned**, for the reason
        :meth:`ForwardPath.bias_parameters` gives: `Tensor.detach` shares
        storage, so without the clone an in-place write through this dict would
        reach the running adapting surface while leaving a perfectly clean tape.
        """
        return {
            name: value.detach().clone() for name, value in self.named_parameters()
        }


def _norm(values: torch.Tensor) -> torch.Tensor:
    """`√(vᵀv + NORM_FLOOR)` along the last dimension. See :data:`NORM_FLOOR`."""
    return (values.pow(2).sum(-1) + NORM_FLOOR).sqrt()


def relative_disagreement(
    outgoing: torch.Tensor, neighbour_beliefs: torch.Tensor
) -> torch.Tensor:
    """`[pairs]`: `‖F_v x_v − y_e‖ / (‖F_v x_v‖ + ‖y_e‖)`, per edge endpoint.

    ADR-0007's **relative** objective, in the locally stateless form ADR-0010
    commits the spec to: disagreement over the restricted beliefs' own current
    magnitudes, and nothing that remembers what the edge's scale used to be.

    Two properties are the whole reason it is this quantity rather than the
    change in disagreement, and both are tested:

    * **Scaling both of an edge's maps together leaves it unchanged.**
      Numerator and denominator are homogeneous of the same degree, so
      shrinking an edge buys nothing and the trivial solution is not on the
      way down.
    * **Sending either map to zero sends it to `1`.** By the triangle
      inequality the ratio lies in `[0, 1]`, and one-sided collapse is the
      worst value the rule can reach rather than something it can fall into.

    ``neighbour_beliefs`` is `y_e`, the belief the neighbour restricted onto
    the shared edge stalk one tick ago. It is a plain array the tick left
    behind: the neighbour's map is in it, applied and dead.

    **The maximum is flat, and where it is reachable that matters.** ADR-0010
    reads `1` as a value the rule is pushed away from, and from the cell's own
    map that is right: shrinking `F_v` alone raises the ratio. But the ratio
    is `1` *identically* — not merely at a point — wherever the two beliefs
    are collinear and opposed, and wherever one of them is zero, so the
    gradient there is exactly zero and the endpoint does not learn. Three
    forms of it are reachable and each is tested:

    * **A silent neighbour.** `y_e = 0` gives `‖F_v x_v‖ / ‖F_v x_v‖ = 1` for
      every `F_v`. This is not a defect: a cell whose neighbour has said
      nothing has no evidence about what basis to transport into, and a flat
      objective is the honest response to it. It is reachable in the shipped
      world rather than a corner case — a quiet touch sensor reads all zeros,
      so its whole stalk and every belief it broadcasts are zero — and it is
      the per-edge form of exactly what :meth:`TransportRule.gradient`'s
      two-tick guard refuses graph-wide.
    * **A one-dimensional edge stalk whose ends disagree in sign**, where
      collinear-and-opposed is half of the possible configurations rather than
      a measure-zero coincidence. Reconciliation moves the node stalk every
      tick, so the endpoint leaves as soon as the signs agree.
    * **A map whose mask leaves one weight open**, where the gauge already
      fixes the map's only degree of freedom.

    None of the three is a trivial solution the rule can descend *into* — the
    objective is at its worst value in all of them, and nothing carries an
    endpoint there. They are places it can sit while the world is quiet.
    """
    return _norm(outgoing - neighbour_beliefs) / (
        _norm(outgoing) + _norm(neighbour_beliefs)
    )


def normalised_l1(maps: torch.Tensor, permitted: torch.Tensor) -> torch.Tensor:
    """`[pairs]`: `‖F‖₁ / (√p ‖F‖_F)`, the sparsity pressure's per-map term.

    An L1 on the **normalised** map, so the pressure redistributes weight
    across a map's directions rather than removing it
    (`docs/spec/06-graph-topology.md`, *Sparsity is a property of the maps, not
    of the graph*). It is blind to a map's overall magnitude, exactly as the
    disagreement term is — which is the whole argument for the magnitude being
    gauge-fixed rather than learned.

    Zeroed entries — masked or padded — contribute nothing to either norm, so
    the quantity is over what the mask permits without a second mask being
    applied here.

    **The `1/√p` is what makes one global `λ` mean the same thing on every
    map**, where `p` is how many weights that map's structural mask leaves open
    (ADR-0010, amended in #89). Without it the term's gradient has norm
    `√(p − ‖F‖₁²/‖F‖_F²)/‖F‖_F`, which grows with the mask, so a single global
    scalar prunes a wide map harder than a narrow one — measured at `+0.985`
    correlation with `p` across the real dome, an eightfold spread. With it the
    gradient is::

        ‖∇(‖F‖₁ / (√p ‖F‖_F))‖  =  √(1 − h²) / ‖F‖_F,   h the value above

    and **`p` is gone from it identically**, not approximately: correlation with
    `p` falls to `+0.071`. What is left, `√(1 − h²)`, is the same function of a
    map's own normalised sparsity for every map at any size. `p` survives only
    in `h`'s own floor of `1/√p` — a fully concentrated map — so the *attainable
    ceiling* still varies by `√(1 − 1/8) / √(1 − 1/384) = 6.8%` across this
    dome's mask sizes, at an extreme the maps do not occupy.

    That line is the identity **without** :data:`NORM_FLOOR`, which is the form
    ADR-0010 argues from and is exact only as `‖F‖_F` stays clear of the floor.
    Since :func:`_norm` gives `n = √(FᵀF + NORM_FLOOR)` rather than `‖F‖_F`, the
    gradient this function actually has is `√(1 − h²(1 + NORM_FLOOR/n²)) / n`
    (#115) — a relative `NORM_FLOOR/(2‖F‖_F²(1 − h²))`, so `1/‖F‖_F²` and not
    `1/‖F‖_F`. It reaches `1e-9` around `‖F‖_F ≈ 4e-8` and a factor of two by
    `1e-12`. **Anywhere in the gauge band the two forms agree to `1e-24`**, so
    nothing above reads differently for a map the projection has touched; the
    correction is what an analytic reference has to carry if it is checked far
    below the band, as `tests/test_transport_rule.py` does. `p` is absent from
    the corrected form too, which is why the claim above survives it intact.

    The quantity `h` itself is Hoyer's sparseness ratio, normalised for exactly
    this reason: to be comparable across dimensions. It runs `1/√p` for a map
    on one direction to `1` for a flat one, so **smaller is sparser** and the
    pressure descends it. Dividing by `√p` is a construction-time constant per
    map and changes nothing *within* one — the pruning `06-graph-topology.md`
    asks for is untouched — only the weight between maps of different sizes.

    ``permitted`` is `[pairs]`: the mask's open-weight count, read off the
    structural mask at construction. It is **not** per-edge state — nothing
    updates it, nothing learns it, and it moves only if the graph does, exactly
    like the `Σ_e m_e` the reconciliation gain divides by.
    """
    flat = maps.flatten(1)
    return flat.abs().sum(-1) / (_norm(flat) * permitted.sqrt())


def transport_objective(
    map_parameters: dict[str, torch.Tensor],
    path: TransportPath,
    gathered: torch.Tensor,
    neighbour_beliefs: torch.Tensor,
    permitted: torch.Tensor,
    pressure: float,
) -> torch.Tensor:
    """The whole graph's transport objective, as one scalar.

    A pure function. The maps arrive as ``map_parameters`` — an explicit
    argument, not the module's ambient state — and everything else is a
    detached array the tick left behind, plus the one global scalar the anneal
    schedule supplies.

    Disagreement is **recomputed** here rather than read off the tick, which is
    the point: it is what makes it live in that cell's own maps.
    :meth:`~patchworks.tick.Sheaf.disagreement` is a number carried over dead
    and has no gradient in anything.

    The sparsity pressure composes as **one additive term inside this one
    descent step**, not as a second update loop running alongside it.

    The objective is a plain **sum** over edge endpoints, and the sum is what
    batches: each term is a function of that endpoint's own row of the map
    tensor and of its own row of the two arrays, so the graph has no cross-cell
    edge in it and `∂/∂F_i` of the sum is endpoint `i`'s own local gradient —
    exactly, not approximately.
    """
    outgoing = functional_call(path, map_parameters, (gathered,))
    disagreement = relative_disagreement(outgoing, neighbour_beliefs).sum()
    penalty = normalised_l1(map_parameters[MAPS_PARAMETER], permitted).sum()
    return disagreement + pressure * penalty


#: `∂ transport_objective / ∂ maps`, and nothing else.
#:
#: `argnums=0` names the map tensor alone. The path, the gathered node stalks,
#: the neighbour beliefs, the mask's open-weight counts and the pressure are
#: ordinary arguments and are not differentiated. The neighbour's map is inside ``neighbour_beliefs``, already
#: applied, so the one cross-cell parameter in the phase is not reachable from
#: here at all.
transport_gradient = grad(transport_objective, argnums=0)


class TransportRule:
    """The transport rule as a phase: one local gradient step per edge endpoint.

    Built against a :class:`~patchworks.tick.Sheaf` and run after its tick.
    Holds a learning rate, an anneal, a path, and **one integer** — the
    position on the global anneal schedule, which is the second permitted
    global signal and is one number for the whole graph. There is nothing
    per-cell and nothing per-edge that *changes*: no momentum, no running
    average, no baseline and no estimate of any edge's recent scale. The one
    per-edge array it holds, :attr:`permitted`, is the structural mask's own
    open-weight count, fixed at construction and never written.
    """

    def __init__(
        self,
        sheaf: Sheaf,
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        anneal: SparsityAnneal | None = None,
    ) -> None:
        self.sheaf = sheaf
        self.learning_rate = checked_learning_rate(learning_rate)
        self.anneal = anneal if anneal is not None else SparsityAnneal()
        self.path = TransportPath(sheaf.maps)
        #: `[pairs]`: how many weights each map's structural mask leaves open,
        #: which the sparsity term divides by the root of
        #: (:func:`normalised_l1`). Read off the mask once, because the mask is
        #: set at construction and closes permanently — this is the same kind
        #: of object as the `Σ_e m_e` in the reconciliation gain's denominator,
        #: a structural constant of the built graph, and **not** per-edge state.
        self.permitted = sheaf.maps.support.flatten(1).sum(-1).to(
            sheaf.maps.maps.dtype
        )
        #: How many steps this rule has taken — the schedule's position, and
        #: the only thing it carries between steps.
        self.steps = 0

    @property
    def pressure(self) -> float:
        """`λ`, the sparsity pressure this step's objective will compose."""
        return self.anneal.at(self.steps)

    def inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The two detached arrays the objective reads, in pair order.

        Each cell's **own** node stalk as the tick left it, gathered onto every
        endpoint it owns, and the neighbour contribution to each of those
        endpoints. No neighbour's raw node stalk is among them: what a cell
        learns about its neighbour is the belief the neighbour already
        restricted onto the shared edge stalk, which is the only form in which
        the two cells' features are comparable at all.

        **Which node stalk, and why it is the reconciled one.**
        `07-local-learning-rule.md` calls the signal disagreement "already
        computed during the message-passing phase", and that phase restricted
        the stalk as it stood *before* reconciliation edited it;
        `09-the-build-stack.md` §2 names the phase's inputs as "the tick's state
        **as the tick left it** — its chart, its node stalk, the neighbour
        contribution to each of its incident edges". This follows `09`, and the
        choice is forced as well as right: the sheaf keeps no whole-population
        pre-reconciliation stalk — :attr:`~patchworks.tick.Sheaf.prediction`
        covers predicting cells only, and a boundary cell has none — and the
        quantity worth training on is the disagreement reconciliation could
        *not* clear rather than the one it just did.
        """
        sheaf = self.sheaf
        return sheaf.stalks[sheaf.layout.pair_positions], sheaf.incoming

    def gradient(self) -> torch.Tensor:
        """`[pairs, m_max, stalk_max]`: the batched gradient of the objective.

        Row `i` is edge endpoint `i`'s own local gradient — which is why no
        `vmap` is needed. Taken without applying it, because #90's perturbation
        test wants to compare updates rather than trajectories.

        The tick's own guard is re-run on the way **in** for the reason
        :meth:`BiasRule.gradient` gives, and a sheaf without a neighbour belief
        to learn from is refused for the same reason: `incoming` would be zeros,
        which are a perfectly well-formed record of nothing, so the step would
        descend on an edge that was never told anything and report a gradient
        rather than a mistake.

        **That takes two ticks, not one**, and the asymmetry with the bias rule
        is the unit delay rather than an off-by-one. The message-passing phase
        reads the broadcast buffer as it stood *before* the phase, so the first
        tick reconciles against the constructor's zeros and leaves `incoming`
        zero; the first tick a neighbour has actually spoken on is the second.
        The bias rule needs one tick because prediction error is a cell's own
        quantity and crosses no edge.
        """
        sheaf = self.sheaf
        if sheaf.ticks < 2:
            raise ValueError(
                "the transport rule needs two ticks to learn from: an edge "
                f"carries a unit delay, so after {sheaf.ticks} tick(s) "
                "`incoming` is still the zero placeholder a fresh Sheaf is "
                "built with, and descending on it would train against a "
                "neighbour that never spoke. Run a second tick first -- note "
                "the order is `agent.tick(); rule.step()`, not the reverse"
            )
        sheaf.assert_no_tape()
        return transport_gradient(
            self.path.map_parameters(),
            self.path,
            *self.inputs(),
            self.permitted,
            self.pressure,
        )[MAPS_PARAMETER]

    def step(self) -> torch.Tensor:
        """Take the gradient, descend it, then project. Returns the gradient.

        `F ← Π(F − η · ∇F)`, under one global `η` and one global `λ`, where `Π`
        is ADR-0010's gauge projection together with the structural mask
        (:meth:`~patchworks.restriction.RestrictionMaps.project`). The
        projection runs **after** the step and **outside** the transform: it is
        enforcement, not an objective, and nothing differentiates through it.

        It is **not inert**. An edge's joint scale grows monotonically under a
        scale-invariant objective, so the upper face binds essentially every
        step once a map reaches `ρ` — which is also why a map's norm is not a
        diagnostic.

        The `no_grad` is a context manager rather than a decorator, for the
        reason :mod:`patchworks.tick` gives: a guard a test cannot remove is a
        guard nobody knows is load-bearing.
        """
        gradient = self.gradient()
        with torch.no_grad():
            self.sheaf.maps.maps.sub_(self.learning_rate * gradient)
        self.sheaf.maps.project()
        self.steps += 1
        # The rule reads the tick's state and writes only the maps, so this
        # should be untouched -- which is exactly why it is cheap to say so.
        self.sheaf.assert_no_tape()
        return gradient
