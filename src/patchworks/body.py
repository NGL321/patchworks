"""The shared frozen cell body: `encode`, `K`, `decode`.

One set of weights, shared by every predicting cell and frozen
(`docs/spec/01-cell-and-sheaf.md`, *The cell body*). The body never adapts.
All adaptation lives in the adapting surface — the per-cell biases and the
per-cell operators here, and the restriction maps, which are not part of the
body.

The forward path factors in three parts::

    encode:  chart (k) x node stalk (n) -> chart (k)     nonlinear, frozen
    K:       chart (k)                  -> chart (k)     linear, per-cell, learned
    decode:  chart (k)                  -> node stalk (n)  linear, frozen

`encode` fuses the persisted chart with new evidence; the cell's own operator
`K` alone advances the fused chart; `decode` reads the prediction back out onto
the node stalk.

**`encode` is the body's only nonlinearity** (ticket #138). `step` was a frozen
nonlinear map and is deleted; what advances a chart is now the cell's own dense
`K`, learned, `a·I` at construction and carrying no bias. `decode` linearised
with it and is **frozen as a gauge**: with a linear readout a cell's prediction
is `D K z`, and if both `D` and `K` were learned the factorisation would be
non-identifiable — `D K = (D M)(M^-1 K)` — so `K` could be rescaled freely and
compensated in `D`, and `sigma_max(K)` would not be a well-defined quantity to
constrain. Since a settable, bounded `sigma_max(K)` is the entire reason the
conversion was taken, freezing `decode` is what makes the knob exist. The same
move as ADR-0010 applied to the body instead of the sheaf; see
`docs/adr/0014-the-linear-readout-is-gauge-fixed.md`.

The commitments from *The body's construction* that survive are structural here
rather than configurable:

* **`encode`'s activation is piecewise-linear; ReLU is the instance.** The
  polytope partition a piecewise-linear map induces *is* the object the
  activation region, fold and fold margin vocabulary is made of. After the
  conversion that vocabulary has one map rather than three, and it is
  **retained as a description of `encode` and retired as a mechanism**: it no
  longer bounds `gamma` (ADR-0007, demoted) and no longer carries timescale,
  which now lives in `K`'s spectrum. Nothing new is built on it.
* **`encode`'s hidden width is its own minimum, `max{d_x + 1, d_y}`.** Written
  as a rule (:func:`hidden_width`) rather than a constant, because `n/k` and
  `k` are both rungs on the flex ladder. At `n = 32`, `k = 12` it gives 45.
* **One hidden layer.** A second is measured expensive (it halves the median
  fold margin) and lands a further geometric job on a bias vector already
  over-subscribed.

**Buffers are the frozen body; parameters are the adapting surface.** That is
the invariant the prediction rule's target is *defined* by rather than filtered
against: :meth:`patchworks.learning.ForwardPath.trained_parameters` returns
`named_parameters()` whole, so registering something as a parameter here **is**
what trains it (ticket #139).

**Execution is batched.** Every parameter that varies per cell carries a
`[cells, ...]` leading dimension, and the population's forward pass is a
handful of matmuls — `K` is `[cells, k, k]` and one `bmm`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "CHART_DIM",
    "DEFAULT_BIAS_VARIANCE",
    "DEFAULT_OPERATOR_SCALE",
    "DEFAULT_RHO_K",
    "DEFAULT_WEIGHT_VARIANCE",
    "NODE_STALK_DIM",
    "BodyShape",
    "CellBiases",
    "CellBody",
    "CellOperators",
    "hidden_width",
]

#: @type stipulated
#: @flexibility fixed and intended to stay fixed: absent from 01-cell-and-sheaf.md's Flex priority ladder, and a per-cell n confounds the private-dimension gradient
#: @warrant docs/spec/06-graph-topology.md, Dimensions
#: `n`, the node stalk dimension of a predicting cell.
#:
#: A module-level constant here rather than a field on
#: :class:`patchworks.graph.DomeSpec` (ticket #186). It is not one of the dome's
#: counts: it sizes the shared frozen body that both domains run, so a graph may
#: not pick its own — #128 fixed one `n` across both domains, and a field
#: defaulting to this constant would let a graph silently disagree with the body
#: it shares. `BodyShape` still takes `n` as an argument, because the body module
#: is shape-general and the tests exercise it at other widths; what is fixed is
#: the *dome's* choice.
#:
#: 32 is `06-graph-topology.md`'s *Dimensions*. #32's delay-embedding comparison
#: checked that it clears a bound; nothing swept it, so there is no re-runnable
#: criterion that would return a different number.
NODE_STALK_DIM = 32

#: @type stipulated
#: @flexibility rung 5, the last rung: may become a range or a gradient ACROSS THE GRAPH if uniformity fails (01-cell-and-sheaf.md, Flex priority). The across-DOMAINS axis is closed by #132 and is not a rung
#: @warrant docs/spec/06-graph-topology.md, Dimensions; docs/research/032-dimensioning-small-predictors.md (#172); docs/adr/0023-the-chart-is-not-a-koopman-lift.md
#: `k`, the chart dimension — the cell's private low-dimensional coordinates,
#: and the memory depth its operator advances.
#:
#: A module-level constant here for the same reason as
#: :data:`NODE_STALK_DIM`, and off `DomeSpec` with it (ticket #186).
#:
#: **`stipulated` for the same reason `NODE_STALK_DIM` is, and by the same
#: sentence of the same spec section.** `06-graph-topology.md`'s *Dimensions*
#: says 12. Three passes have checked that 12 *clears* a bound and none has
#: swept it: #145 and ADR-0023 established there is no `k_lift` to size against,
#: `032` found nothing in the literature speaks to `n`/`k`/`m` as a set, and
#: #132 measured the dome piece's correlation dimension at **1.43** — against
#: which `032`'s capacities leave `k = 12` about **4x**. A bound cleared with
#: margin is not a re-runnable criterion that would return a different number,
#: so this is not `selected`; and since the literature does not fix the value it
#: is not `literature` either.
#:
#: **The width with the least margin is not this one.** On #132's reading the
#: margins rank `n = 32` ~11x, `k = 12` ~4x, `m = 4` ~1.4x. A session looking
#: for the dimension that is actually tight wants `m`.
#:
#: **This entry was `@provisional 132` and the debt is discharged (#442).**
#: #132 asked whether the chart must be a **per-domain** number — what it
#: threatened was the global-constant *form*, never the value — and resolved
#: **no**: an L1 wedge cell's piece is a finite set with no box-counting
#: dimension to be per-domain about, so #128's one frozen dictionary stands and
#: one global `k` is the right kind of thing. Nothing open now waits on `k`;
#: memory is capped by `K`'s **shape**, not by `k` (#166, #167), so #357's
#: non-normality shortfall is not a debt this constant carries.
#:
#: **Rung 5 and #132's axis are different axes, and must not blur.**
#: `01-cell-and-sheaf.md`'s *Flex priority* ladder puts `k` last — it *"may become
#: a range or a gradient across the graph if uniformity fails"* — which is `k`
#: varying **across the graph**, per cell. #132 asked whether it varies **across
#: domains**, which the ladder never licensed and which is now closed. The
#: ordering is itself a reversal: `k` was formerly the first thing the spec would
#: flex and is now the last, because widening it weakens the low-dimensional
#: claim.
CHART_DIM = 12

#: @type selected
#: @flexibility unknown
#: @warrant #42's rig; docs/spec/05-timescales.md, What this requires elsewhere
#: `sigma_w^2`, the variance of the weight draw scaled by fan-in. A global,
#: shared, frozen quantity whose only job is containment — keeping the body's
#: realised contraction negative with margin — never spread
#: (`docs/spec/05-timescales.md`, *What this requires elsewhere*). 1.2 is the
#: operating point #42's rig measured the fold margins in
#: `01-cell-and-sheaf.md` at.
DEFAULT_WEIGHT_VARIANCE = 1.2

#: @type selected
#: @flexibility the weak knob of the two: three orders of magnitude barely move the regional spectra
#: @warrant docs/research/027-regional-jacobian-spectra.md
#: `sigma_b^2`, the variance of the iid bias draw. The rig's default, and the
#: weak knob of the two: sweeping it over three orders of magnitude barely
#: moves the regional spectra (`docs/research/027-regional-jacobian-spectra.md`).
DEFAULT_BIAS_VARIANCE = 0.5

#: @type stipulated
#: @flexibility unknown
#: @warrant docs/adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md
#: `rho_K`, the operator band's lower edge as a reciprocal:
#: `sigma_max(K) in [1/rho_K, 1]` (ticket #140,
#: `docs/adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md`). A single
#: number for the whole graph, deliberately mirroring ADR-0010's `rho = 2` in
#: :mod:`patchworks.restriction` — one global band, not one per level, because a
#: per-level gauge would be a second timescale mechanism competing with the
#: biases' and would make the per-hop transmission budget level-dependent.
#:
#: **The upper face is exactly 1 and needs no margin.** What the band forbids is
#: *amplification*, and `sigma_max(K) <= 1` forbids it exactly; a cell sitting
#: at 1 is non-expansive (`‖Kz‖ <= ‖z‖` always), not divergent. It is the
#: maximal-transmission end of every band that excludes amplification, which is
#: what the transmission budget wants. Note that it permits `rho(K) = 1` — a
#: mode that neither grows nor decays — and so is **not** the claim
#: `|lambda| < 1`.
DEFAULT_RHO_K = 2.0

#: @type stipulated
#: @flexibility superseded per body by a rule, patchworks.bias_selection.operator_scale_rule; this is the band ceiling, the rule's answer where no rig has run
#: @warrant docs/adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md
#: `a`, the scalar in `K = a·I` at construction.
#:
#: **This is a rule, not a number** (ticket #140), and the rule is
#: :func:`patchworks.bias_selection.operator_scale`: *the largest value in the
#: band for which the selection rig's `slow_cap` still admits the target `tau`
#: band* — a number the rig produces per body, exactly as `slow_cap` already is.
#: Read plainly: take the longest memory that still demonstrably forgets.
#:
#: The coupling is why it is not free. `bias_selection` places every cell's
#: timescale at construction from the realised recurrence, which after the
#: conversion is `K @ J_encode`; at `K = a·I` that is `a · J_encode`, so `a`
#: multiplies every cell's placed `tau`, and pushing it up pushes cells past
#: `slow_cap`. The two faces guard opposite failures: too small and the chart is
#: wiped every tick and the cell collapses toward its bias; too large and the
#: cell never forgets and stops settling at all.
#:
#: The value here is the band ceiling — the rule's answer where no rig has run,
#: and the right default because nothing rides on transmission at construction
#: while the construction-time go/no-go must be *valid*. See
#: `docs/spec/05-timescales.md` for what the rig returns on the default body.
DEFAULT_OPERATOR_SCALE = 1.0


def hidden_width(d_x: int, d_y: int) -> int:
    """The hidden width of a `d_x -> d_y` map: its own minimum, `max{d_x + 1, d_y}`.

    Park et al.'s exact floor for universal approximation by ReLU networks,
    below which Lu et al. find a phase transition rather than a degradation.
    The body is sized *at* the floor: wider bodies pay fold margin for nothing,
    and dropping below buys nothing back.

    After the Koopman conversion `encode` is the only map this applies to — `K`
    and `decode` are linear and have no hidden layer. At `n = 32`, `k = 12` it
    gives 45.
    """
    if d_x < 1 or d_y < 1:
        raise ValueError(f"map dimensions must be positive, got {d_x} -> {d_y}")
    return max(d_x + 1, d_y)


@dataclass(frozen=True)
class BodyShape:
    """The body's interface dimensions, and the width they derive.

    `n` is the node stalk dimension and `k` the chart dimension, both global
    constants identical for every predicting cell — and, since #128, identical
    across *domains* as well as across cells: one frozen dictionary forces one
    `n`, because `encode` is `R^k x R^n -> R^k`. `k < n` is the low-dimensional
    requirement — a shape invariant no training story may violate — and is
    checked here rather than assumed.

    At the proof of concept's `n = 32`, `k = 12`, `encode`'s hidden width is 45.
    """

    n: int
    k: int

    def __post_init__(self) -> None:
        if self.k < 1 or self.n < 1:
            raise ValueError(f"n and k must be positive, got n={self.n}, k={self.k}")
        if self.k >= self.n:
            raise ValueError(
                "k < n is fixed by construction (docs/spec/01-cell-and-sheaf.md, "
                f"The cell); got n={self.n}, k={self.k}"
            )

    @property
    def encode_width(self) -> int:
        """Hidden width of `encode`, `R^k x R^n -> R^k`. 45 at n=32, k=12.

        The body's only hidden width: `K` and `decode` are linear.
        """
        return hidden_width(self.k + self.n, self.k)


class CellBody(torch.nn.Module):
    """The frozen half of the forward path: nonlinear `encode`, linear `decode`.

    The weights are registered as buffers, not parameters: they are absent from
    :meth:`torch.nn.Module.parameters`, so no optimiser can reach them and the
    freeze is enforced by construction rather than by a convention someone has
    to remember. They still take part in autograd as constants, which is what
    the prediction rule needs — a local gradient step *through* the frozen
    forward path onto the biases and the operator.

    Initialisation is random and non-degenerate: iid Gaussian scaled by fan-in,
    the reservoir-computing precedent, which requires no corpus. Pretraining the
    body is a documented swap-in, not the baseline — load a state dict over
    these buffers and nothing else about the body changes.

    **`decode` is one weight and no hidden layer.** It is `D`, `[n, k]`, frozen
    for the identifiability reason in this module's docstring. Its per-cell
    output bias survives in :class:`CellBiases` and is the constant observable;
    ADR-0004's no-constant-term argument reaches `K` and not this one, because
    an affine `K` makes the *dynamics* affine — a drift compounding every tick —
    while a readout offset never accumulates.

    Every method takes the per-cell surface as an argument and evaluates the
    whole population at once. There is no per-cell entry point, because there is
    no per-cell body.
    """

    def __init__(
        self,
        shape: BodyShape,
        *,
        weight_variance: float = DEFAULT_WEIGHT_VARIANCE,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if weight_variance <= 0:
            raise ValueError(f"weight_variance must be positive, got {weight_variance}")
        self.shape = shape
        self.weight_variance = weight_variance
        self.register_buffer(
            "encode_hidden_weight",
            self._draw(
                (shape.encode_width, shape.k + shape.n),
                weight_variance,
                generator,
                device,
                dtype,
            ),
        )
        self.register_buffer(
            "encode_output_weight",
            self._draw(
                (shape.k, shape.encode_width), weight_variance, generator, device, dtype
            ),
        )
        self.register_buffer(
            "decode_weight",
            self._draw((shape.n, shape.k), weight_variance, generator, device, dtype),
        )
        # The denominators of :meth:`fold_margin`, computed once and never
        # again: `encode`'s hyperplane normals are the rows of a **frozen**
        # weight shared by every cell in the graph, so the whole per-cell part
        # of the margin is the pre-activation numerator. This is what makes the
        # live read affordable (ADR-0019) — a per-cell quantity whose expensive
        # half is one graph-wide constant.
        self.register_buffer("fold_gradient_norms", self._gradient_norms(), persistent=False)
        # A pretrained body is a documented swap-in, and a cached constant is
        # exactly the thing a swap-in leaves stale. Non-persistent so it is
        # never *loaded*, and recomputed after every load so it is never wrong:
        # the trap this would otherwise set is a margin read off weights the
        # body no longer has, which reports a perfectly plausible number.
        self.register_load_state_dict_post_hook(CellBody._refresh_gradient_norms)

    @staticmethod
    def _draw(
        size: tuple[int, int],
        variance: float,
        generator: torch.Generator | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        fan_in = size[1]
        weight = torch.empty(size, device=device, dtype=dtype)
        return weight.normal_(0.0, (variance / fan_in) ** 0.5, generator=generator)

    def encode_parts(
        self,
        chart: torch.Tensor,
        node_stalk: torch.Tensor,
        biases: CellBiases,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`encode`, as `(hidden pre-activation, output)` — batched over cells.

        Affine, ReLU, affine. The **pre-activation** is returned alongside the
        output because it is what the activation region and fold margin are read
        from (`docs/spec/05-timescales.md`), and a rig that measures those has to
        read them off *this* forward path rather than off a copy of it — a body
        swapped in under these buffers would otherwise leave the measurement
        behind, measuring a body that is not the one that runs.

        Since #138 this is the whole of the body's nonlinearity, so it is also
        the whole of what a fold margin can be read from.
        """
        self._check(chart, self.shape.k, "chart")
        self._check(node_stalk, self.shape.n, "node_stalk")
        if biases.shape != self.shape:
            raise ValueError(
                f"biases are for n={biases.shape.n}, k={biases.shape.k}; this body "
                f"is n={self.shape.n}, k={self.shape.k}"
            )
        if biases.cells != chart.shape[0]:
            # Refused rather than broadcast: one bias vector spread over many
            # cells is a population whose cells are all the same cell, which is
            # the one thing the per-cell surface exists to prevent.
            raise ValueError(
                f"{chart.shape[0]} cells passed to biases holding {biases.cells}"
            )
        x = torch.cat((chart, node_stalk), dim=-1)
        pre_activation = x @ self.encode_hidden_weight.T + biases.encode_hidden_bias
        output = (
            torch.relu(pre_activation) @ self.encode_output_weight.T
            + biases.encode_output_bias
        )
        return pre_activation, output

    def _gradient_norms(self) -> torch.Tensor:
        #: The **node stalk block alone**, not the full `R^(k+n)` row. Corrected
        #: by #206 on #195's finding. Reconciliation displaces the stalk and
        #: leaves the chart where it was, so the distance that matters is the one
        #: measured in the coordinates the displacement actually moves along; a
        #: denominator carrying the chart columns divides by a gradient partly
        #: perpendicular to any reachable motion and reports the margin tighter
        #: than it is. Measured 1.183x looser on the default dome — conservative
        #: in the direction it was wrong, so nothing read before this was unsafe.
        stalk_block = self.encode_hidden_weight[:, self.shape.k :]
        return torch.linalg.vector_norm(stalk_block, dim=-1).clamp(min=1e-12)

    @staticmethod
    def _refresh_gradient_norms(module: "CellBody", incompatible_keys: object) -> None:
        module.fold_gradient_norms = module._gradient_norms()

    def fold_margin(self, pre_activation: torch.Tensor) -> torch.Tensor:
        """`min_i |z_i| / ‖∇z_i‖` over `encode`'s folds, `[cells]`.

        Hanin & Rolnick's distance from the operating point to the nearest
        boundary of the activation region it sits in — measured **along the node
        stalk's `R^n`**, the coordinates reconciliation displaces, rather than
        across the whole of `encode`'s input space `R^k x R^n`. With one hidden
        layer the gradient of the `i`th pre-activation is that row of the hidden
        weight, so the denominator is that row's **stalk block**, frozen and
        shared, hence :attr:`fold_gradient_norms`.

        **Why the stalk block and not the row** (#206, on #195's finding). The
        margin is only ever compared against a reconciliation displacement, and
        that displacement moves the stalk while leaving the chart alone. A
        denominator built from the full row divides by a gradient partly
        perpendicular to any motion the comparison can see, which reports every
        margin tighter than it is — 1.183x on the default dome. Reading it in the
        subspace the displacement lives in is what makes the two sides of
        ADR-0007's inequality lengths in the same space.

        **One definition, two readers.** The construction sweep
        (:mod:`patchworks.bias_selection`) and the live read
        (:class:`patchworks.tick.FoldRead`) are the same measurement at two
        moments, which is the whole of ADR-0019's *construction nominates, the
        run decides*: a second implementation would let the two drift and the
        comparison between them is the point.

        Takes the pre-activation :meth:`encode_parts` already returned rather
        than recomputing it, so a caller inside the forward path pays for an
        `abs`, a divide and a `min` and nothing else.
        """
        self._check(pre_activation, self.shape.encode_width, "pre_activation")
        return (pre_activation.abs() / self.fold_gradient_norms).min(dim=-1).values

    def encode(
        self,
        chart: torch.Tensor,
        node_stalk: torch.Tensor,
        biases: CellBiases,
    ) -> torch.Tensor:
        """Fuse the persisted chart with the node stalk into a single chart.

        `chart` is `[cells, k]`, `node_stalk` is `[cells, n]`, the result is
        `[cells, k]`. Prior belief and new evidence arrive as one argument each
        and leave as one chart, which the cell's own `K` alone advances.
        """
        return self.encode_parts(chart, node_stalk, biases)[1]

    def decode(self, chart: torch.Tensor, biases: CellBiases) -> torch.Tensor:
        """Read the advanced chart back out as a predicted node stalk, `[cells, n]`.

        `D z + b`: one frozen linear map and the cell's own constant observable.
        """
        self._check(chart, self.shape.k, "chart")
        return chart @ self.decode_weight.T + biases.decode_output_bias

    def forward(
        self,
        chart: torch.Tensor,
        node_stalk: torch.Tensor,
        biases: CellBiases,
        operators: "CellOperators",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The whole population's inference-phase forward path, in one evaluation.

        Returns `(advanced_chart, predicted_node_stalk)`: the chart that
        persists into the next tick, and the prediction `decode` reads off it.
        """
        return self.advance(self.encode(chart, node_stalk, biases), biases, operators)

    def advance(
        self,
        fused_chart: torch.Tensor,
        biases: CellBiases,
        operators: "CellOperators",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`K` then `decode` on an already-fused chart.

        Split out from :meth:`forward` because the chart's own round trip —
        `d chart_{t+1} / d chart_t`, the quantity the selection rig measures —
        runs through `encode` and `K` only, and callers measuring it need the
        two halves separately.
        """
        advanced = operators.advance(fused_chart)
        return advanced, self.decode(advanced, biases)

    def _check(self, x: torch.Tensor, width: int, name: str) -> None:
        if x.ndim != 2 or x.shape[-1] != width:
            raise ValueError(f"{name} must be [cells, {width}], got {tuple(x.shape)}")

    def extra_repr(self) -> str:
        return (
            f"n={self.shape.n}, k={self.shape.k}, "
            f"encode_width={self.shape.encode_width}, frozen=True"
        )


class CellBiases(torch.nn.Module):
    """The per-cell biases: `encode`'s two, and `decode`'s constant observable.

    Three bias vectors, each carrying a `[cells, ...]` leading dimension. The
    Koopman conversion deleted the other three — `step`'s two with the map
    itself, and `decode`'s hidden bias with its hidden layer — so what remains
    is 89 numbers per cell where there were 146, before `K`'s 144 are added
    alongside in :class:`CellOperators`.

    Geometrically `encode`'s pair is still what the fold vocabulary describes:
    the weights fix the *directions* of every folding hyperplane and the biases
    fix where each fold sits. What that no longer is, since #138, is the
    architecture's instrument of cell individuality — `K`'s 144 learned
    parameters are a far more direct one, and the fold framing survives only as
    a description of `encode`.

    `decode_output_bias` is not geometry at all: it is the constant observable,
    the offset without which a cell's prediction is pinned to a subspace through
    the origin and any nonzero mean in its stalk is permanently unreachable
    error.

    Drawn iid here. `docs/spec/05-timescales.md` requires the spread over cells
    to be imposed by *selection* — draw candidates, measure the timescale each
    produces, keep a set covering the target band — which
    :mod:`patchworks.bias_selection` does without changing the shape of anything.
    """

    #: The three surviving vectors, as `(name, width rule)`. `step`'s pair went
    #: with the map and `decode`'s hidden bias with the hidden layer (#138).
    _WIDTHS = (
        ("encode_hidden_bias", lambda shape: shape.encode_width),
        ("encode_output_bias", lambda shape: shape.k),
        ("decode_output_bias", lambda shape: shape.n),
    )

    def __init__(
        self,
        shape: BodyShape,
        cells: int,
        *,
        bias_variance: float = DEFAULT_BIAS_VARIANCE,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if cells < 1:
            raise ValueError(f"cells must be positive, got {cells}")
        if bias_variance < 0:
            raise ValueError(f"bias_variance must be non-negative, got {bias_variance}")
        self.shape = shape
        self.cells = cells
        self.bias_variance = bias_variance
        for name, width in self._WIDTHS:
            self.register_parameter(
                name,
                torch.nn.Parameter(
                    self._draw(
                        (cells, width(shape)), bias_variance, generator, device, dtype
                    )
                ),
            )

    @staticmethod
    def _draw(
        size: tuple[int, int],
        variance: float,
        generator: torch.Generator | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        bias = torch.empty(size, device=device, dtype=dtype)
        return bias.normal_(0.0, variance**0.5, generator=generator)

    def subset(self, index: torch.Tensor) -> "CellBiases":
        """The same biases over the cells `index` names, in that order.

        What selection keeps: `docs/spec/05-timescales.md` draws candidate bias
        vectors, measures the timescale each produces, and keeps a set covering
        the target band. That kept set is this — a population of the same shape
        holding a subset of the rows, detached from the draw it came out of, so
        the discarded candidates are not carried into training on the graph.
        """
        _check_index(index)
        cells = int(index.numel())
        source = self.encode_hidden_bias
        # Keeping nothing is a real outcome -- a band no draw reached -- so an
        # empty subset is representable, even though drawing zero cells is not.
        # The draw this constructor makes is overwritten below, and it is made
        # on a private generator so that keeping a set of biases does not
        # advance the global RNG by an amount that depends on how many were
        # kept -- inside a construction whose contract is that it reproduces.
        kept = CellBiases(
            self.shape,
            max(cells, 1),
            bias_variance=self.bias_variance,
            generator=torch.Generator(device=source.device),
            device=source.device,
            dtype=source.dtype,
        )
        kept.cells = cells
        with torch.no_grad():
            for name, _ in self._WIDTHS:
                getattr(kept, name).data = getattr(self, name).data[index].clone()
        return kept

    def extra_repr(self) -> str:
        return f"cells={self.cells}, n={self.shape.n}, k={self.shape.k}"


class CellOperators(torch.nn.Module):
    """The per-cell operators `K`: the rest of the body's adapting surface.

    One dense `[k, k]` matrix per cell, learned, carrying **no bias**, and
    `a·I` at construction (ticket #138). A sibling of :class:`CellBiases`
    rather than a member of it, because that class builds its parameters by
    width and `K` does not have a width (#139).

    **Identity at construction rather than a random draw**, so that an untrained
    graph is *quiescent* rather than noisy: a cell that is not yet predicting
    anything should not be emitting. It is not a critical point of prediction
    error, so the widened rule has a gradient to descend from the first tick.

    **No bias**, and the distinction from `decode`'s is principled: an affine
    `K` makes the dynamics affine, a drift that compounds every tick, which is
    exactly the persistent offset ADR-0004 refuses to let a linear map launder
    away.

    **Dense.** Structure — real Schur, normal, low-rank — is a *named* fallback
    rather than a silent one, and it now has a trigger as well as a name: a
    projection that fights the gradient every step is the observable that calls
    it (#139), since Fan et al. find the right template for a stability
    constraint is a direct parameterisation rather than a penalty.
    """

    def __init__(
        self,
        shape: BodyShape,
        cells: int,
        *,
        scale: float = DEFAULT_OPERATOR_SCALE,
        rho_k: float = DEFAULT_RHO_K,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if cells < 1:
            raise ValueError(f"cells must be positive, got {cells}")
        if rho_k < 1.0:
            raise ValueError(f"the operator band needs rho_k >= 1, got {rho_k}")
        if not 1.0 / rho_k <= scale <= 1.0:
            raise ValueError(
                "a is the scalar in K = a.I and lives in the band [1/rho_k, 1] = "
                f"[{1.0 / rho_k:.4f}, 1] (docs/adr/"
                "0015-the-cell-operator-band-is-on-the-spectral-norm.md); "
                f"got {scale}"
            )
        self.shape = shape
        self.cells = cells
        self.scale = scale
        self.rho_k = rho_k
        eye = torch.eye(shape.k, device=device, dtype=dtype)
        self.register_parameter(
            "K", torch.nn.Parameter(scale * eye.expand(cells, shape.k, shape.k).clone())
        )

    def advance(self, chart: torch.Tensor) -> torch.Tensor:
        """Advance the fused chart one tick: `z(t) -> z_hat(t+1)`, `[cells, k]`.

        One batched matmul, and the whole of what replaced `step`. A linear map
        and a single evaluation — never an inner solve.
        """
        if chart.ndim != 2 or chart.shape[-1] != self.shape.k:
            raise ValueError(
                f"chart must be [cells, {self.shape.k}], got {tuple(chart.shape)}"
            )
        if chart.shape[0] != self.cells:
            raise ValueError(
                f"{chart.shape[0]} cells passed to operators holding {self.cells}"
            )
        return torch.bmm(self.K, chart.unsqueeze(-1)).squeeze(-1)

    @property
    def norms(self) -> torch.Tensor:
        """`[cells]`: each operator's spectral norm, `sigma_max(K)`.

        The **constrained** quantity (#140). `rho(K)` is the *reported* one — it
        is what timescale wants — but writing the band on the radius would leave
        `body` an unbounded factor in the transmission budget, since for a
        non-normal matrix `rho = 0.5` is compatible with `sigma_max = 50`, and a
        dense `K` trained on a temporal objective will find exactly that,
        because transient growth is *how* linear systems move content. Bounding
        the norm bounds the radius for free.
        """
        return torch.linalg.matrix_norm(self.K, ord=2)

    def radii(self) -> torch.Tensor:
        """`[cells]`: each operator's spectral radius, `rho(K)`.

        Reported, never constrained — the quantity timescale reads (#143) and
        the one #149 predicts contact cells will show small.
        """
        return torch.linalg.eigvals(self.K).abs().amax(dim=-1)

    def project(self) -> torch.Tensor:
        """Restore the band, in place. Returns which cells it moved.

        `sigma_max(K) in [1/rho_K, 1]`, by rescaling the whole operator — which
        moves its norm proportionally and so restores the band exactly, without
        an SVD reconstruction. ADR-0010's mechanism, deliberately **not**
        ADR-0010's norm: Frobenius is wanted on a restriction map because it
        leaves learned rank-deficiency available, and rank-deficiency is the
        failure on the body, where #138 put the whole of the cell's
        expressiveness in the `k`-dimensional chart. The Frobenius proxy is also
        loose exactly where it matters, by up to `sqrt(k)`.

        It is enforcement, not an objective: it runs after the step and outside
        the transform, has no gradient, and reads nothing the cell did not
        already own — a cell owns its own `K` outright and needs nothing from a
        neighbour to take its norm.

        **The return value is the observable, not a side effect.**
        :meth:`~patchworks.learning.PredictionRule.step` names it in those
        words — *a projection that binds every step is instead the observable
        that calls #138's named fallback from a dense `K` to a structured one* —
        and until #351 nothing could read it, because the mask was computed
        here and dropped. `[cells]` of `bool`, true where the operator was out
        of band and was rescaled. It is a **report**: the projection is
        enforcement either way, and nothing here branches on it.
        """
        with torch.no_grad():
            norms = self.norms.clamp(min=1e-12)
            target = norms.clamp(min=1.0 / self.rho_k, max=1.0)
            self.K.mul_((target / norms).view(-1, 1, 1))
            return target != norms

    def subset(self, index: torch.Tensor) -> "CellOperators":
        """The same operators over the cells `index` names, in that order.

        The counterpart of :meth:`CellBiases.subset`, so that what selection
        keeps is a whole per-cell surface rather than half of one.
        """
        _check_index(index)
        cells = int(index.numel())
        kept = CellOperators(
            self.shape,
            max(cells, 1),
            scale=self.scale,
            rho_k=self.rho_k,
            device=self.K.device,
            dtype=self.K.dtype,
        )
        kept.cells = cells
        with torch.no_grad():
            kept.K.data = self.K.data[index].clone()
        return kept

    def extra_repr(self) -> str:
        return (
            f"cells={self.cells}, k={self.shape.k}, a={self.scale}, "
            f"band=[{1.0 / self.rho_k:.4f}, 1]"
        )


def _check_index(index: torch.Tensor) -> None:
    """The shared guard both per-cell surfaces' `subset` takes.

    A bool mask is the natural thing to reach for and would silently give a
    population whose `cells` was the mask's length rather than its count.
    Refused with the fix named, since the caller has the mask and `nonzero` is
    the whole of the conversion.
    """
    if index.ndim != 1:
        raise ValueError(f"index must be [cells], got {tuple(index.shape)}")
    if index.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"index must name cells, not mask them; got dtype {index.dtype}. "
            "Pass index.nonzero(as_tuple=False).flatten() for a mask."
        )
