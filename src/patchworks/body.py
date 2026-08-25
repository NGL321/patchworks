"""The shared frozen cell body: `encode`, `step`, `decode`.

One set of weights, shared by every predicting cell and frozen
(`docs/spec/01-cell-and-sheaf.md`, *The cell body*). The body never adapts.
All adaptation lives in the adapting surface — the per-cell biases here, and
the restriction maps, which are not part of the body.

The forward path factors in three parts::

    encode:  chart (k) x node stalk (n) -> chart (k)
    step:    chart (k)                  -> chart (k)
    decode:  chart (k)                  -> node stalk (n)

`encode` fuses the persisted chart with new evidence; `step` alone advances
the fused chart; `decode` reads the prediction back out onto the node stalk.

Three commitments from *The body's construction* are structural here rather
than configurable:

* **The activation is piecewise-linear; ReLU is the instance.** The polytope
  partition a piecewise-linear map induces *is* the object the timescale
  mechanism is made of — activation region, fold, regional spectrum, region
  dwell, fold margin. Under a smooth activation that mechanism loses its
  referent, so the activation is not a hyperparameter of this module.
* **Each map's hidden width is its own minimum, `max{d_x + 1, d_y}`.** Written
  as a rule (:func:`hidden_width`) rather than three constants, because `n/k`
  and `k` are both rungs on the flex ladder. At `n = 32`, `k = 12` it gives
  45 / 13 / 32.
* **One hidden layer per map.** A second layer is measured expensive (it halves
  the median fold margin) and lands a fourth geometric job on a bias vector
  already over-subscribed with three.

**Execution is batched.** Every parameter that varies per cell carries a
`[cells, ...]` leading dimension, and the population's forward pass is a
handful of matmuls rather than a loop over cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "DEFAULT_BIAS_VARIANCE",
    "DEFAULT_WEIGHT_VARIANCE",
    "BodyShape",
    "CellBiases",
    "CellBody",
    "hidden_width",
]

#: `sigma_w^2`, the variance of the weight draw scaled by fan-in. A global,
#: shared, frozen quantity whose only job is containment — keeping the body's
#: realised contraction negative with margin — never spread
#: (`docs/spec/05-timescales.md`, *What this requires elsewhere*). 1.2 is the
#: operating point #42's rig measured the fold margins in
#: `01-cell-and-sheaf.md` at.
DEFAULT_WEIGHT_VARIANCE = 1.2

#: `sigma_b^2`, the variance of the iid bias draw. The rig's default, and the
#: weak knob of the two: sweeping it over three orders of magnitude barely
#: moves the regional spectra (`docs/research/027-regional-jacobian-spectra.md`).
DEFAULT_BIAS_VARIANCE = 0.5


def hidden_width(d_x: int, d_y: int) -> int:
    """The hidden width of a `d_x -> d_y` map: its own minimum, `max{d_x + 1, d_y}`.

    Park et al.'s exact floor for universal approximation by ReLU networks,
    below which Lu et al. find a phase transition rather than a degradation.
    The body is sized *at* the floor: wider bodies pay fold margin for nothing,
    and dropping below buys nothing back.

    At `n = 32`, `k = 12` the rule gives 45 / 13 / **32**; the record briefly
    printed 33 for `decode`, an arithmetic slip corrected in ticket #84.
    """
    if d_x < 1 or d_y < 1:
        raise ValueError(f"map dimensions must be positive, got {d_x} -> {d_y}")
    return max(d_x + 1, d_y)


@dataclass(frozen=True)
class BodyShape:
    """The body's interface dimensions, and the widths they derive.

    `n` is the node stalk dimension and `k` the chart dimension, both global
    constants identical for every predicting cell. `k < n` is the
    low-dimensional requirement — a shape invariant no training story may
    violate — and is checked here rather than assumed.

    At the proof of concept's `n = 32`, `k = 12` the widths are 45 / 13 / 32.
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
        """Hidden width of `encode`, `R^k x R^n -> R^k`. 45 at n=32, k=12."""
        return hidden_width(self.k + self.n, self.k)

    @property
    def step_width(self) -> int:
        """Hidden width of `step`, `R^k -> R^k`. 13 at k=12."""
        return hidden_width(self.k, self.k)

    @property
    def decode_width(self) -> int:
        """Hidden width of `decode`, `R^k -> R^n`. 32 at n=32, k=12."""
        return hidden_width(self.k, self.n)


def _map_dimensions(shape: BodyShape) -> dict[str, tuple[int, int, int]]:
    """`(d_x, hidden, d_y)` for each of the three maps."""
    return {
        "encode": (shape.k + shape.n, shape.encode_width, shape.k),
        "step": (shape.k, shape.step_width, shape.k),
        "decode": (shape.k, shape.decode_width, shape.n),
    }


class CellBody(torch.nn.Module):
    """One set of weights, shared by every predicting cell and frozen.

    The weights are registered as buffers, not parameters: they are absent from
    :meth:`torch.nn.Module.parameters`, so no optimiser can reach them and the
    freeze is enforced by construction rather than by a convention someone has
    to remember. They still take part in autograd as constants, which is what
    the bias rule needs — a local gradient step *through* the frozen forward
    path onto the biases.

    Initialisation is random and non-degenerate: iid Gaussian scaled by fan-in,
    the reservoir-computing precedent, which requires no corpus. Pretraining the
    body is a documented swap-in, not the baseline — load a state dict over
    these buffers and nothing else about the body changes.

    Every method takes a :class:`CellBiases` and evaluates the whole population
    at once. There is no per-cell entry point, because there is no per-cell
    body.
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
        for name, (d_x, hidden, d_y) in _map_dimensions(shape).items():
            self.register_buffer(
                f"{name}_hidden_weight",
                self._draw((hidden, d_x), weight_variance, generator, device, dtype),
            )
            self.register_buffer(
                f"{name}_output_weight",
                self._draw((d_y, hidden), weight_variance, generator, device, dtype),
            )

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

    def apply_map(
        self,
        name: str,
        x: torch.Tensor,
        biases: CellBiases,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One map, as `(hidden pre-activation, output)` — batched over cells.

        Affine, ReLU, affine. The **pre-activation** is returned alongside the
        output because it is what the map's activation region and fold margin
        are read from (`docs/spec/05-timescales.md`), and a rig that measures
        those has to read them off *this* forward path rather than off a copy of
        it — a body swapped in under these buffers would otherwise leave the
        measurement behind, measuring a body that is not the one that runs.
        """
        if biases.shape != self.shape:
            raise ValueError(
                f"biases are for n={biases.shape.n}, k={biases.shape.k}; this body "
                f"is n={self.shape.n}, k={self.shape.k}"
            )
        if biases.cells != x.shape[0]:
            # Refused rather than broadcast: one bias vector spread over many
            # cells is a population whose cells are all the same cell, which is
            # the one thing the per-cell surface exists to prevent.
            raise ValueError(
                f"{x.shape[0]} cells passed to biases holding {biases.cells}"
            )
        hidden_weight = getattr(self, f"{name}_hidden_weight")
        output_weight = getattr(self, f"{name}_output_weight")
        hidden_bias, output_bias = biases.of(name)
        pre_activation = x @ hidden_weight.T + hidden_bias
        return pre_activation, torch.relu(pre_activation) @ output_weight.T + output_bias

    def encode(
        self,
        chart: torch.Tensor,
        node_stalk: torch.Tensor,
        biases: CellBiases,
    ) -> torch.Tensor:
        """Fuse the persisted chart with the node stalk into a single chart.

        `chart` is `[cells, k]`, `node_stalk` is `[cells, n]`, the result is
        `[cells, k]`. Prior belief and new evidence arrive as one argument each
        and leave as one chart, which `step` alone advances.
        """
        self._check(chart, self.shape.k, "chart")
        self._check(node_stalk, self.shape.n, "node_stalk")
        return self.apply_map("encode", torch.cat((chart, node_stalk), dim=-1), biases)[1]

    def step(self, chart: torch.Tensor, biases: CellBiases) -> torch.Tensor:
        """Advance the fused chart one tick: `z(t) -> z_hat(t+1)`, `[cells, k]`.

        A feed-forward map, a single forward pass — never an inner solve.
        """
        self._check(chart, self.shape.k, "chart")
        return self.apply_map("step", chart, biases)[1]

    def decode(self, chart: torch.Tensor, biases: CellBiases) -> torch.Tensor:
        """Read the advanced chart back out as a predicted node stalk, `[cells, n]`."""
        self._check(chart, self.shape.k, "chart")
        return self.apply_map("decode", chart, biases)[1]

    def forward(
        self,
        chart: torch.Tensor,
        node_stalk: torch.Tensor,
        biases: CellBiases,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The whole population's inference-phase forward path, in one evaluation.

        Returns `(advanced_chart, predicted_node_stalk)`: the chart that
        persists into the next tick, and the prediction `decode` reads off it.
        """
        return self.advance(self.encode(chart, node_stalk, biases), biases)

    def advance(
        self,
        fused_chart: torch.Tensor,
        biases: CellBiases,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`step` then `decode` on an already-fused chart.

        Split out from :meth:`forward` because the chart's own round trip —
        `d chart_{t+1} / d chart_t`, the quantity the fold margin is read from —
        runs through `encode` and `step` only, and callers measuring it need
        the two halves separately.
        """
        advanced = self.step(fused_chart, biases)
        return advanced, self.decode(advanced, biases)

    def _check(self, x: torch.Tensor, width: int, name: str) -> None:
        if x.ndim != 2 or x.shape[-1] != width:
            raise ValueError(
                f"{name} must be [cells, {width}], got {tuple(x.shape)}"
            )

    def extra_repr(self) -> str:
        return (
            f"n={self.shape.n}, k={self.shape.k}, "
            f"widths=({self.shape.encode_width}, {self.shape.step_width}, "
            f"{self.shape.decode_width}), frozen=True"
        )


class CellBiases(torch.nn.Module):
    """The per-cell biases: the body's whole trainable surface.

    Two bias vectors per map — one on the hidden layer, one on the output — each
    carrying a `[cells, ...]` leading dimension. Geometrically these are what
    make one shared body many different cells: the weights fix the *directions*
    of every folding hyperplane, the biases fix **where each fold sits**, so
    cells share one arrangement of folds up to translation.

    Drawn iid here. `docs/spec/05-timescales.md` requires the spread over cells
    to be imposed by *selection* — draw candidates, measure the timescale each
    produces, keep a set covering the target band — which replaces this draw
    without changing the shape of anything; a later ticket builds it.
    """

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
        for name, (_, hidden, d_y) in _map_dimensions(shape).items():
            for role, width in (("hidden", hidden), ("output", d_y)):
                self.register_parameter(
                    f"{name}_{role}_bias",
                    torch.nn.Parameter(
                        self._draw(
                            (cells, width), bias_variance, generator, device, dtype
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

    def of(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The `(hidden, output)` bias pair for one map, each `[cells, ...]`."""
        return (
            getattr(self, f"{name}_hidden_bias"),
            getattr(self, f"{name}_output_bias"),
        )

    def subset(self, index: torch.Tensor) -> "CellBiases":
        """The same biases over the cells `index` names, in that order.

        What selection keeps: `docs/spec/05-timescales.md` draws candidate bias
        vectors, measures the timescale each produces, and keeps a set covering
        the target band. That kept set is this — a population of the same shape
        holding a subset of the rows, detached from the draw it came out of, so
        the discarded candidates are not carried into training on the graph.
        """
        if index.ndim != 1:
            raise ValueError(f"index must be [cells], got {tuple(index.shape)}")
        if index.dtype not in (torch.int32, torch.int64):
            # A bool mask is the natural thing to reach for and would silently
            # give a population whose `cells` was the mask's length rather than
            # its count. Refused with the fix named, since the caller has the
            # mask and `nonzero` is the whole of the conversion.
            raise ValueError(
                f"index must name cells, not mask them; got dtype {index.dtype}. "
                "Pass index.nonzero(as_tuple=False).flatten() for a mask."
            )
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
            for name in _map_dimensions(self.shape):
                for role in ("hidden", "output"):
                    getattr(kept, f"{name}_{role}_bias").data = (
                        getattr(self, f"{name}_{role}_bias").data[index].clone()
                    )
        return kept

    def extra_repr(self) -> str:
        return f"cells={self.cells}, n={self.shape.n}, k={self.shape.k}"
