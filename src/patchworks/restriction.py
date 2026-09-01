"""The restriction maps: one masked linear map per edge endpoint, gauge-fixed.

Each cell holds one map from its node stalk into each incident edge stalk
(`docs/spec/01-cell-and-sheaf.md`, *Restriction maps*). The two ends of an edge
are independent maps belonging to different cells, so the object here is
indexed by **edge endpoint**, never by edge: pair `2 * edge.id + side`, side 0
for `edge.u` and side 1 for `edge.v`. The partner of pair `i` is `i ^ 1`, which
is what makes the delay buffer in :mod:`patchworks.tick` an index flip rather
than a lookup.

Three properties are structural rather than learned:

* **Masked.** The structural mask names which node stalk directions may
  participate on that edge. It is set at construction, it closes and never
  re-opens, and :meth:`RestrictionMaps.project` re-applies it after every
  transport step so a learning rule cannot re-open it by accident.
* **Linear.** All nonlinearity lives inside the cell. Under a nonlinear map the
  identification of prediction error with the sheaf coboundary stops being true
  rather than merely becoming expensive (ADR-0004).
* **Gauge-fixed.** A map's overall magnitude is not identified by the transport
  rule's objective and grows without bound if left free, so it is fixed rather
  than learned (ADR-0010): interior maps carry the band `‖F‖_F ∈ [1/ρ, ρ]` with
  `ρ = 2`, and a **boundary cell's own** maps carry the exact gauge `‖F‖_F = 1`,
  because a boundary cell runs no body and has no metric individuality to
  protect. The norm is Frobenius, not spectral.

**The population is one padded tensor.** `m` varies across edges and boundary
cells are exempt from `n`, so the maps are ragged; they are stored as one
`[pairs, m_max, stalk_max]` tensor whose out-of-range entries are structurally
zero, exactly like masked-out ones. Restriction is then a single `bmm` over
every edge endpoint in the graph rather than 1364 small matmuls, which is what
keeps the message-passing phase inside the tick's budget. The padding costs
arithmetic on zeros and nothing else: a zero row contributes zero to a
Frobenius norm, zero to a restricted belief, and zero back to a node stalk.
"""

from __future__ import annotations

import torch

from .graph import Dome

__all__ = ["GAUGE_C", "GAUGE_RHO", "INITIAL_NORM", "RestrictionMaps", "pair_index"]

#: @type stipulated
#: @flexibility measured: 2 -> 16 buys 1.008x on the apex floor (#150)
#: @warrant docs/spec/01-cell-and-sheaf.md, Scale is gauge-fixed
#: `ρ`, the scale gauge's band edge, fixed at construction
#: (`docs/spec/01-cell-and-sheaf.md`, *Scale is gauge-fixed*). Interior maps live
#: in `[1/ρ, ρ]`; it also appears in the reconciliation gain's denominator as
#: `g_v`, the gauge each cell's maps are held to — `ρ` for a predicting cell and
#: exactly 1 for a boundary cell's pinned maps.
GAUGE_RHO = 2.0

#: @type stipulated
#: @flexibility measured: the overlap count runs 2.42 at the rim to 1.75-1.98 in the core (#182) against a ~1.05 floor set by effective rank, so 2 is conservative -- it is what levels 4-7 already satisfy untouched, and tightening it wants the cost to cross-edge alignment measured first (#184)
#: @warrant docs/adr/0010-restriction-map-scale-is-gauge-fixed.md, Incoherence is gauge-fixed too
#: `c`, the **effective overlap count**: how many of a cell's incident maps load
#: the same input direction. Declared globally alongside `ρ` and held by the same
#: projection, it is what makes `λ_max(Σ_e F_evᵀF_ev) ≤ g_v² · c_v` a true bound
#: rather than the fully-coherent `deg(v)` (`docs/spec/02-tick-semantics.md`,
#: *Reconciliation gain*; ruled by #190).
#:
#: **Declared, not yet enforced.** :meth:`RestrictionMaps.project` restores the
#: mask and the norm band and does not yet hold the top singular directions
#: apart, so nothing may divide by this constant until it does --
#: :func:`patchworks.tick.reconciliation_gain` still forms the superseded
#: denominator for exactly that reason. Both land together; see #220.
GAUGE_C = 2

#: @type chosen
#: @flexibility unknown
#: @warrant here
#: The Frobenius norm every map is drawn at. **Chosen here, not recorded.**
#: ADR-0010 fixes the band and the exact gauge but not where inside the band a
#: run starts. One is the band's geometric centre and the same value the
#: boundary maps are pinned at, so no edge starts with a scale ratio built into
#: it; the transport rule's own dynamics grow the larger end into `ρ` from
#: there (`docs/spec/01-cell-and-sheaf.md`, *Scale is gauge-fixed*).
INITIAL_NORM = 1.0


def pair_index(edge_id: int, side: int) -> int:
    """The endpoint index of `(edge, side)`. Side 0 is `edge.u`, side 1 `edge.v`."""
    return 2 * edge_id + side


class RestrictionMaps(torch.nn.Module):
    """Every cell's map into every incident edge stalk, as one padded tensor.

    The maps are the adapting surface's other half — per-cell, per-edge, and
    learned by the transport rule. They are registered as a parameter so that
    rule can reach them; the tick never differentiates through them, because
    the tick carries no tape.
    """

    def __init__(
        self,
        dome: Dome,
        *,
        rho: float = GAUGE_RHO,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if rho < 1.0:
            raise ValueError(f"the gauge band needs rho >= 1, got {rho}")
        self.dome = dome
        self.rho = rho
        self.pairs = 2 * len(dome.edges)
        self.edge_width = max(e.m for e in dome.edges)
        self.stalk_width = max(c.stalk for c in dome.cells)

        support = torch.zeros(
            (self.pairs, self.edge_width, self.stalk_width),
            dtype=torch.bool,
            device=device,
        )
        owner = torch.zeros(self.pairs, dtype=torch.long, device=device)
        # A boundary cell's own maps are pinned; every other map carries the
        # band. The test is who *holds* the map, not what the edge connects, so
        # the predicting end of a sensory edge is an ordinary interior map.
        pinned = torch.zeros(self.pairs, dtype=torch.bool, device=device)
        for edge in dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                i = pair_index(edge.id, side)
                permitted = dome.restriction_mask(edge.id, cell_id).to(device)
                support[i, : edge.m, : permitted.numel()] = permitted
                owner[i] = cell_id
                pinned[i] = dome.cells[cell_id].is_boundary

        self.register_buffer("support", support)
        self.register_buffer("owner", owner)
        self.register_buffer("pinned", pinned)

        draw = torch.empty(
            (self.pairs, self.edge_width, self.stalk_width), device=device, dtype=dtype
        ).normal_(0.0, 1.0, generator=generator)
        self.register_parameter("maps", torch.nn.Parameter(draw))
        with torch.no_grad():
            self.maps.mul_(self.support)
            self.maps.mul_(
                (INITIAL_NORM / self.norms().clamp_min(1e-12)).unsqueeze(-1).unsqueeze(-1)
            )

    # -- the gauge ---------------------------------------------------------

    def norms(self) -> torch.Tensor:
        """`[pairs]`: the Frobenius norm of each map."""
        return self.maps.flatten(1).norm(dim=-1)

    @property
    def gauge_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`[pairs]` lower and upper Frobenius bounds, per map.

        A pinned map's two bounds coincide at 1, which is the exact gauge
        written as a degenerate band so that projection is one operation.
        """
        exact = torch.ones(self.pairs, dtype=self.maps.dtype, device=self.maps.device)
        return (
            torch.where(self.pinned, exact, exact / self.rho),
            torch.where(self.pinned, exact, exact * self.rho),
        )

    @torch.no_grad()
    def project(self) -> None:
        """Restore the mask and the gauge, in place. Runs after a transport step.

        Both together rather than separately: a step that walked a weight out
        of the mask and a step that grew a map past `ρ` are the same kind of
        event — the surface leaving the shape construction gave it — and
        neither is ever wanted.
        """
        self.maps.mul_(self.support)
        lower, upper = self.gauge_bounds
        norms = self.norms().clamp_min(1e-12)
        scale = norms.clamp(lower, upper) / norms
        self.maps.mul_(scale.unsqueeze(-1).unsqueeze(-1))

    # -- what the message-passing phase runs -------------------------------

    def restrict(self, stalks: torch.Tensor) -> torch.Tensor:
        """`[pairs, stalk_max]` node stalks in, `[pairs, m_max]` beliefs out.

        One `bmm` over every edge endpoint in the graph. The caller gathers each
        pair's owning cell's node stalk into the padded row; rows and columns
        past that pair's `m` and stalk dimension are zero on both sides.
        """
        self._check(stalks, self.stalk_width, "stalks")
        return torch.bmm(self.maps, stalks.unsqueeze(-1)).squeeze(-1)

    def spread(self, edge_values: torch.Tensor) -> torch.Tensor:
        """`[pairs, m_max]` in, `[pairs, stalk_max]` out: `Fᵀ` applied per pair.

        The transpose of :meth:`restrict`, which is what carries a disagreement
        on an edge stalk back to the node stalk directions that produced it.
        Masked and padded directions come back zero, so a private feature is
        untouched by construction rather than by a second mask being applied.
        """
        self._check(edge_values, self.edge_width, "edge_values")
        return torch.bmm(self.maps.transpose(1, 2), edge_values.unsqueeze(-1)).squeeze(-1)

    def _check(self, x: torch.Tensor, width: int, name: str) -> None:
        if x.shape != (self.pairs, width):
            raise ValueError(
                f"{name} must be [{self.pairs}, {width}], got {tuple(x.shape)}"
            )

    def extra_repr(self) -> str:
        return (
            f"pairs={self.pairs}, m_max={self.edge_width}, "
            f"stalk_max={self.stalk_width}, rho={self.rho}"
        )
