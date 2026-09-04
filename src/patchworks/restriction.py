"""The restriction maps: one masked linear map per edge endpoint, gauge-fixed.

Each cell holds one map from its node stalk into each incident communication lane
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

import math

import torch

from .graph import Dome

__all__ = [
    "GAUGE_C",
    "GAUGE_RHO",
    "INITIAL_NORM",
    "RestrictionMaps",
    "cell_gauges",
    "gain_denominators",
    "overlap_counts",
    "pair_index",
]

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
#: **Enforced by the projection, which is what lets the gain divide by it**
#: (#220). :meth:`RestrictionMaps.project` restores the mask, then the norm
#: band, then caps each holding cell's Gram spectrum at `g_v^2 . c_v`, so
#: :func:`patchworks.tick.reconciliation_gain` divides by a bound that is true
#: of the surface at every tick rather than one hoped for.
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


def cell_gauges(dome: Dome, *, rho: float = GAUGE_RHO) -> torch.Tensor:
    """`[cells]`: `g_v`, the gauge each cell's own maps are held to.

    `rho` at a predicting cell, whose maps carry the band, and exactly 1 at a
    boundary cell, whose maps carry the exact gauge (ADR-0010). The test is who
    holds the map, not what the edge connects — the same test
    :class:`RestrictionMaps` applies when it decides which maps are pinned.
    """
    return torch.tensor(
        [1.0 if cell.is_boundary else rho for cell in dome.cells], dtype=torch.float32
    )


def overlap_counts(dome: Dome, *, c: int = GAUGE_C) -> torch.Tensor:
    """`[cells]`: `c_v = min(deg(v), max(c, ceil(deg(v) / n_v)))`.

    The effective overlap count the gain divides by, with both of its clamps
    (`docs/adr/0010-restriction-map-scale-is-gauge-fixed.md`, *The floor is not
    optional, and the drive cell is why*). The **pigeonhole floor** is a fact
    about dimension: `deg(v)` directions in an `n_v`-dimensional stalk cannot be
    spread further than `deg(v) / n_v` apart, so a bare global `c` is an unsafe
    bound at the drive, which carries 8 maps on a stalk of dimension 1. The
    outer `min` keeps the result inside the bound the band alone already gave.

    **Where `c_v < deg(v)` at a boundary cell, nothing enforces it.**
    :meth:`RestrictionMaps.project` reaches only the maps with scale freedom to
    spend, which is the predicting cells'. On this dome that is one cell, the
    actuator, and it is measured rather than constructed; the note on
    :meth:`RestrictionMaps._push_apart` carries the reading and #228 carries the
    question.
    """
    return torch.tensor(
        [
            float(min(deg, max(c, -(-deg // cell.stalk))))
            for cell, deg in zip(dome.cells, dome.degrees, strict=True)
        ],
        dtype=torch.float32,
    )


def gain_denominators(
    dome: Dome, *, rho: float = GAUGE_RHO, c: int = GAUGE_C
) -> torch.Tensor:
    """`[cells]`: `g_v^2 . c_v`, the bound on `lambda_max(sum_e F_ev^T F_ev)`.

    The reconciliation gain's denominator, ruled by
    [#190](https://github.com/NGL321/patchworks/issues/190) and published at
    `docs/spec/02-tick-semantics.md`, *Reconciliation gain*. **One definition,
    three readers**: the gain itself
    (:func:`patchworks.tick.reconciliation_gain`), the fold-margin nomination
    that divides by it (:func:`patchworks.bias_selection.fold_margin_check`),
    and the projection that makes it true
    (:meth:`RestrictionMaps.project`). A second implementation would let the
    bound the surface is held to drift away from the bound the gain assumes,
    and the whole of #220 is that those two are one thing.
    """
    return cell_gauges(dome, rho=rho) ** 2 * overlap_counts(dome, c=c)


def pair_index(edge_id: int, side: int) -> int:
    """The endpoint index of `(edge, side)`. Side 0 is `edge.u`, side 1 `edge.v`."""
    return 2 * edge_id + side


class RestrictionMaps(torch.nn.Module):
    """Every cell's map into every incident lane, as one padded tensor.

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
        blocks: list[tuple[int, int]] = [(0, 0)] * self.pairs
        for edge in dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                i = pair_index(edge.id, side)
                permitted = dome.restriction_mask(edge.id, cell_id).to(device)
                support[i, : edge.m, : permitted.numel()] = permitted
                owner[i] = cell_id
                pinned[i] = dome.cells[cell_id].is_boundary
                # The active block, which is what the spectral floor is stated
                # on: `m_e` rows against the `k_v` columns the mask leaves open.
                # The mask is a prefix and it is applied to every row alike, so
                # the block is the rectangle `[:m_e, :k_v]` and nothing outside
                # it is ever read or written.
                blocks[i] = (edge.m, int(permitted.sum()))

        self.register_buffer("support", support)
        self.register_buffer("owner", owner)
        self.register_buffer("pinned", pinned)

        # What the incoherence term needs, read off the built graph once. A
        # pinned map is out of the projection's reach -- the exact gauge leaves
        # no freedom to spend -- so the cells that hold the term are exactly
        # the predicting ones, and their pairs are exactly the unpinned ones.
        holding = torch.tensor(
            [not cell.is_boundary for cell in dome.cells], device=device
        )
        exposed = torch.zeros(
            (len(dome.cells), self.stalk_width), dtype=torch.float32, device=device
        )
        exposed.index_add_(0, owner, support.any(dim=1).to(torch.float32))
        column_mask = exposed > 0
        hold_rows = torch.full((len(dome.cells),), -1, dtype=torch.long, device=device)
        hold_rows[holding] = torch.arange(
            int(holding.sum()), dtype=torch.long, device=device
        )
        # The mask is a prefix of the node stalk, shared by all of a cell's
        # incident maps, so the whole incoherence step fits in the leading
        # block and the padding never reaches an eigendecomposition.
        held_columns = column_mask[holding].any(dim=0)
        width = int(held_columns.nonzero().max()) + 1 if bool(held_columns.any()) else 0

        self.register_buffer("holding", holding)
        self.register_buffer("holding_cells", holding.nonzero().squeeze(-1))
        self.register_buffer("hold_rows", hold_rows)
        self.register_buffer("hold_pairs", (~pinned).nonzero().squeeze(-1))
        self.register_buffer("column_mask", column_mask)
        self.register_buffer(
            "overlap_target", gain_denominators(dome, rho=rho).to(device)
        )
        self.hold_width = width

        # -- what the spectral floor reaches (ADR-0032) --------------------
        #
        # **The exclusion is by attainability, not by pinning**, and the two
        # populations differ. `_push_apart` skips a pinned map because the
        # exact gauge leaves it no *scale* freedom to spend; the floor needs
        # none, because projecting onto the nearest scaled co-isometry
        # preserves `‖F‖_F` exactly. What it needs is *rank*: a mask with
        # `k_v < m_e` open columns cannot contain a co-isometry at all, so
        # `σ_min = 0` there whatever the projection does and the projection
        # would shrink `‖F‖_F` by `√(k/m)` — fighting the exact gauge rather
        # than sitting beside it. Those masks are excluded by name, computed
        # here from the mask rather than listed: on `DEFAULT_SPEC` they are the
        # nine ADR-0032 names, all pinned — three touch (`m = 8, k = 1`), three
        # proprioceptive (`8, 2`) and the actuator's three (`8, 6`); see
        # `prototypes/mask-attainability-415/`.
        #
        # **`m = 1` is skipped because the floor is vacuous there**, not
        # because it is unattainable: one singular value is `‖F‖_F/√1`
        # identically, so the projection is the identity and running an SVD to
        # discover that every tick is cost with no consequence. That is the
        # drive's eight edges.
        #
        # **Grouped by `(m_e, k_v)` so the ragged shapes batch.** A single SVD
        # over the padded `[pairs, m_max, stalk_max]` tensor would flatten the
        # *padding* — structural zeros are not directions, and a co-isometry
        # fitted to them would write nonzero weights into rows and columns
        # construction closed. Grouping means every batched SVD sees a dense
        # block of one shape, and the padded rows and masked columns come back
        # exactly zero because they are never touched: the same guarantee
        # `_push_apart` gets from the mask being a shared prefix.
        self.floor_shapes: list[tuple[int, int]] = []
        reachable = [
            i
            for i in range(self.pairs)
            for m, k in (blocks[i],)
            if m > 1 and k >= m
        ]
        for m, k in sorted({blocks[i] for i in reachable}):
            members = [i for i in reachable if blocks[i] == (m, k)]
            self.register_buffer(
                f"floor_group_{len(self.floor_shapes)}",
                torch.tensor(members, dtype=torch.long, device=device),
            )
            self.floor_shapes.append((m, k))
        self.register_buffer(
            "floored",
            torch.zeros(self.pairs, dtype=torch.bool, device=device).index_fill_(
                0, torch.tensor(reachable, dtype=torch.long, device=device), True
            )
            if reachable
            else torch.zeros(self.pairs, dtype=torch.bool, device=device),
        )

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
        """Restore the mask, the gauge and the incoherence, in place.

        Runs after a transport step. All three together rather than separately:
        a step that walked a weight out of the mask, a step that grew a map past
        `ρ`, and a step that turned a cell's incident maps to face the same way
        are the same kind of event — the surface leaving the shape construction
        gave it — and none of them is ever wanted.

        **The order is load-bearing.** The mask first, because the three later
        steps read norms and Gram blocks that a stray weight would pollute; the
        band next, because the incoherence target `g_v² · c_v` is stated against
        the banded gauge; the spectral floor after it, because the floor is
        stated against a map's own `‖F‖_F` and preserves it exactly, so it
        neither disturbs the band it follows nor needs it re-applied; the
        incoherence cap last, because it is the one whose guarantee the others
        could undo. What follows it may only shrink a map, and shrinking a map
        can only shrink the Gram it contributes to, so the cap holds at exit.

        **The floor sits before the cap, and that is a choice with a price.**
        ADR-0032 leaves the placement to the build and it is a real collision:
        water-filling moves energy between a cell's directions and so un-flattens
        what the floor flattens, while a flattening step redistributes a map's
        singular values and so can raise the Gram the cap just bounded. Two
        invariants, and only one can be exactly true at exit.

        **The cap wins the last slot because it is the one with an external
        reader.** :func:`patchworks.tick.reconciliation_gain` divides by
        `g_v² · c_v` on every tick, and the whole of
        [#220](https://github.com/NGL321/patchworks/issues/220) is that the bound
        held and the bound assumed are one thing — a cap that is only
        approximately true at exit is a false denominator in the gain, where a
        floor that is only approximately true at exit is a measurement. So **what
        holds at exit is `λ_max(Σ_e F_evᵀF_ev) ≤ g_v² · c_v`, exactly and by
        construction, unchanged from before this step existed**, and the floor
        holds exactly wherever the cap does not bite and approximately where it
        does. :meth:`flatness` is what reads the residual, and it is the
        instrument ADR-0032's second pre-registration needs.

        **Flatness alone cannot carry the cap, which is why the cap is kept.**
        A flat map has `σ_max² = ‖F‖²_F/m_e`, so the triangle bound gives
        `λ_max ≤ Σ_e ‖F_e‖²_F/m_e ≤ g_v² Σ_e 1/m_e` — a derived bound needing no
        constant, and it discharges the cap wherever `Σ_e 1/m_e ≤ c_v`. On
        `DEFAULT_SPEC` it does not: four degree-9 interior cells carry nine
        `m = 4` lanes, so `Σ_e 1/m_e = 2.25` against `c_v = 2`, and the bound
        comes to `1.125x` the target. Ordering the floor last would trade a
        construction guarantee for a measurement at exactly the cells where the
        denominator is tightest.

        **The floor also makes the cap bite less often**, which is the other
        half of putting it first: flattening lowers every map's `σ_max` to the
        RMS of its own spectrum, so the Gram the cap inspects is already spread
        when it arrives, and a cap that does not fire leaves the flatness it
        found exactly alone.
        """
        self.maps.mul_(self.support)
        lower, upper = self.gauge_bounds
        norms = self.norms().clamp_min(1e-12)
        scale = norms.clamp(lower, upper) / norms
        self.maps.mul_(scale.unsqueeze(-1).unsqueeze(-1))
        self._flatten()
        self._push_apart()
        # The band's upper edge, once more and downward only. Water-filling
        # moves energy between a cell's directions, so a map that was inside the
        # band can come out of it above; shrinking it back is the one correction
        # that cannot cost the incoherence cap, because a smaller map
        # contributes a smaller Gram. The lower edge is not re-applied for the
        # mirror-image reason, and it is restored above, before the step that
        # can move it.
        over = (self.norms() / upper).clamp_min(1.0)
        self.maps.div_(over.unsqueeze(-1).unsqueeze(-1))

    @torch.no_grad()
    def _flatten(self) -> None:
        """Project every reachable map onto the nearest scaled co-isometry.

        ADR-0032. The restriction maps are learning **isometric transport**, and
        the constraint that expresses it is a per-map spectral floor at
        `σ_min ≥ ‖F‖_F/√m`. Since `Σᵢσᵢ² = ‖F‖²_F`, that floor is attainable
        only with equality throughout, so **the floor at its one derivable value
        and the projection onto the nearest scaled co-isometry are the same
        operation**: for `F = UΣVᵀ` on the active block,

            `F ← (‖F‖_F/√m) · UVᵀ`.

        It costs no invented constant — any weaker floor needs a fraction, and
        the fraction would have no warrant.

        **It preserves `‖F‖_F` exactly**, which is why it sits beside ADR-0010's
        gauge rather than against it: the band holds a Frobenius norm and says
        nothing about how that budget is spread across the singular values, and
        this spreads it evenly without spending any of it. `σ_max` does move —
        down by `√m`, which ADR-0022 prices as the hop — and that cost is booked
        on the ADR, not discovered here.

        **This is the operation :meth:`_push_apart` structurally cannot
        perform.** The water-fill is gated on `live = eigenvalues > peak * 1e-9`,
        because scaling zero leaves zero: a cap flattens survivors and can never
        resurrect a dead direction. The projection can, and does — for a
        rank-deficient block the thin SVD's trailing right-singular vectors are
        an arbitrary orthonormal completion *inside the mask's open columns*, so
        `UVᵀ` returns a direction the map had lost while leaving every closed
        column at exactly zero. A cap exists; only a floor lifts a dead
        direction.

        **Isometry is a property of the edge pair, not of one map.** An interior
        map is `4 × 32` and has a 28-dimensional kernel by arithmetic; what the
        constraint buys is one of the two halves of `F_v⁺F_u ∈ O(m)` — flat
        spectra on both ends — and the other half is ADR-0010's matched edge
        scale, which [#429](https://github.com/NGL321/patchworks/issues/429)
        owns. Nothing here claims a map is an isometry.

        **The SVD is the cost, and the cheaper route was refused on
        correctness.** `UVᵀ` is also `(FFᵀ)^{-1/2}F`, an `m × m` eigendecomposition
        with `m ≤ 8` instead of an `m × k` SVD — but that is a *left*
        multiplication, and a left multiplication cannot raise a rank. It would
        reproduce the one property that distinguishes this step from
        :meth:`_push_apart` and silently drop it, which is the worst of the
        available trades. Measured on the real dome at 6 threads, the floor takes
        the projection from **11.1 ms to 27.7 ms**, and the projection runs once
        per transport step; the two tick phases beside it are 1.2 ms. A 100,000
        tick horizon — which is the horizon ADR-0032's falsification is
        pre-registered at — is about 46 minutes of projection against 18 before.
        Recorded rather than optimised: it is inside what the rig can pay, and
        the first cheaper route is a conditioning threshold nobody has a warrant
        for.

        Which maps it reaches, and why the nine are excluded, is recorded on the
        `floor_shapes` block in :meth:`__init__`.
        """
        for group, (m, k) in enumerate(self.floor_shapes):
            pairs = getattr(self, f"floor_group_{group}")
            block = self.maps.index_select(0, pairs)[:, :m, :k]
            left, spectrum, right = torch.linalg.svd(block, full_matrices=False)
            # `‖F‖_F` read off the spectrum rather than the tensor: the block is
            # the whole of the map, so the two agree, and taking it here keeps
            # the scale and the factorisation exactly consistent.
            flat = spectrum.square().sum(dim=-1).sqrt() / math.sqrt(m)
            self.maps[pairs, :m, :k] = (left @ right) * flat.unsqueeze(-1).unsqueeze(-1)

    @torch.no_grad()
    def _push_apart(self) -> None:
        """Cap each holding cell's Gram spectrum at `g_v² · c_v`, in place.

        ADR-0010's *Incoherence is gauge-fixed too*. The gain divides by a bound
        on `λ_max(Σ_e F_evᵀF_ev)`, and the band alone only supports the
        fully-coherent `g_v² · deg(v)`. This is what makes the smaller number
        true: one shared right-transform per cell, applied to every map that
        cell holds, chosen so the summed Gram's largest eigenvalue lands on the
        target.

        **It redistributes rather than shrinks.** The excess above the target is
        water-filled back onto the cell's under-loaded directions, so the trace
        — which is `Σ_e ‖F_ev‖_F²`, the thing the band holds — is preserved
        wherever there is headroom to preserve it into. A cap alone would take
        energy out of the surface on every tick and ratchet the maps down onto
        the band's lower edge, which is the same map collapse the band exists to
        prevent, arriving by a different road. Where the headroom will not take
        the whole excess the step falls back to the bare cap: conservative, and
        the bound holds either way.

        **The transform cannot re-open the mask, and that is why it is one
        transform per cell rather than one per map.** All of a cell's incident
        maps share the same structural mask, a prefix of its node stalk
        (:meth:`patchworks.graph.Dome.restriction_mask`), so a right-transform
        supported on that prefix leaves every masked column zero. A per-map
        transform would have no such common support.

        **Pinned maps are untouched, deliberately.** A boundary cell's maps
        carry the exact gauge, so there is no scale freedom for a transform to
        spend; the only norm-preserving right-transform is orthogonal, and an
        orthogonal one leaves the spectrum exactly where it found it. Those
        cells get their correction from `g_v = 1` instead, which is
        :func:`cell_gauges`' business and not this one.

        **And that leaves exactly one cell on this dome whose target is held by
        measurement rather than by construction: the actuator.** Every other
        boundary cell has `deg(v) = 1` or, at the drive, a pigeonhole floor that
        raises `c_v` to `deg(v)`, and at `c_v = deg(v)` the exact gauge makes
        `Σ_e ‖F‖_F² = deg(v)` an equality and the bound true unaided. The
        actuator carries `deg = 3` on a stalk of 6, so `c_v` is the global `c`
        of 2 while nothing pushes its three maps apart; the fully-coherent
        arrangement would put `λ_max` at 3. Measured, it is **1.0164 at
        construction and 1.0029 after 5,000 taught ticks, against a target of
        2.0** — it holds with 1.97x to spare and does not drift toward coherence
        over a run, so nothing here is unsafe today. It is recorded rather than
        quietly corrected because the correction would be a change to a ruled
        denominator (#190) and this is a build, not a ruling; :meth:`gram_peaks`
        is what reads it. See #228.
        """
        if self.hold_pairs.numel() == 0 or self.hold_width == 0:
            return
        width = self.hold_width
        tiny = torch.finfo(self.maps.dtype).tiny

        held = self.maps.index_select(0, self.hold_pairs)[:, :, :width]
        grams = torch.zeros(
            (len(self.dome.cells), width, width),
            dtype=self.maps.dtype,
            device=self.maps.device,
        )
        grams.index_add_(
            0,
            self.owner.index_select(0, self.hold_pairs),
            held.transpose(1, 2) @ held,
        )
        eigenvalues, directions = torch.linalg.eigh(
            grams.index_select(0, self.holding_cells)
        )

        target = self.overlap_target.index_select(0, self.holding_cells).unsqueeze(-1)
        peak = eigenvalues.amax(dim=-1, keepdim=True)
        capped = torch.minimum(eigenvalues, target)
        excess = (eigenvalues - capped).sum(dim=-1, keepdim=True)
        # A direction the cell's maps do not load at all cannot be filled into:
        # scaling zero leaves zero, whatever the scale. The threshold is
        # relative to the cell's own peak so it means the same thing at every
        # cell, whatever the maps' magnitudes.
        live = eigenvalues > peak * 1e-9
        headroom = (target - capped) * live
        total = headroom.sum(dim=-1, keepdim=True)
        share = torch.where(
            total > 0,
            (excess / total.clamp_min(tiny)).clamp(max=1.0),
            torch.zeros_like(total),
        )
        filled = capped + headroom * share
        scale = torch.where(
            live,
            (filled / eigenvalues.clamp_min(tiny)).sqrt(),
            torch.ones_like(eigenvalues),
        )
        # A cell already inside the bound is left exactly alone, rather than
        # multiplied by a numerical approximation of the identity every tick.
        scale = torch.where(peak > target, scale, torch.ones_like(scale))

        transform = directions @ (scale.unsqueeze(-1) * directions.transpose(1, 2))
        transform = transform * self.column_mask.index_select(0, self.holding_cells)[
            :, :width
        ].unsqueeze(1)
        rows = self.hold_rows.index_select(
            0, self.owner.index_select(0, self.hold_pairs)
        )
        self.maps[self.hold_pairs, :, :width] = torch.bmm(
            held, transform.index_select(0, rows)
        )

    def gram_peaks(self) -> torch.Tensor:
        """`[cells]`: `λ_max(Σ_{e∈v} F_evᵀF_ev)`, the quantity the gain bounds.

        Reporting, not runtime — the tick never calls it. It exists so a test
        and a benchmark can read the bound the projection is supposed to hold,
        against the denominator :func:`gain_denominators` hands the gain, and so
        the two can be compared at a boundary cell as well as an interior one.

        **The eigendecomposition is taken in float64, and the surface this is
        for is why.** A cell's Gram is a sum of outer products of maps that the
        transport rule is free to drive rank-deficient, and a float32 `eigvalsh`
        on a matrix with repeated or near-zero eigenvalues does not merely lose
        digits — it *fails to converge* and raises. Reading a surface without
        ADR-0032's floor at 100k does exactly that (`benchmarks/spectral_floor_read.py`,
        the `--no-floor` arm, whose maps reach `σ_min/σ_max` of 1e-27). An
        instrument that raises on the degenerate surface is an instrument that
        cannot report the case it exists to catch, so the promotion is the fix
        and it costs nothing the tick pays: #393's rank reading takes its ratio
        in float64 for the same reason and by the same argument. The result is
        returned in the maps' own dtype, so every caller sees what it always saw.
        """
        maps = self.maps.detach().to(torch.float64)
        grams = torch.zeros(
            (len(self.dome.cells), self.stalk_width, self.stalk_width),
            dtype=torch.float64,
            device=self.maps.device,
        )
        grams.index_add_(0, self.owner, maps.transpose(1, 2) @ maps)
        return torch.linalg.eigvalsh(grams).amax(dim=-1).to(self.maps.dtype)

    def flatness(self) -> torch.Tensor:
        """`[pairs]`: `σ_min/σ_max` of each map's active block, 1 when flat.

        Reporting, not runtime — the tick never calls it. It exists because
        :meth:`project` orders the spectral floor **before** the incoherence cap
        and so holds the floor exactly only where the cap does not bite; this is
        what reads the residual, and it is the instrument ADR-0032's second
        pre-registration is written against.

        A map the floor does not reach comes back at its own honest ratio rather
        than at 1 — the nine unattainable masks read 0, since `rank(F) ≤ k < m`
        makes `σ_min` zero there whatever anything does, and `m = 1` reads 1
        because one singular value is trivially its own smallest and largest.

        **The spectrum is the map's `m` singular values, not the block's.**
        `svdvals` on an `[m, k]` block returns `min(m, k)` of them, so where the
        mask leaves fewer columns open than the lane is wide the remaining
        `m − k` are structural zeros that the factorisation simply does not
        report. Reading `σ_min` off the returned list there would say a map is
        nearly flat when it is rank-deficient by construction, which is exactly
        the population the floor excludes and the last place to flatter it.
        """
        maps = self.maps.detach()
        tiny = torch.finfo(maps.dtype).tiny
        ratios = torch.zeros(self.pairs, dtype=maps.dtype, device=maps.device)
        for edge in self.dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                i = pair_index(edge.id, side)
                k = int(self.support[i].any(dim=0).sum())
                spectrum = torch.linalg.svdvals(maps[i, : edge.m, :k])
                top = spectrum.amax().clamp_min(tiny)
                floor = spectrum.amin() if k >= edge.m else spectrum.new_zeros(())
                ratios[i] = floor / top
        return ratios

    def flatness(self) -> torch.Tensor:
        """`[pairs]`: `σ_min/σ_max` of each map's active block, 1 when flat.

        Reporting, not runtime — the tick never calls it. It exists because
        :meth:`project` orders the spectral floor **before** the incoherence cap
        and so holds the floor exactly only where the cap does not bite; this is
        what reads the residual, and it is the instrument ADR-0032's second
        pre-registration is written against.

        A map the floor does not reach comes back at its own honest ratio rather
        than at 1 — the nine unattainable masks read 0, since `rank(F) ≤ k < m`
        makes `σ_min` zero there whatever anything does, and `m = 1` reads 1
        because one singular value is trivially its own smallest and largest.

        **The spectrum is the map's `m` singular values, not the block's.**
        `svdvals` on an `[m, k]` block returns `min(m, k)` of them, so where the
        mask leaves fewer columns open than the lane is wide the remaining
        `m − k` are structural zeros that the factorisation simply does not
        report. Reading `σ_min` off the returned list there would say a map is
        nearly flat when it is rank-deficient by construction, which is exactly
        the population the floor excludes and the last place to flatter it.
        """
        maps = self.maps.detach()
        tiny = torch.finfo(maps.dtype).tiny
        ratios = torch.zeros(self.pairs, dtype=maps.dtype, device=maps.device)
        for edge in self.dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                i = pair_index(edge.id, side)
                k = int(self.support[i].any(dim=0).sum())
                spectrum = torch.linalg.svdvals(maps[i, : edge.m, :k])
                top = spectrum.amax().clamp_min(tiny)
                floor = spectrum.amin() if k >= edge.m else spectrum.new_zeros(())
                ratios[i] = floor / top
        return ratios

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
        on a lane back to the node stalk directions that produced it.
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
