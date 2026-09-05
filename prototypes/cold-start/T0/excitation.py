"""Per-cell **excitation rank**: #154 §3's edge instrument, applied to a cell's evidence stream.

The instrument the cold-start map (`#517
<https://github.com/NGL321/patchworks/issues/517>`_) needs everywhere, built
once here by T0 (`#518 <https://github.com/NGL321/patchworks/issues/518>`_) and
imported by T2 and T3.

**What it measures.** How many directions a cell's evidence turns through over
a window. Take the `[T, d]` matrix of the node stalk a cell read as evidence on
each of the last `T` ticks; its singular values `σᵢ` give the participation
ratio `(Σσᵢ²)² / Σσᵢ⁴`, which reads `1` for a stream that never leaves one
direction and `d` for one spread evenly over all of them. It is the same
statistic `patchworks.diagnostics` takes of a restriction map's singular values
(ADR-0010) and `#154 <https://github.com/NGL321/patchworks/issues/154>`_ §3
takes of an edge -- here the matrix is a *stream* rather than a map.

**Uncentred is the primary reading.** The mechanism this instrument was built to
read (#517's diagnosis, from the audit's toy) is that the prediction rule's
update to `K` is an outer product whose row space lies along the cell's
`encode` output, so updates under evidence whose *direction* does not turn over
add coherently. A constant stream is one direction, and the uncentred
participation ratio reads exactly 1 for it; centred, the same stream has no
singular values at all. The centred ratio is reported beside it, because the
toy also collapsed under two alternating fixed directions (uncentred rank 2,
centred rank 1) and did not under rank-3 evidence, so which of the two
separates collapsing from live cells is itself a reading and not a choice.

**Per block.** A predicting cell's node stalk is a shared-prefix mask
(`Dome.restriction_mask`): the leading `permitted` columns are exposed on every
incident edge, the rest are private (`H⁰` by construction). Within the exposed
block, a drive-adjacent cell's drive edge reads one direction -- the row of the
cell's own map onto that edge, learned, so it is read at the checkpoint rather
than fixed -- and the rest of the exposed block is what the interior edges can
see. So the stream splits into three orthogonal pieces:

* ``private``  -- the masked suffix, columns `permitted:`;
* ``drive``    -- the component of the exposed block along the unit row of the
  cell's map onto its drive edge (zero for cells with no drive edge);
* ``interior`` -- the exposed block with the drive component removed.

`excitation_rank` of the drive piece is degenerate (a one-dimensional stream
has participation ratio 1 whenever it is non-zero), so the drive lane is read
by its share of the stream's energy and variance rather than by rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def participation_ratio(stream: torch.Tensor, *, centred: bool = False) -> torch.Tensor:
    """`[cells, T, d]` in, `[cells]` out: `(Σσᵢ²)² / Σσᵢ⁴` of each cell's window.

    A stream that is identically zero over the window has no direction at all
    and reads `0`, which is distinct from the `1` a constant non-zero stream
    reads; a reader wanting "how many directions" should treat `0` as *none*.
    """
    if stream.ndim != 3:
        raise ValueError(f"stream must be [cells, T, d], got {tuple(stream.shape)}")
    x = stream.double()
    if centred:
        x = x - x.mean(dim=1, keepdim=True)
    # `Σσᵢ² = ‖X‖_F²` and `Σσᵢ⁴ = ‖XᵀX‖_F²`, the identity `diagnostics.py`
    # uses for maps, so no SVD is needed and a `[cells, d, d]` Gram is all the
    # arithmetic. Taken on the unit-normalised matrix so the quotient cannot
    # overflow or need a floor: for `G = X / ‖X‖_F` the ratio is `1 / ‖GᵀG‖_F²`.
    norm = torch.linalg.matrix_norm(x, ord="fro")
    live = norm > 0
    g = x / norm.clamp(min=1e-300).view(-1, 1, 1)
    gram = torch.bmm(g.transpose(1, 2), g)
    denominator = (gram * gram).sum(dim=(1, 2))
    ratio = torch.where(live, 1.0 / denominator.clamp(min=1e-300), torch.zeros_like(denominator))
    return ratio


def excitation_rank(
    stream: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    centred: bool = False,
) -> torch.Tensor:
    """The participation ratio of the columns `mask` keeps, per cell.

    `mask` is `[cells, d]` bool (or None for the whole stalk). Zeroed columns
    contribute zero singular values and change neither sum, so masking is a
    multiply rather than a gather, and cells with different exposed widths sit
    in one batch.
    """
    if mask is not None:
        stream = stream * mask.to(stream.dtype).unsqueeze(1)
    return participation_ratio(stream, centred=centred)


@dataclass(frozen=True)
class Blocks:
    """The three-way split of every predicting cell's node stalk, `[cells, n]` each.

    `drive_direction` is a unit vector inside the exposed block for a
    drive-adjacent cell and zero elsewhere; it is **learned**, so build one
    per checkpoint rather than once.
    """

    private: torch.Tensor
    exposed: torch.Tensor
    drive_direction: torch.Tensor
    #: `[cells, n, n]`: orthogonal projector onto the span of the cell's
    #: interior maps' rows (what its interior neighbours can read). Rank at most
    #: `Σ_interior m_e`, inside the exposed block.
    interior_rowspace: torch.Tensor

    def decompose(self, stream: torch.Tensor) -> dict[str, torch.Tensor]:
        """`[cells, T, n]` in; the three orthogonal pieces out, same shape each."""
        f = self.drive_direction.to(stream.dtype)
        exposed = stream * self.exposed.to(stream.dtype).unsqueeze(1)
        along = torch.einsum("ctn,cn->ct", exposed, f)
        drive = along.unsqueeze(-1) * f.unsqueeze(1)
        return {
            "private": stream * self.private.to(stream.dtype).unsqueeze(1),
            "interior": exposed - drive,
            "drive": drive,
        }


def blocks(agent) -> Blocks:
    """Read the split off a live agent: masks from the dome, directions from the maps."""
    from patchworks.graph import CellKind, EdgeKind
    from patchworks.restriction import pair_index

    dome = agent.dome
    maps = agent.sheaf.maps
    predicting = list(dome.predicting)
    n = dome.shape.n
    private = dome.private_mask.clone()
    exposed = ~private
    drive_cells = {cid for cid, cell in enumerate(dome.cells) if cell.kind == CellKind.DRIVE}
    drive_direction = torch.zeros(len(predicting), n)
    rowspace = torch.zeros(len(predicting), n, n)
    with torch.no_grad():
        for row, cid in enumerate(predicting):
            interior_rows = []
            for edge_id in dome.incident[cid]:
                edge = dome.edges[edge_id]
                side = 0 if edge.u == cid else 1
                f = maps.maps[pair_index(edge.id, side), : edge.m, :n].detach().double()
                if edge.other(cid) in drive_cells:
                    if edge.m != 1:
                        raise ValueError(f"drive edge {edge.id} is {edge.m} wide; T0 reads one drive lane")
                    d = f[0]
                    norm = d.norm()
                    if norm > 0:
                        drive_direction[row] = (d / norm).float()
                elif edge.kind == EdgeKind.INTERIOR:
                    interior_rows.append(f)
            if interior_rows:
                stacked = torch.cat(interior_rows, dim=0)
                # Orthonormal basis of the row span, by SVD, tolerance at torch's default.
                u, s, vh = torch.linalg.svd(stacked, full_matrices=False)
                tol = s.max() * max(stacked.shape) * torch.finfo(torch.float64).eps
                basis = vh[s > tol]
                rowspace[row] = (basis.T @ basis).float()
    return Blocks(
        private=private,
        exposed=exposed,
        drive_direction=drive_direction,
        interior_rowspace=rowspace,
    )
