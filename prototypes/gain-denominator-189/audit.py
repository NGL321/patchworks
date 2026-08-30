"""The denominator audit's own readings: what binds, where, and how loose.

#189 assembles a ledger from the record. Three of its entries are claims about
quantities that are **read straight off the built graph**, so they are checked
here rather than quoted: which arm of the `max` binds at each cell, what the
looseness is against the bound that actually holds for that cell's maps, and
whether the boundary cells -- which the record's tables never separate out --
are the same story as the predicting ones.

Everything here is construction-time. `sum_e m_e` and `deg(v)` are integers
fixed by `build_graph`, so the binder table is exact and needs no training; the
`lambda_max` readings are at initialisation and are reported as such. #182's
trained readings are the same quantity after 30k ticks and are not re-run here.

Run inside the supported container (ADR-0012):

    docker run --rm -v "$PWD:/work" -w /work --entrypoint python patchworks:189 \
        prototypes/gain-denominator-189/audit.py
"""

from __future__ import annotations

import numpy as np
import torch

from patchworks.graph import build_graph
from patchworks.restriction import GAUGE_RHO, RestrictionMaps, pair_index
from patchworks.tick import DEFAULT_GAMMA, reconciliation_gain

RHO = GAUGE_RHO


def rows(dome):
    """One record per cell: the two arms, which binds, and the gain."""
    gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=RHO)
    out = []
    for cell in dome.cells:
        summ = dome.stalk_sums[cell.id]
        deg = dome.degrees[cell.id]
        bound = RHO * RHO * deg
        if summ > bound:
            binder = "sum_m"
        elif summ < bound:
            binder = "rho2deg"
        else:
            binder = "tie"
        out.append(
            {
                "id": cell.id,
                "level": cell.index.level,
                "boundary": cell.is_boundary,
                "kind": cell.kind.value,
                "deg": deg,
                "sum_m": summ,
                "rho2deg": bound,
                "binder": binder,
                "denominator": max(summ, bound),
                "gain": float(gains[cell.id]),
            }
        )
    return out


def binder_table(recs):
    """Which arm takes the `max`, split by cell kind -- ties named as ties.

    #182 reports 142 of 150 cells binding on `sum_e m_e`; #150 reports 80 of 150
    already on the `rho^2 deg` arm. Both are over the 150 **predicting** cells
    and both are right: the 72 exact ties are on either arm by choice of
    tie-break, and neither ticket says which it took. Naming the tie column is
    what makes the two readings the same reading.
    """
    print("\n### which arm takes the max\n")
    print(f"  {'population':>22} {'cells':>6} {'sum_m':>7} {'tie':>6} {'rho2deg':>9}")
    for label, sel in (
        ("predicting", lambda r: not r["boundary"]),
        ("boundary", lambda r: r["boundary"]),
        ("all", lambda r: True),
    ):
        sub = [r for r in recs if sel(r)]
        counts = {k: sum(1 for r in sub if r["binder"] == k) for k in ("sum_m", "tie", "rho2deg")}
        print(
            f"  {label:>22} {len(sub):>6} {counts['sum_m']:>7} {counts['tie']:>6} "
            f"{counts['rho2deg']:>9}"
        )

    print("\n  predicting cells, by level:\n")
    print(f"  {'level':>6} {'cells':>6} {'deg':>7} {'sum_m':>7} {'rho2deg':>9} {'binder':>9}")
    levels = sorted({r["level"] for r in recs if not r["boundary"]})
    for lvl in levels:
        sub = [r for r in recs if not r["boundary"] and r["level"] == lvl]
        binders = sorted({r["binder"] for r in sub})
        print(
            f"  {lvl:>6} {len(sub):>6} {np.mean([r['deg'] for r in sub]):>7.2f} "
            f"{np.mean([r['sum_m'] for r in sub]):>7.1f} "
            f"{np.mean([r['rho2deg'] for r in sub]):>9.1f} {'/'.join(binders):>9}"
        )


def boundary_gauge(recs):
    """The bound that actually holds at a boundary cell, against the one applied.

    `restriction.py:23` pins a **boundary cell's own** maps at the exact gauge
    `||F||_F = 1` -- they are not in the interior band `[1/rho, rho]`, because a
    boundary cell has no metric individuality to protect. So the provable bound
    at such a cell is `lambda_max(sum_e F^T F) <= sum_e ||F||_F^2 = deg(v)`, and
    `tick.py:238` says as much: `rho^2 deg(v)` is "a valid bound for them too and
    merely a looser one". Looser by exactly `rho^2 = 4` -- except that at every
    boundary cell `sum_e m_e = 8 deg(v)` takes the max instead, so what is
    actually applied is looser by **8x**, by construction, before a single map
    is drawn or trained.
    """
    print("\n### the boundary cells, against their own exact gauge\n")
    sub = [r for r in recs if r["boundary"]]
    kinds = sorted({r["kind"] for r in sub})
    print(
        f"  {'kind':>14} {'cells':>6} {'deg':>5} {'sum_m':>7} {'rho2deg':>9} "
        f"{'exact':>7} {'applied/exact':>14}"
    )
    for kind in kinds:
        cells = [r for r in sub if r["kind"] == kind]
        deg = np.mean([r["deg"] for r in cells])
        summ = np.mean([r["sum_m"] for r in cells])
        applied = np.mean([r["denominator"] for r in cells])
        print(
            f"  {kind:>14} {len(cells):>6} {deg:>5.2f} {summ:>7.1f} "
            f"{np.mean([r['rho2deg'] for r in cells]):>9.1f} {deg:>7.2f} "
            f"{applied / deg:>13.2f}x"
        )


def lambda_max_at_init(dome, seed=0):
    """`denominator / lambda_max(sum_e F_ev^T F_ev)` per cell, at construction.

    #142 read this over the 150 predicting cells (41.8 untrained, 5.585 taught)
    and #182 split it per level. Neither reports the boundary cells, and this
    does, because the acceptance demo's instrument reads a motor boundary cell's
    stalk and the return path's last step is taken with that cell's gain.
    """
    maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(seed))
    gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=RHO)
    blocks: dict[int, torch.Tensor] = {}
    for edge in dome.edges:
        for cell_id in (edge.u, edge.v):
            side = 0 if cell_id == edge.u else 1
            stalk = dome.cells[cell_id].stalk
            f = maps.maps[pair_index(edge.id, side)][: edge.m, :stalk].detach()
            blocks[cell_id] = blocks.get(cell_id, torch.zeros(stalk, stalk)) + f.T @ f
    out = []
    for cell in dome.cells:
        block = blocks.get(cell.id)
        if block is None:
            continue
        lam = float(torch.linalg.eigvalsh(block)[-1])
        denominator = DEFAULT_GAMMA / float(gains[cell.id])
        out.append(
            {
                "id": cell.id,
                "level": cell.index.level,
                "boundary": cell.is_boundary,
                "lambda_max": lam,
                "denominator": denominator,
                "ratio": denominator / lam if lam > 0 else float("inf"),
            }
        )
    return out


def looseness_table(lam):
    print("\n### denominator / true lambda_max, at initialisation, seed 0\n")
    print(f"  {'population':>22} {'cells':>6} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}")
    for label, sel in (
        ("predicting", lambda r: not r["boundary"]),
        ("boundary", lambda r: r["boundary"]),
    ):
        sub = [r["ratio"] for r in lam if sel(r)]
        print(
            f"  {label:>22} {len(sub):>6} {np.mean(sub):>9.2f} {np.median(sub):>9.2f} "
            f"{np.min(sub):>9.2f} {np.max(sub):>9.2f}"
        )
    print("\n  predicting cells, by level:\n")
    print(f"  {'level':>6} {'cells':>6} {'mean':>9} {'median':>9}")
    for lvl in sorted({r["level"] for r in lam if not r["boundary"]}):
        sub = [r["ratio"] for r in lam if not r["boundary"] and r["level"] == lvl]
        print(f"  {lvl:>6} {len(sub):>6} {np.mean(sub):>9.2f} {np.median(sub):>9.2f}")


def gain_spread(recs):
    """What the denominator does to the gain across the taper.

    `02-tick-semantics.md` defends `sum_e m_e` as an **equaliser**: every cell
    takes roughly the same descent on its own local energy regardless of degree.
    The claim is checkable at construction, and what it equalises is the ratio
    of the applied step to the *bound*, not to the true `lambda_max`.
    """
    print("\n### the gain itself, by level (predicting cells)\n")
    print(f"  {'level':>6} {'cells':>6} {'gain':>10} {'vs apex':>9}")
    levels = sorted({r["level"] for r in recs if not r["boundary"]})
    apex = np.mean([r["gain"] for r in recs if not r["boundary"] and r["level"] == levels[-1]])
    for lvl in levels:
        sub = [r["gain"] for r in recs if not r["boundary"] and r["level"] == lvl]
        print(f"  {lvl:>6} {len(sub):>6} {np.mean(sub):>10.5f} {np.mean(sub) / apex:>8.3f}x")
    bs = [r["gain"] for r in recs if r["boundary"]]
    print(f"  {'bdy':>6} {len(bs):>6} {np.mean(bs):>10.5f} {np.mean(bs) / apex:>8.3f}x")


def main() -> None:
    dome = build_graph()
    recs = rows(dome)
    print(f"cells {len(recs)}, predicting {sum(1 for r in recs if not r['boundary'])}, "
          f"edges {len(dome.edges)}, rho {RHO}, gamma {DEFAULT_GAMMA}")
    binder_table(recs)
    boundary_gauge(recs)
    gain_spread(recs)
    looseness_table(lambda_max_at_init(dome))
    actuator_close_up(dome)
    equalisation_check(dome)
    margin_subspace_factor(dome)


def equalisation_check(dome, seed=0):
    """Does `sum_e m_e` equalise what `02` says it equalises?

    The claim (`02-tick-semantics.md`, *Reconciliation gain*): "every cell takes
    roughly the same descent on its own local energy regardless of how many
    edges it sits on. It removes a degree artifact."

    The descent a cell takes on its own local energy, relative to that energy's
    own curvature, is `gain_v * lambda_max(sum_e F^T F)` -- the step in units of
    the largest stable one. That is the quantity "the same descent on its own
    local energy" has to mean; the raw `gain_v` is a step in stalk units and
    says nothing about the energy it descends. So this reads the normalised
    step per level.
    """
    lam = lambda_max_at_init(dome, seed=seed)
    print("\n### what the equaliser equalises: gain x lambda_max, at init\n")
    print(f"  {'level':>6} {'cells':>6} {'norm. step':>11} {'vs level 1':>11}")
    levels = sorted({r["level"] for r in lam if not r["boundary"]})
    per = {
        lvl: float(np.mean([1.0 / r["ratio"] for r in lam if not r["boundary"] and r["level"] == lvl]))
        for lvl in levels
    }
    base = per[levels[0]]
    for lvl in levels:
        cells = sum(1 for r in lam if not r["boundary"] and r["level"] == lvl)
        print(f"  {lvl:>6} {cells:>6} {per[lvl]:>11.5f} {per[lvl] / base:>10.2f}x")
    print(f"\n  spread across the taper: {max(per.values()) / min(per.values()):.2f}x, "
          f"monotone in depth: {list(per.values()) == sorted(per.values())}")


def margin_subspace_factor(dome):
    """The fold-margin check's own dimensional mismatch, and its size.

    `bias_selection._fold_margin` reads `min_i |z_i| / ||grad z_i||` over
    `encode`'s **whole** input, `R^k x R^n` -- chart and node stalk together.
    What reconciliation displaces is the node stalk alone: `tick.py`'s
    message-passing phase edits `x_v` and never the chart. So the displacement
    is confined to the `R^n` block while the distance it is compared against is
    measured over all `k + n` directions.

    The mismatch is conservative -- distance to a boundary along a subspace is
    never less than the distance in the full space -- so nothing is unsafe. It
    is reported because the check's tightness is load-bearing (#178 reads a
    3.79x permitted factor off it) and this says how much of that tightness is
    an artifact of measuring in the wrong space.
    """
    from patchworks.body import BodyShape, CellBody

    shape = BodyShape(n=dome.spec.n, k=dome.spec.k)
    body = CellBody(shape, generator=torch.Generator().manual_seed(0))
    weight = body.encode_hidden_weight
    full = torch.linalg.vector_norm(weight, dim=-1)
    stalk_block = torch.linalg.vector_norm(weight[:, shape.k :], dim=-1)
    factor = (full / stalk_block.clamp(min=1e-12))
    print("\n### the fold margin's input space against reconciliation's\n")
    print(f"  encode: R^{shape.k} x R^{shape.n} -> R^{shape.k}, hidden width {weight.shape[0]}")
    print(f"  ||row|| over all {shape.k + shape.n} inputs / ||row|| over the {shape.n} stalk inputs")
    print(f"  mean {float(factor.mean()):.4f}, median {float(factor.median()):.4f}, "
          f"max {float(factor.max()):.4f}")
    print(f"  sqrt((k+n)/n) = {((shape.k + shape.n) / shape.n) ** 0.5:.4f}  -- the isotropic expectation")


def actuator_close_up(dome, seed=0):
    """The one boundary cell whose reconciliation step reaches the world.

    `agent.py:19` -- the actuator's *commanded* components "appear nowhere in
    `Agent.write` ... so reconciliation fills them and the world reads them".
    Every other boundary stalk is overwritten by the external write at the end
    of the same tick, so this is the only cell outside the 150 predicting ones
    whose gain has an effect, and it is the last step of the return path the
    acceptance demo's onset-latency instrument reads.

    Its maps are pinned at the exact gauge (`||F||_F = 1`), so
    `sum_e ||F||_F^2 = deg(v)` is not a bound training tightens into -- it is an
    equality that holds for the life of the run. The looseness reported here is
    therefore permanent, unlike the interior's, which #142 watched fall
    41.8 -> 5.585 as the gauge headroom was spent.
    """
    from patchworks.graph import CellKind

    maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(seed))
    gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=RHO)
    cell = next(c for c in dome.cells if c.kind is CellKind.ACTUATOR)
    block = torch.zeros(cell.stalk, cell.stalk)
    norms = []
    for edge in dome.edges:
        for side, cid in enumerate((edge.u, edge.v)):
            if cid != cell.id:
                continue
            f = maps.maps[pair_index(edge.id, side)][: edge.m, : cell.stalk].detach()
            block = block + f.T @ f
            norms.append(float(f.norm()))
    lam = float(torch.linalg.eigvalsh(block)[-1])
    denominator = DEFAULT_GAMMA / float(gains[cell.id])
    deg = dome.degrees[cell.id]
    print("\n### the actuator boundary cell, the return path's last step\n")
    print(f"  stalk {cell.stalk}, degree {deg}, sum_e m_e {dome.stalk_sums[cell.id]}")
    print(f"  map Frobenius norms   {['%.3f' % n for n in norms]}  (pinned, exact gauge)")
    print(f"  denominator applied   {denominator:.1f}")
    print(f"  provable bound        {deg:.1f}   (sum_e ||F||_F^2, an equality here)")
    print(f"  true lambda_max       {lam:.4f}")
    print(f"  applied / provable    {denominator / deg:.2f}x   -- permanent, by construction")
    print(f"  applied / true        {denominator / lam:.2f}x   -- at initialisation")


if __name__ == "__main__":
    main()
