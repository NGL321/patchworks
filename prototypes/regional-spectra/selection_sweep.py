"""Draw-then-select construction for the body's timescales (patchworks#42).

`spread_pilot.py` (#27) established that regional Jacobian spectra spread, and found the
conflict this script exists to resolve: every `sigma_w^2` that bought a usable tau ratio
also put a material fraction of regions past `rho = 1`. Three measurements take that
apart, and `05-timescales.md` cites all three.

1. `sweep_widths` -- separate encode/step widths plus a Hanin & Rolnick fold-margin
   column. The margin follows the *narrower* map, so an encode wide enough for #49's
   minimum costs nothing if `step` stays narrow.
2. `trajectory_lambda` -- the realised contraction rate along a trajectory, against the
   per-region rho distribution. A region with `rho >= 1` is not an unstable cell; what
   has to contract is the product. Resampling the operating point every tick is the
   no-dwell extreme, and flatters lambda -- read it as the bound it is.
3. `tail_reachability` -- how far up the slow tail selection can reach, at what
   acceptance rate, and whether selecting a cell slow costs it fold margin (it does not).

The body is still a stand-in: iid Gaussian ReLU MLPs at `k = 12`, `n = 32`. These
establish the shape of the construction, not the body's numbers.

    python prototypes/regional-spectra/selection_sweep.py
"""

import numpy as np

from spread_pilot import K, N, jac

CONFIGS = [
    ("[12,12]/[12,12]", [12, 12], [12, 12]),
    ("[45]/[13]", [45], [13]),
    ("[48,48]/[16,16]", [48, 48], [16, 16]),
    ("[64,64]/[64,64]", [64, 64], [64, 64]),
    ("[128]/[32]", [128], [32]),
]


def make_body(rng, enc_hidden, step_hidden, sw2):
    """As `spread_pilot.make_body`, but `encode` and `step` size independently."""

    def mlp(din, dout, hidden):
        dims = [din] + hidden + [dout]
        return [
            rng.normal(0, np.sqrt(sw2 / dims[i]), size=(dims[i + 1], dims[i]))
            for i in range(len(dims) - 1)
        ]

    return mlp(K + N, K, enc_hidden), mlp(K, K, step_hidden)


def fold_margin(weights, biases, x):
    """Distance to the nearest region boundary: `min_i |z_i| / ||grad z_i||`.

    Hanin & Rolnick's quantity, the construction-time proxy for region dwell.
    """
    jacobian = np.eye(weights[0].shape[1])
    h = x
    margins = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        h = w @ h + b
        jacobian = w @ jacobian
        if i < len(weights) - 1:
            grad = np.linalg.norm(jacobian, axis=1)
            margins.append(np.min(np.abs(h) / np.maximum(grad, 1e-12)))
            active = (h > 0).astype(float)
            h = h * active
            jacobian = active[:, None] * jacobian
    return min(margins) if margins else np.inf


def _biases(rng, weights, sb2):
    return [rng.normal(0, np.sqrt(sb2), size=w.shape[0]) for w in weights]


def _cell(rng, body, b_enc, b_step, sigma_state):
    """One cell at one operating point: its regional Jacobian and its fold margin."""
    w_enc, w_step = body
    x = np.concatenate([rng.normal(0, sigma_state, size=K), rng.normal(0, sigma_state, size=N)])
    chart, j_enc = jac(w_enc, b_enc, x)
    _, j_step = jac(w_step, b_step, chart)
    margin = min(fold_margin(w_enc, b_enc, x), fold_margin(w_step, b_step, chart))
    return j_step @ j_enc[:, :K], margin


def sweep_widths(enc, step, sw2, sb2=0.5, ncells=400, seed=3, sigma_state=1.0):
    """Across-cell rho and fold margin for one frozen body."""
    rng = np.random.default_rng(seed)
    body = make_body(rng, enc, step, sw2)
    rho, margins = [], []
    for _ in range(ncells):
        j, margin = _cell(rng, body, _biases(rng, body[0], sb2), _biases(rng, body[1], sb2), sigma_state)
        rho.append(np.abs(np.linalg.eigvals(j)).max())
        margins.append(margin)
    rho, margins = np.array(rho), np.array(margins)
    tau = -1.0 / np.log(np.clip(rho[rho < 1.0], 1e-12, 0.999999))
    return {
        "rho_p05": np.quantile(rho, 0.05), "rho_med": np.median(rho),
        "rho_p95": np.quantile(rho, 0.95), "rho_max": rho.max(),
        "tau_ratio": np.quantile(tau, 0.95) / np.quantile(tau, 0.05),
        "tau_p95": np.quantile(tau, 0.95),
        "unstable": float((rho >= 1.0).mean()),
        "margin_med": np.median(margins),
    }


def trajectory_lambda(enc, step, sw2, sb2=0.5, ncells=200, ticks=64, seed=5, sigma_state=1.0):
    """Realised contraction `lambda` per cell, against its own per-region rho draws.

    Biases fixed per cell; the operating point resampled every tick. That is the
    fastest-mixing case -- zero region dwell -- so the lambda it reports is the most
    favourable one available, and the spread it reports the least.
    """
    rng = np.random.default_rng(seed)
    body = make_body(rng, enc, step, sw2)
    rows = []
    for _ in range(ncells):
        b_enc, b_step = _biases(rng, body[0], sb2), _biases(rng, body[1], sb2)
        product, logs, rho = np.eye(K), [], []
        for _t in range(ticks):
            j, _ = _cell(rng, body, b_enc, b_step, sigma_state)
            rho.append(np.abs(np.linalg.eigvals(j)).max())
            product = j @ product
            norm = np.linalg.norm(product, 2)
            # Renormalise every tick: the raw product under/overflows well before 64.
            if norm > 0:
                logs.append(np.log(norm))
                product = product / norm
            else:
                logs.append(-50.0)
                product = np.eye(K)
        rho = np.array(rho)
        rows.append((sum(logs) / ticks, float((rho >= 1.0).mean())))
    lam = np.array([r[0] for r in rows])
    expansive = np.array([r[1] for r in rows])
    tau = np.where(lam < 0, -1.0 / lam, np.inf)
    return {
        "expansive_regions": expansive.mean(),
        "cells_any_expansive": float((expansive > 0).mean()),
        "lam_med": np.median(lam), "divergent": float((lam > 0).mean()),
        "tau_med": np.median(tau),
        "tau_ratio": np.quantile(tau, 0.95) / np.quantile(tau, 0.05),
    }


def tail_reachability(enc, step, sw2, sb2=0.5, ndraws=20000, seed=17, sigma_state=1.0):
    """How thick the slow tail is, and whether reaching it costs fold margin."""
    rng = np.random.default_rng(seed)
    body = make_body(rng, enc, step, sw2)
    rho, margins = [], []
    for _ in range(ndraws):
        j, margin = _cell(rng, body, _biases(rng, body[0], sb2), _biases(rng, body[1], sb2), sigma_state)
        rho.append(np.abs(np.linalg.eigvals(j)).max())
        margins.append(margin)
    rho, margins = np.array(rho), np.array(margins)
    slow, fast = rho >= np.quantile(rho, 0.99), rho <= np.quantile(rho, 0.5)
    return {
        "accept": {thr: float((rho >= thr).mean()) for thr in (0.90, 0.95, 0.98, 0.99, 1.0)},
        "corr_rho_margin": float(np.corrcoef(np.log(np.clip(rho, 1e-12, None)),
                                             np.log(np.clip(margins, 1e-12, None)))[0, 1]),
        "margin_slowest_1pct": np.median(margins[slow]),
        "margin_fastest_50pct": np.median(margins[fast]),
    }


def main():
    print("=== 1. widths: the fold margin follows the narrower map (sb2=0.5) ===")
    for name, enc, step in CONFIGS:
        for sw2 in (1.2, 1.4, 1.6, 1.8, 2.0):
            s = sweep_widths(enc, step, sw2)
            print(f"  {name:18s} sw2={sw2}  rho p05/med/p95/max = {s['rho_p05']:.3f} {s['rho_med']:.3f} "
                  f"{s['rho_p95']:.3f} {s['rho_max']:.3f}  tau_ratio={s['tau_ratio']:6.1f} "
                  f"tau_p95={s['tau_p95']:7.2f}  unstable={s['unstable']:.3f}  margin={s['margin_med']:.4f}")

    print("\n=== 2. a region with rho >= 1 is not an unstable cell (no-dwell extreme) ===")
    for name, enc, step in (CONFIGS[1], CONFIGS[0]):
        for sw2 in (1.6, 1.8, 2.0):
            s = trajectory_lambda(enc, step, sw2)
            print(f"  {name:18s} sw2={sw2}  expansive regions={s['expansive_regions']:.3f} "
                  f"(cells touching one: {s['cells_any_expansive']:.2f})  lambda_med={s['lam_med']:+.3f}  "
                  f"cells with lambda>0={s['divergent']:.3f}  realised tau_med={s['tau_med']:.2f} "
                  f"ratio={s['tau_ratio']:.1f}")

    print("\n=== 3. the slow tail is reachable, and reaching it costs no fold margin ===")
    for name, enc, step, sw2 in (("[128]/[32]", [128], [32], 1.4), ("[45]/[13]", [45], [13], 1.2)):
        s = tail_reachability(enc, step, sw2)
        acc = "  ".join(f"rho>={t}: {100 * v:.3f}%" for t, v in s["accept"].items())
        print(f"  {name:12s} sw2={sw2}  {acc}")
        print(f"  {'':12s} corr(log rho, log margin)={s['corr_rho_margin']:+.3f}  "
              f"margin slowest-1%={s['margin_slowest_1pct']:.4f} vs fastest-50%={s['margin_fastest_50pct']:.4f}")


if __name__ == "__main__":
    main()
