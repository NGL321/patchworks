"""Regional Jacobian spectra of a shared frozen cell body (patchworks#27).

The go/no-go rig `docs/spec/05-timescales.md` asks for: sample the bias vectors cells
occupy, measure each cell's regional Jacobian, and look at the distribution of spectral
radii. If it is a spike, timescale-by-persistence is dead.

The body here is a stand-in, not the body: iid Gaussian ReLU MLPs at `k = 12`, `n = 32`
(`06-graph-topology.md`), since nothing else about the body is specified yet. The
regional Jacobian measured is `d chart_{t+1} / d chart_t` for one cell — the chart's own
round trip through `encode` then `step`, which is what has to fail to contract for a
cell to hold state.

Findings are written up in `docs/research/027-regional-jacobian-spectra.md`.

    python prototypes/regional-spectra/spread_pilot.py
"""

import numpy as np

K, N = 12, 32


def make_body(rng, widths, sw2):
    """The shared frozen weights: `encode: [k+n] -> k` and `step: k -> k`, ReLU MLPs."""

    def mlp(din, dout, hidden):
        dims = [din] + hidden + [dout]
        return [
            rng.normal(0, np.sqrt(sw2 / dims[i]), size=(dims[i + 1], dims[i]))
            for i in range(len(dims) - 1)
        ]

    return mlp(K + N, K, widths), mlp(K, K, widths)


def jac(weights, biases, x):
    """Forward pass through a ReLU MLP, carrying the Jacobian of the active region."""
    jacobian = np.eye(weights[0].shape[1])
    h = x
    for i, (w, b) in enumerate(zip(weights, biases)):
        h = w @ h + b
        jacobian = w @ jacobian
        if i < len(weights) - 1:
            active = (h > 0).astype(float)
            h = h * active
            jacobian = active[:, None] * jacobian
    return h, jacobian


def regional_jacobians(rng, widths, sw2, sb2, ncells=150, vary="both", sigma_state=1.0):
    """One frozen body; `ncells` cells differing in bias, operating point, or both."""
    w_enc, w_step = make_body(rng, widths, sw2)
    biases = lambda ws: [rng.normal(0, np.sqrt(sb2), size=w.shape[0]) for w in ws]
    b0_enc, b0_step = biases(w_enc), biases(w_step)
    c0, s0 = rng.normal(0, sigma_state, size=K), rng.normal(0, sigma_state, size=N)

    out = []
    for _ in range(ncells):
        b_enc, b_step = (biases(w_enc), biases(w_step)) if vary in ("bias", "both") else (b0_enc, b0_step)
        if vary in ("state", "both"):
            c = rng.normal(0, sigma_state, size=K)
            s = rng.normal(0, sigma_state, size=N)
        else:
            c, s = c0, s0
        chart, j_enc = jac(w_enc, b_enc, np.concatenate([c, s]))
        _, j_step = jac(w_step, b_step, chart)
        out.append(j_step @ j_enc[:, :K])
    return out


def stats(jacobians):
    rho = np.array([np.abs(np.linalg.eigvals(j)).max() for j in jacobians])
    tau = -1.0 / np.log(np.clip(rho[rho < 1.0], 1e-12, 0.999999))
    log_rho = np.log10(np.clip(rho, 1e-12, None))
    return {
        "rho_p05": np.quantile(rho, 0.05),
        "rho_med": np.median(rho),
        "rho_p95": np.quantile(rho, 0.95),
        "sd_log10_rho": log_rho.std(),
        "tau_p05": np.quantile(tau, 0.05),
        "tau_med": np.median(tau),
        "tau_p95": np.quantile(tau, 0.95),
        "tau_ratio": np.quantile(tau, 0.95) / np.quantile(tau, 0.05),
        "unstable": float(np.mean(rho >= 1.0)),
    }


def line(tag, s):
    print(
        f"{tag:40s} rho p05/med/p95 = {s['rho_p05']:6.3f} {s['rho_med']:6.3f} {s['rho_p95']:6.3f}  "
        f"sd(log10)={s['sd_log10_rho']:5.2f}  tau p05/med/p95 = {s['tau_p05']:6.2f} {s['tau_med']:7.2f} "
        f"{s['tau_p95']:8.2f}  ratio={s['tau_ratio']:6.1f}  unstable={s['unstable']:.2f}"
    )


def main():
    print("=== width: narrow layers widen the spread (Hanin's beta), sw2=1.7, sb2=0.5 ===")
    for widths in ([12, 12], [32], [64, 64], [128, 128, 128]):
        s = stats(regional_jacobians(np.random.default_rng(0), widths, 1.7, 0.5))
        line(f"widths={widths}", s)

    print("\n=== sigma_w^2: spread and stability are the same knob (widths=[64,64], sb2=0.5) ===")
    for sw2 in (1.2, 1.5, 1.7, 1.9, 2.0):
        s = stats(regional_jacobians(np.random.default_rng(7), [64, 64], sw2, 0.5))
        line(f"sw2={sw2}", s)

    print("\n=== sigma_b^2 barely moves it at frozen weights (widths=[12,12], sw2=1.7) ===")
    for sb2 in (0.01, 0.1, 0.5, 2.0, 8.0):
        s = stats(regional_jacobians(np.random.default_rng(7), [12, 12], 1.7, sb2))
        line(f"sb2={sb2}", s)

    print("\n=== what varies across cells: bias alone reproduces the whole spread ===")
    for widths, sw2 in (([12, 12], 1.7), ([64, 64], 1.9), ([32], 1.5)):
        for vary in ("bias", "state", "both"):
            s = stats(regional_jacobians(np.random.default_rng(11), widths, sw2, 0.5, vary=vary))
            line(f"widths={widths} sw2={sw2} vary={vary}", s)

    print("\n=== non-normality: rho understates one-tick amplification (widths=[64,64], sw2=1.9) ===")
    js = regional_jacobians(np.random.default_rng(1), [64, 64], 1.9, 0.5)
    rho = np.array([np.abs(np.linalg.eigvals(j)).max() for j in js])
    smax = np.array([np.linalg.svd(j, compute_uv=False)[0] for j in js])
    eight = np.array([np.linalg.norm(np.linalg.matrix_power(j, 8), 2) ** (1 / 8) for j in js])
    ok = rho > 1e-12
    print(f"  sigma_max / rho: median={np.median(smax[ok] / rho[ok]):.2f}  "
          f"||J^8||^(1/8) / rho: median={np.median(eight[ok] / rho[ok]):.2f} "
          f"p95={np.quantile(eight[ok] / rho[ok], 0.95):.2f}")

    print("\n=== how many cells the estimate needs (widths=[64,64], sw2=1.9) ===")
    for m in (20, 50, 150, 600):
        ests = [
            stats(regional_jacobians(np.random.default_rng(100 + seed), [64, 64], 1.9, 0.5, ncells=m))["sd_log10_rho"]
            for seed in range(8)
        ]
        print(f"  m={m:4d}  sd(log10 rho): mean={np.mean(ests):.3f}  across-seed sd={np.std(ests):.3f}")


if __name__ == "__main__":
    main()
