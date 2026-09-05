"""A pure-Python toy of the constant-evidence collapse (docs/audit/collapse-mechanism-note.md).

No dependencies. Twelve-dimensional, deliberately naive: it keeps only the parts of the
prediction rule that the argument needs and drops everything else.

What it keeps, and where the real thing is:

* `K` starts at `a * I`                      -- body.py, `CellOperators.__init__` (K = scale * eye)
* the used operator is `K / max(1, sigma_max)` -- body.py `project()` before #466, the forward
                                                  normalisation after it; same used operator either way
* `dL/dK = (D^T e) h^T`, an outer product     -- learning.py `prediction_error`: the prediction is
                                                  `D K h + b`, so the gradient in `K` is rank one, with
                                                  row space along `h`, encode's output for that tick

What it drops: encode's fusion of chart and evidence (here `h` is supplied directly), reconciliation's
feedback into the target, and the biases (except in the `absorb` condition, where a decode bias
takes the persistent part of the error at rate eta, as the real rule would if the target held still).

Run:  python3 prototypes/audit-collapse-toy/collapse_toy.py [--steps 10000]

It prints two tables. The first varies the evidence direction with a white error; the second holds
the evidence direction constant and varies whether the error is persistent and whether a bias
absorbs it. Read the columns against #477's four statistics: sigma_max pinned at ~1, rho falling
3-7x beneath it, stable rank toward 1, non-normality toward its rank-1 ceiling of sqrt(2) (#357).
"""
import argparse
import math
import random

K_DIM, N_DIM = 12, 32


def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def eye(size, scale=1.0):
    m = zeros(size, size)
    for i in range(size):
        m[i][i] = scale
    return m


def matvec(m, v):
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def transpose(m):
    return [list(row) for row in zip(*m)]


def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def frobenius(m):
    return math.sqrt(sum(x * x for row in m for x in row))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def sigma_max(m, iters=60):
    """Largest singular value by power iteration on M^T M."""
    v = [random.gauss(0, 1) for _ in range(len(m[0]))]
    nv = norm(v)
    v = [x / nv for x in v]
    mt = transpose(m)
    for _ in range(iters):
        w = matvec(mt, matvec(m, v))
        nw = norm(w)
        if nw == 0:
            return 0.0
        v = [x / nw for x in w]
    return math.sqrt(norm(matvec(mt, matvec(m, v))))


def spectral_radius(m, power=32):
    """Gelfand's formula, ||M^t||^(1/t) at t = 32, with rescaling at each squaring."""
    p = [row[:] for row in m]
    logscale = 0.0
    t = 1
    while t < power:
        s = sigma_max(p)
        if s == 0:
            return 0.0
        p = [[x / s for x in row] for row in p]
        logscale = 2 * (logscale + math.log(s))
        p = matmul(p, p)
        t *= 2
    s = sigma_max(p)
    if s == 0:
        return 0.0
    return math.exp((logscale + math.log(s)) / t)


def statistics(m):
    """(sigma_max, rho, stable rank, non-normality) of the used operator."""
    s = sigma_max(m)
    f = frobenius(m)
    mt = transpose(m)
    c1 = matmul(mt, m)
    c2 = matmul(m, mt)
    nonnormality = frobenius([[c1[i][j] - c2[i][j] for j in range(K_DIM)] for i in range(K_DIM)]) / (f * f)
    return round(s, 3), round(spectral_radius(m), 3), round(f * f / (s * s), 2), round(nonnormality, 3)


def run(evidence, error, absorb, decode, steps, eta=1e-2, scale=1.0, noise=0.1, checkpoints=(100, 1000, 5000)):
    """One cell, `steps` prediction-rule updates. Returns rows of (t, sigma_max, rho, stable rank, nn)."""
    decode_t = transpose(decode)
    k = eye(K_DIM, scale)
    h0 = [random.gauss(0, 1) for _ in range(K_DIM)]
    nh = norm(h0)
    h0 = [x / nh for x in h0]
    e0 = [random.gauss(0, 1) for _ in range(N_DIM)]
    ne = norm(e0)
    e0 = [0.3 * x / ne for x in e0]  # a persistent offset of norm 0.3, the standing offset's stand-in
    bias = [0.0] * N_DIM
    rows = []
    marks = set(checkpoints) | {steps - 1}
    for t in range(steps):
        if evidence == "constant":
            h = h0
        elif evidence == "rank3":
            h = [random.gauss(0, 1) if i < 3 else 0.0 for i in range(K_DIM)]
        else:
            h = [random.gauss(0, 1) / math.sqrt(K_DIM) for _ in range(K_DIM)]
        if error == "persistent":
            e = [e0[i] - bias[i] + random.gauss(0, noise) for i in range(N_DIM)]
            if absorb:
                for i in range(N_DIM):
                    bias[i] += eta * e[i]
        else:
            e = [random.gauss(0, noise) for _ in range(N_DIM)]
        g = matvec(decode_t, e)  # the error pulled back through the frozen readout
        for i in range(K_DIM):
            gi = eta * g[i]
            for j in range(K_DIM):
                k[i][j] -= gi * h[j]  # the outer product: rank one, row space along h
        if t in marks:
            s = sigma_max(k)
            used = [[x / max(1.0, s) for x in row] for row in k]
            rows.append((t,) + statistics(used))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    decode = [[random.gauss(0, 1) / math.sqrt(K_DIM) for _ in range(K_DIM)] for _ in range(N_DIM)]

    header = "  t      sigma_max   rho     stable_rank   nonnormality"
    print("Table 1 -- white error, the evidence direction varied")
    print("  (a rank-1 ceiling on non-normality is sqrt(2) = 1.414; #477 reads 0.66-0.78 at 100k)")
    for evidence in ("constant", "rank3", "full"):
        print(f"evidence direction: {evidence}")
        print(header)
        for row in run(evidence, "white", False, decode, args.steps):
            print("  {:>6}   {:>7}   {:>6}   {:>9}   {:>10}".format(*row))
    print()
    print("Table 2 -- constant evidence direction, the error's persistence varied")
    for error, absorb in (("persistent", False), ("persistent", True), ("white", False)):
        label = f"error: {error}" + (", a bias absorbs it" if absorb else "")
        print(label)
        print(header)
        for row in run("constant", error, absorb, decode, args.steps):
            print("  {:>6}   {:>7}   {:>6}   {:>9}   {:>10}".format(*row))
    print()
    print("Table 3 -- varying evidence direction, persistent error, no absorption (the control)")
    print(header)
    for row in run("full", "persistent", False, decode, args.steps):
        print("  {:>6}   {:>7}   {:>6}   {:>9}   {:>10}".format(*row))


if __name__ == "__main__":
    main()
