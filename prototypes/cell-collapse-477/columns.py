import sys, json, collections
sys.path.insert(0, 'src')
import numpy as np
from patchworks.graph import DEFAULT_SPEC, build_graph, CellKind

dome = build_graph(DEFAULT_SPEC)
pred = list(dome.predicting)
n = len(pred)
lvl = np.array([int(dome.cells[c].index.level) for c in pred], float)
col = [str(dome.cells[c].index.column) for c in pred]
deg = np.array([int(dome.degrees[c]) for c in pred], float)
pv = np.array([int(dome.private_dimensions[i]) for i in range(n)], float)
soma = np.array([1.0 if c == 'somatomotor' else 0.0 for c in col])
apex = (lvl == 7).astype(float)

data = {s: json.load(open(
    'prototypes/chart-per-domain-132/132-postfloor-real-train-seed%d-100000.json' % s))
    for s in (42, 43, 44)}

print("=== column x level census ===")
cen = collections.Counter(zip(col, [int(x) for x in lvl]))
for k in sorted(cen, key=lambda t: (t[1], t[0])):
    print("   L%d %-12s %3d" % (k[1], k[0], cen[k]))

print()
print("=== dead-rate by column at 100k, pooled over 3 seeds ===")
dead = {s: set(np.where(np.array(data[s]['checkpoints'][-1]['per_cell']['modes_retaining']) == 0)[0].tolist())
        for s in (42, 43, 44)}
for cname in sorted(set(col)):
    members = [i for i in range(n) if col[i] == cname]
    d = sum(1 for s in dead for i in dead[s] if col[i] == cname)
    print("   %-12s cells %3d   dead %3d   rate %6.3f" % (cname, len(members), d, d / (3 * len(members))))

print()
print("=== somatomotor vs vision within L1 ===")
for cname in ('vision', 'somatomotor'):
    m = [i for i in range(n) if col[i] == cname and lvl[i] == 1]
    d = sum(1 for s in dead for i in dead[s] if i in m)
    print("   L1 %-12s cells %2d  dead %2d  rate %6.3f" % (cname, len(m), d, d / (3 * len(m))))

print()
print("=== R^2 of construction against the 100k collapse (the #233 standard) ===")


def r2(X, y):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot


for s in (42, 43, 44):
    rho = np.array(data[s]['checkpoints'][-1]['per_cell']['rho_K'])
    y = np.log(np.maximum(rho, 1e-6))
    mr = np.array(data[s]['checkpoints'][-1]['per_cell']['modes_retaining'], float)
    print(" seed %d" % s)
    print("    log rho ~ level                      R2 = %.3f" % r2([lvl], y))
    print("    log rho ~ level + degree + p_v       R2 = %.3f" % r2([lvl, deg, pv], y))
    print("    log rho ~ apex + somatomotor         R2 = %.3f" % r2([apex, soma], y))
    print("    log rho ~ apex + soma + lvl + deg+pv R2 = %.3f" % r2([apex, soma, lvl, deg, pv], y))
    print("    modes_retaining ~ apex + somatomotor R2 = %.3f" % r2([apex, soma], mr))

print()
print("=== shape statistics at 100k, by class (pooled seeds) ===")
classes = {
    'apex L7': [i for i in range(n) if lvl[i] == 7],
    'somatomotor L1/L2': [i for i in range(n) if col[i] == 'somatomotor' and lvl[i] <= 2],
    'vision L1': [i for i in range(n) if col[i] == 'vision' and lvl[i] == 1],
    'core L3-L6': [i for i in range(n) if 3 <= lvl[i] <= 6],
}
for key in ('rho_K', 'stable_rank', 'nonnormality', 'effective_rank', 'sigma_min'):
    print("  %s" % key)
    for cname, m in classes.items():
        vals = np.concatenate([np.array(data[s]['checkpoints'][-1]['per_cell'][key])[m]
                               for s in (42, 43, 44)])
        print("     %-20s median %8.4f   min %8.4f   max %8.4f"
              % (cname, np.median(vals), vals.min(), vals.max()))
