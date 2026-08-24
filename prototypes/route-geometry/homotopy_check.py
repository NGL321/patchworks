"""Does the sandbox actually generate tasks with two distinct route homotopy classes?

For each sampled task, take the goal puck's start and the goal zone's centre.
The pedestal (radius 0.08) inflated by the puck's radius is the forbidden disk.
- Does the straight segment start->goal intersect that disk? (= route must wrap)
- If so, how long is each of the two wrapping routes (tangent-arc-tangent), and
  how close in length are they? (near-equal = a genuine tie, the blending case)
"""
import sys, math
import numpy as np

sys.path.insert(0, "/Users/angl/Documents/patchworks/prototypes/sandbox")
from sandbox_env import PlanarPushSandbox, ZONE_XY  # noqa

PED = 0.08
PUCK_R = [0.035, 0.045, 0.055]


def seg_hits_disk(p, q, R):
    d = q - p
    t = np.clip(-p @ d / (d @ d), 0.0, 1.0)
    return np.linalg.norm(p + t * d) < R


def wrap_lengths(p, q, R):
    """Tangent-arc-tangent path length around a disk of radius R at origin, both ways."""
    rp, rq = np.linalg.norm(p), np.linalg.norm(q)
    if rp <= R or rq <= R:
        return None
    tp, tq = math.sqrt(rp**2 - R**2), math.sqrt(rq**2 - R**2)
    ap, aq = math.atan2(p[1], p[0]), math.atan2(q[1], q[0])
    # angle from centre-line to tangent point
    bp, bq = math.acos(R / rp), math.acos(R / rq)
    out = []
    for s in (+1, -1):
        # tangent points at ap + s*bp and aq - s*bq; arc between them, going in direction s
        a1, a2 = ap + s * bp, aq - s * bq
        arc = (a2 - a1) * s
        arc = arc % (2 * math.pi)
        out.append(tp + tq + R * arc)
    return out  # [ccw-ish, cw-ish]


def main(n=400):
    rows = []
    for split in ("train", "heldout"):
        env = PlanarPushSandbox(split=split, seed=7, render_obs=False)
        for _ in range(n):
            t = env.sample_task()
            p = np.asarray(t.puck_xy[t.goal_puck][:2], dtype=float)
            q = ZONE_XY[t.goal_zone].astype(float)
            R = PED + PUCK_R[t.goal_puck]
            hit = seg_hits_disk(p, q, R)
            L = wrap_lengths(p, q, R)
            straight = float(np.linalg.norm(q - p))
            rows.append((split, t.goal_puck, t.goal_zone, straight, hit, L))
        env.close()

    for split in ("train", "heldout"):
        rs = [r for r in rows if r[0] == split]
        hits = [r for r in rs if r[4]]
        print(f"[{split}] {len(hits)}/{len(rs)} tasks = {100*len(hits)/len(rs):.1f}% "
              f"need to route around the pedestal")
        if hits:
            ratios, detours = [], []
            for _, _, _, straight, _, L in hits:
                if L is None:
                    continue
                short, long_ = min(L), max(L)
                ratios.append(long_ / short)
                detours.append(short / straight)
            ratios, detours = np.array(ratios), np.array(detours)
            print(f"    detour cost (shortest wrap / straight): median {np.median(detours):.3f}, "
                  f"max {detours.max():.3f}")
            print(f"    asymmetry (longer wrap / shorter wrap): median {np.median(ratios):.2f}, "
                  f"min {ratios.min():.2f}")
            near_tie = (ratios < 1.15).sum()
            print(f"    near-ties (<15% apart, i.e. genuinely ambiguous): "
                  f"{near_tie}/{len(ratios)} = {100*near_tie/len(ratios):.1f}% of wrapping tasks "
                  f"({100*near_tie/len(rs):.1f}% of all tasks)")


if __name__ == "__main__":
    main()
