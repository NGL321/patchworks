"""PROTOTYPE — throwaway. Measures precedence depth for issue #60.

    ../../.venv-proto/bin/python precedence_probe.py

Two halves, matching the ticket:

  A. Per-joint gearing — is the arm ALREADY heterogeneous in timescale, and does
     `gear` move the quantity we actually mean by "timescale"?
  B. Route-blocking layouts — how often does a blocker land across the route by
     accident, what does a blocking test cost, and is the induced order real?
"""
import math
import sys

import mujoco
import numpy as np

from sandbox_env import (N_PUCKS, N_ZONES, PlanarPushSandbox, SPAWN_R,
                         ZONE_RADIUS, ZONE_XY)

XML = __import__("sandbox_env").XML
PUCK_R = [0.035, 0.045, 0.055]
PADDLE = 0.03
PED = 0.08


def fresh():
    m = mujoco.MjModel.from_xml_path(XML)
    return m, mujoco.MjData(m)


def jadr(m):
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("j0", "j1", "j2")]
    return [m.jnt_qposadr[i] for i in ids], [m.jnt_dofadr[i] for i in ids]


# Poses to average over: the arm's timescale is configuration-dependent, so a
# single pose would be an artefact.
POSES = [(0.0, 0.0, 0.0),        # straight out: max inertia at the shoulder
         (0.0, 1.3, 1.3),        # folded
         (0.6, -1.1, -0.9),      # the IK seed the dynamics probe uses
         (0.0, 2.2, -2.2)]       # doubled back


# ------------------------------------------------------------ A1. passive decay
def decay_tau(m, d, j, pose, v0=1.0, seconds=3.0):
    """Impulse joint j, fit an exponential to |qvel_j|, return the time constant.

    The fit degenerates in one pose: a fast joint in a folded arm gets re-driven by
    coupling from the slower ones, and the slope comes back positive (a negative tau).
    That is real physics and a bad instrument, so read the MEDIAN across poses, never
    a single pose.
    """
    qadr, dofadr = jadr(m)
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    for k in range(3):
        d.qpos[qadr[k]] = pose[k]
    # park the pucks far away so no contact contaminates the decay
    for i in range(N_PUCKS):
        a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_x")]
        d.qpos[a:a + 2] = [0.45, 0.45 - 0.12 * i]
    d.qvel[dofadr[j]] = v0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)

    ts, vs = [], []
    n = int(seconds / m.opt.timestep)
    for i in range(n):
        mujoco.mj_step(m, d)
        v = abs(d.qvel[dofadr[j]])
        if v < 1e-4 * v0:
            break
        ts.append(d.time)
        vs.append(v)
    ts, vs = np.array(ts), np.array(vs)
    keep = vs > 0.05 * v0            # fit the first decade only: it is not a pure exponential
    if keep.sum() < 10:
        return float("nan")
    slope = np.polyfit(ts[keep], np.log(vs[keep]), 1)[0]
    return float(-1.0 / slope)


def a1_passive():
    print("\nA1. passive decay time constants (impulse each joint, fit |qvel|)")
    m, d = fresh()
    qadr, dofadr = jadr(m)
    print(f"    damping = {[float(m.dof_damping[a]) for a in dofadr]}, "
          f"armature = {[float(m.dof_armature[a]) for a in dofadr]}")
    table = []
    for pose in POSES:
        taus = [decay_tau(m, d, j, pose) for j in range(3)]
        table.append(taus)
        print(f"    pose {pose}:  tau = "
              + ", ".join(f"j{j} {t*1000:6.1f} ms" for j, t in enumerate(taus)))
    table = np.array(table)
    med = np.median(table, axis=0)
    print(f"    median tau: j0 {med[0]*1000:.1f} ms, j1 {med[1]*1000:.1f} ms, "
          f"j2 {med[2]*1000:.1f} ms")
    print(f"    LADDER RATIO (slowest / fastest) = {med.max()/med.min():.2f}x   "
          f"[j0/j2 = {med[0]/med[2]:.2f}x]")
    print(f"    spread across poses for a single joint: "
          + ", ".join(f"j{j} {table[:,j].max()/table[:,j].min():.2f}x" for j in range(3)))
    return med


# ---------------------------------------- A2. effective inertia, and what gear does
def a2_inertia_and_gear():
    print("\nA2. effective inertia M_ii, and what `gear` actually moves")
    m, d = fresh()
    qadr, dofadr = jadr(m)
    for pose in POSES:
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        for k in range(3):
            d.qpos[qadr[k]] = pose[k]
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        mii = [M[a, a] for a in dofadr]
        tau_pred = [mii[k] / m.dof_damping[dofadr[k]] for k in range(3)]
        print(f"    pose {pose}:  M_ii = "
              + ", ".join(f"{v:.5f}" for v in mii)
              + "   M/b = " + ", ".join(f"{t*1000:.0f} ms" for t in tau_pred))

    print("\n    gear invariance check — passive decay under gear 1 vs gear 4 on j0:")
    base = decay_tau(m, d, 0, POSES[2])
    m2, d2 = fresh()
    m2.actuator_gear[0, 0] = 4.0
    geared = decay_tau(m2, d2, 0, POSES[2])
    print(f"      tau(j0) gear=1: {base*1000:.1f} ms    gear=4: {geared*1000:.1f} ms")
    print("      (gear multiplies ctrl -> joint torque; it is NOT in the passive M/b decay)")

    print("\n    driven timescale — time to sweep 0.5 rad from rest at full ctrl:")
    for gear in (1.0, 2.0, 4.0):
        row = []
        for j in range(3):
            mg, dg = fresh()
            mg.actuator_gear[j, 0] = gear
            qa, da = jadr(mg)
            dg.qpos[:] = 0.0
            dg.qvel[:] = 0.0
            for k in range(3):
                dg.qpos[qa[k]] = POSES[2][k]
            for i in range(N_PUCKS):
                a = mg.jnt_qposadr[mujoco.mj_name2id(mg, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_x")]
                dg.qpos[a:a + 2] = [0.45, 0.45 - 0.12 * i]
            mujoco.mj_forward(mg, dg)
            q0 = dg.qpos[qa[j]]
            dg.ctrl[:] = 0.0
            dg.ctrl[j] = mg.actuator_ctrlrange[j, 1]
            t_hit = float("nan")
            for _ in range(int(5.0 / mg.opt.timestep)):
                mujoco.mj_step(mg, dg)
                if abs(dg.qpos[qa[j]] - q0) > 0.5:
                    t_hit = dg.time
                    break
            row.append(t_hit)
        print(f"      gear={gear:g}: " + ", ".join(
            f"j{j} {row[j]*1000:6.1f} ms" if row[j] == row[j] else f"j{j}  never" for j in range(3)))


# ----------------------------------------------- A3. torque sufficiency at the tip
def a3_tip_force(gears=(1.0, 1.0, 1.0)):
    m, d = fresh()
    for j in range(3):
        m.actuator_gear[j, 0] = gears[j]
    qadr, dofadr = jadr(m)
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    worst = math.inf
    for pose in POSES:
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        for k in range(3):
            d.qpos[qadr[k]] = pose[k]
        mujoco.mj_forward(m, d)
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jac, None, tip)
        J = jac[:2][:, dofadr]                      # 2 x 3
        tau_max = np.array([m.actuator_ctrlrange[j, 1] * gears[j] for j in range(3)])
        # smallest tip force achievable in the worst direction: min over unit dirs of
        # max |f| s.t. |J^T f| <= tau_max  -> for direction u, f = c*u needs |J^T u| c <= tau
        best = math.inf
        for a in np.linspace(0, math.pi, 36):
            u = np.array([math.cos(a), math.sin(a)])
            need = np.abs(J.T @ u)
            cs = [tau_max[k] / need[k] for k in range(3) if need[k] > 1e-6]
            if not cs:
                continue
            best = min(best, min(cs))
        worst = min(worst, best)
    return worst


def a3():
    print("\nA3. worst-direction tip force available (needs > 2 N for the heaviest puck)")
    for gears in ((1, 1, 1), (2, 1, 0.5), (3, 1, 0.5), (2, 1.5, 1)):
        print(f"    gear {gears}: {a3_tip_force(gears):.2f} N")


# ================================================================ B. route blocking
def seg_clips_disk(p, q, c, R):
    """Does segment p->q pass within R of point c?"""
    d = q - p
    t = np.clip((c - p) @ d / (d @ d + 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p + t * d - c)) < R


def blockers(task, corridor_slack=0.0):
    """Which non-target pucks sit across the target's straight route to its zone."""
    p = np.asarray(task.puck_xy[task.goal_puck], dtype=float)
    q = ZONE_XY[task.goal_zone].astype(float)
    out = []
    for i in range(N_PUCKS):
        if i == task.goal_puck:
            continue
        R = PUCK_R[i] + PUCK_R[task.goal_puck] + corridor_slack
        if seg_clips_disk(p, q, np.asarray(task.puck_xy[i], dtype=float), R):
            out.append(i)
    return out


def standoff_blockers(task):
    """Which non-target pucks sit where the PADDLE must stand to push straight."""
    p = np.asarray(task.puck_xy[task.goal_puck], dtype=float)
    q = ZONE_XY[task.goal_zone].astype(float)
    u = (q - p) / (np.linalg.norm(q - p) + 1e-12)
    stand = p - u * (PUCK_R[task.goal_puck] + PADDLE)
    out = []
    for i in range(N_PUCKS):
        if i == task.goal_puck:
            continue
        if np.linalg.norm(stand - np.asarray(task.puck_xy[i], dtype=float)) < PUCK_R[i] + PADDLE:
            out.append(i)
    return out


def b1_incidental(n=4000):
    print("\nB1. incidental blocking under the CURRENT sampler")
    for split in ("train", "heldout_pair", "heldout_sector"):
        env = PlanarPushSandbox(split=split, seed=7, render_obs=False)
        route = stand = both = 0
        for _ in range(n):
            t = env.sample_task()
            r = bool(blockers(t))
            s = bool(standoff_blockers(t))
            route += r
            stand += s
            both += (r or s)
        env.close()
        print(f"    [{split}] route-clipped {100*route/n:5.1f}%   "
              f"standoff-occupied {100*stand/n:5.1f}%   either {100*both/n:5.1f}%")


def b2_cost(n=2000):
    print("\nB2. cost of rejection-sampling a blocked layout")
    env = PlanarPushSandbox(split="train", seed=3, render_obs=False)
    draws = []
    for _ in range(n):
        k = 0
        while True:
            k += 1
            t = env.sample_task()
            if blockers(t):
                break
            if k > 2000:
                break
        draws.append(k)
    env.close()
    draws = np.array(draws)
    print(f"    mean draws to find a route-clipped layout: {draws.mean():.1f}  "
          f"(median {np.median(draws):.0f}, p95 {np.percentile(draws,95):.0f}, max {draws.max()})")

    print("\n    how deep in the corridor does an accidental blocker sit?")
    env = PlanarPushSandbox(split="train", seed=5, render_obs=False)
    depths = []
    for _ in range(4000):
        t = env.sample_task()
        p = np.asarray(t.puck_xy[t.goal_puck], dtype=float)
        q = ZONE_XY[t.goal_zone].astype(float)
        for i in blockers(t):
            c = np.asarray(t.puck_xy[i], dtype=float)
            d = q - p
            s = np.clip((c - p) @ d / (d @ d), 0.0, 1.0)
            perp = float(np.linalg.norm(p + s * d - c))
            R = PUCK_R[i] + PUCK_R[t.goal_puck]
            depths.append((R - perp) / R)          # 1 = dead centre, 0 = grazing
    env.close()
    depths = np.array(depths)
    print(f"    overlap depth (1 = dead centre, 0 = graze): median {np.median(depths):.2f}, "
          f"fraction > 0.5: {100*(depths>0.5).mean():.0f}%")


# ------------------------------------ B3. does a blocker actually change the outcome?
def _blocked_now(env, info, slack=0.0):
    """Route-clip test against the LIVE state, not the sampled layout."""
    poses = info["puck_pose"]
    gp = info["goal_puck"]
    p = poses[gp, :2]
    q = ZONE_XY[info["goal_zone"]]
    out = []
    for i in range(N_PUCKS):
        if i == gp:
            continue
        R = PUCK_R[i] + PUCK_R[gp] + slack
        if seg_clips_disk(p, q, poses[i, :2], R):
            out.append(i)
    return out


def depth_of(task):
    """Max normalised corridor overlap over the non-target pucks (0 = graze, 1 = centred)."""
    p = np.asarray(task.puck_xy[task.goal_puck], dtype=float)
    q = ZONE_XY[task.goal_zone].astype(float)
    d = q - p
    best = 0.0
    for i in range(N_PUCKS):
        if i == task.goal_puck:
            continue
        c = np.asarray(task.puck_xy[i], dtype=float)
        s = np.clip((c - p) @ d / (d @ d), 0.0, 1.0)
        perp = float(np.linalg.norm(p + s * d - c))
        R = PUCK_R[i] + PUCK_R[task.goal_puck]
        best = max(best, (R - perp) / R)
    return best


def b3_achievability(n_each=16, cap=3000, deep=0.5):
    """Scripted-controller solve rate on clear vs deeply-blocked layouts."""
    from watch import ScriptedPusher
    import time
    print(f"\nB3. scripted solve rate, clear vs blocked (depth > {deep}), "
          f"{n_each} tasks each, {cap*0.02:.0f} s cap")
    env = PlanarPushSandbox(split="any", seed=11, render_obs=False)
    policy = ScriptedPusher(env)

    buckets = {"clear": [], "blocked": []}
    tasks = {"clear": [], "blocked": []}
    while len(tasks["clear"]) < n_each or len(tasks["blocked"]) < n_each:
        t = env.sample_task()
        d = depth_of(t)
        k = "blocked" if d > deep else ("clear" if d <= 0.0 else None)
        if k and len(tasks[k]) < n_each:
            tasks[k].append(t)

    obs, info = env.reset(seed=11)
    t0 = time.time()
    detail = {"clear": [], "blocked": []}
    for k in ("clear", "blocked"):
        for t in tasks[k]:
            obs, info = env.reset(options={"task": t})
            d0 = info["goal_distance"]
            still_blocked_at_end = None
            done = False
            for i in range(cap):
                obs, _, _, _, info = env.step(policy(info))
                if info["goal_satisfied"]:
                    done = True
                    break
            still_blocked_at_end = bool(_blocked_now(env, info))
            buckets[k].append(done)
            detail[k].append((d0, info["goal_distance"], still_blocked_at_end))
    env.close()

    for k in ("clear", "blocked"):
        r = buckets[k]
        worse = sum(1 for a, b, _ in detail[k] if b > a)
        endblk = sum(1 for *_, b in detail[k] if b)
        med_final = np.median([b for _, b, _ in detail[k]])
        print(f"    [{k}] solved {sum(r)}/{len(r)} = {100*sum(r)/len(r):.0f}%   "
              f"ended farther than it started: {worse}/{len(r)}   "
              f"route still blocked at end: {endblk}/{len(r)}   "
              f"median final distance {med_final:.3f} (zone r = {ZONE_RADIUS})")
    print(f"    ({time.time()-t0:.0f} s wall)")


# ---- a deterministic, layout-specific start pose: fold the arm, sweep the shoulder
def _park(env, task):
    """Place the layout, then fold the arm and rotate the shoulder until nothing
    is in contact. Deterministic given the layout, and identical for both arms of
    a paired comparison — the env's own reset resamples the LAYOUT instead, which
    a paired test cannot do."""
    qadr = [env.model.jnt_qposadr[j] for j in env._jid]
    dadr = [env.model.jnt_dofadr[j] for j in env._jid]
    for a in dadr:
        env.data.qvel[a] = 0.0
    env.data.ctrl[:] = 0.0
    for j0 in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        env.data.qpos[qadr[0]] = j0 if j0 <= np.pi else j0 - 2 * np.pi
        env.data.qpos[qadr[1]] = 2.5
        env.data.qpos[qadr[2]] = 2.5
        env._place(task)
        if not env._pucks_touching_arm():
            return True
    return False


# --------------- B4. PAIRED test: same layout, blocker in the corridor vs moved out
def _shift_blocker_out(task, deep=0.5):
    """Return a copy of the layout with the deepest blocker pushed perpendicular
    out of the corridor. Same target, same zone, same everything else."""
    p = np.asarray(task.puck_xy[task.goal_puck], dtype=float)
    q = ZONE_XY[task.goal_zone].astype(float)
    d = q - p
    u = d / np.linalg.norm(d)
    perp_u = np.array([-u[1], u[0]])
    best_i, best_depth, best_side = None, 0.0, 1.0
    for i in range(N_PUCKS):
        if i == task.goal_puck:
            continue
        c = np.asarray(task.puck_xy[i], dtype=float)
        s_ = np.clip((c - p) @ d / (d @ d), 0.0, 1.0)
        foot = p + s_ * d
        perp = float(np.linalg.norm(foot - c))
        R = PUCK_R[i] + PUCK_R[task.goal_puck]
        dep = (R - perp) / R
        if dep > best_depth:
            best_i, best_depth = i, dep
            best_side = np.sign((c - foot) @ perp_u) or 1.0
    if best_i is None:
        return None
    xy = np.array(task.puck_xy, dtype=float).copy()
    for push in (0.16, 0.20, 0.24, 0.12):
        cand = xy[best_i] + perp_u * best_side * push
        if not (SPAWN_R[0] * 0.6 < np.linalg.norm(cand) < 0.44):
            continue
        ok = all(np.linalg.norm(cand - xy[j]) > PUCK_R[best_i] + PUCK_R[j] + 0.03
                 for j in range(N_PUCKS) if j != best_i)
        ok = ok and all(np.linalg.norm(cand - ZONE_XY[z]) > ZONE_RADIUS + 0.04
                        for z in range(N_ZONES))
        if ok:
            xy[best_i] = cand
            from sandbox_env import Task
            t2 = Task(xy, np.array(task.puck_theta).copy(), task.goal_puck, task.goal_zone)
            if depth_of(t2) <= 0.0:
                return t2
    return None


def _run(env, policy, task, cap, hold_needed=25):
    obs, info = env.reset(options={"task": task})
    _park(env, task)
    info = env._info()
    d0 = info["goal_distance"]
    hold = 0
    for i in range(cap):
        obs, _, _, _, info = env.step(policy(info))
        hold = hold + 1 if info["goal_satisfied"] else 0
        if hold >= hold_needed:
            return True, i * 0.02, d0, info["goal_distance"]
    return False, None, d0, info["goal_distance"]


def b4_paired(n=32, cap=3000, deep=0.5):
    from watch import ScriptedPusher
    import time
    print(f"\nB4. PAIRED: same layout with the blocker in the corridor vs moved out "
          f"(depth > {deep}, n={n}, {cap*0.02:.0f} s cap)")
    env = PlanarPushSandbox(split="any", seed=101, render_obs=False)
    policy = ScriptedPusher(env)
    pairs = []
    tries = 0
    while len(pairs) < n and tries < 20000:
        tries += 1
        t = env.sample_task()
        if depth_of(t) <= deep:
            continue
        t2 = _shift_blocker_out(t)
        if t2 is not None:
            pairs.append((t, t2))
    print(f"    built {len(pairs)} pairs from {tries} draws")

    env.reset(seed=101)
    t0 = time.time()
    rows = []
    for a, b in pairs:
        ra = _run(env, policy, a, cap)
        rb = _run(env, policy, b, cap)
        rows.append((ra, rb))
    env.close()

    sa = sum(1 for ra, rb in rows if ra[0])
    sb = sum(1 for ra, rb in rows if rb[0])
    both = sum(1 for ra, rb in rows if ra[0] and rb[0])
    only_clear = sum(1 for ra, rb in rows if rb[0] and not ra[0])
    only_blocked = sum(1 for ra, rb in rows if ra[0] and not rb[0])
    print(f"    blocked solved {sa}/{len(rows)}    blocker-removed solved {sb}/{len(rows)}")
    print(f"    both {both}   only-when-clear {only_clear}   only-when-blocked {only_blocked}")
    ta = [ra[1] for ra, rb in rows if ra[0] and rb[0]]
    tb = [rb[1] for ra, rb in rows if ra[0] and rb[0]]
    if both:
        print(f"    time to solve on the {both} both-solved pairs: blocked "
              f"{np.median(ta):.1f} s vs clear {np.median(tb):.1f} s")
    print(f"    ({time.time()-t0:.0f} s wall)")


def b5_control(n=24, cap=3000):
    """Sanity arm: the same harness on unconditioned tasks. Should reproduce ~25%."""
    from watch import ScriptedPusher
    import time
    print(f"\nB5. control: unconditioned tasks through the same harness (n={n})")
    env = PlanarPushSandbox(split="any", seed=101, render_obs=False)
    policy = ScriptedPusher(env)
    env.reset(seed=101)
    t0 = time.time()
    solved = pen = 0
    depths = []
    for _ in range(n):
        t = env.sample_task()
        depths.append(depth_of(t))
        env.reset(options={"task": t})
        if not _park(env, t):
            pen += 1
        ok, *_ = _run(env, policy, t, cap)
        solved += ok
    env.close()
    print(f"    solved {solved}/{n} = {100*solved/n:.0f}%   "
          f"layouts spawned in contact with the arm: {pen}/{n}   "
          f"median blocking depth {np.median(depths):.2f}")
    print(f"    ({time.time()-t0:.0f} s wall)")


def b6_by_blocker(n=20, cap=3000, deep=0.5):
    """Does WHICH puck blocks matter? Puck 0 is light (0.05 kg, mu 0.20); puck 2 is
    heavy (0.20 kg, mu 0.45) and should be far harder to shove out of the way."""
    from watch import ScriptedPusher
    import time
    print(f"\nB6. paired, split by blocker identity (n={n} per blocker, depth > {deep})")
    env = PlanarPushSandbox(split="any", seed=202, render_obs=False)
    policy = ScriptedPusher(env)
    env.reset(seed=202)
    for blk in (0, 2):
        pairs, tries = [], 0
        while len(pairs) < n and tries < 60000:
            tries += 1
            t = env.sample_task()
            if t.goal_puck == blk or depth_of(t) <= deep:
                continue
            # the deepest blocker must be the one we are studying
            p = np.asarray(t.puck_xy[t.goal_puck], float)
            q = ZONE_XY[t.goal_zone].astype(float)
            d = q - p
            deps = {}
            for i in range(N_PUCKS):
                if i == t.goal_puck:
                    continue
                c = np.asarray(t.puck_xy[i], float)
                s_ = np.clip((c - p) @ d / (d @ d), 0.0, 1.0)
                R = PUCK_R[i] + PUCK_R[t.goal_puck]
                deps[i] = (R - float(np.linalg.norm(p + s_ * d - c))) / R
            if max(deps, key=deps.get) != blk:
                continue
            t2 = _shift_blocker_out(t)
            if t2 is not None:
                pairs.append((t, t2))
        t0 = time.time()
        sa = sb = 0
        moved = []
        for a, b in pairs:
            start = np.asarray(a.puck_xy[blk], float).copy()
            ok_a, *_ = _run(env, policy, a, cap)
            moved.append(float(np.linalg.norm(env._puck_pose(blk)[:2] - start)))
            ok_b, *_ = _run(env, policy, b, cap)
            sa += ok_a
            sb += ok_b
        print(f"    blocker = puck {blk} (mass {env.model.body_mass[env._puck_bid[blk]]:.3f} kg): "
              f"blocked {sa}/{len(pairs)}  removed {sb}/{len(pairs)}   "
              f"median blocker displacement over the run {np.median(moved):.3f} m  "
              f"({time.time()-t0:.0f} s)")
    env.close()


def _place_at_standoff(task, blk):
    """CONSTRUCT a standoff-blocked layout: put puck `blk` exactly where the paddle
    must stand to push the target straight at its zone. Returns (blocked, cleared)."""
    from sandbox_env import Task
    p = np.asarray(task.puck_xy[task.goal_puck], float)
    q = ZONE_XY[task.goal_zone].astype(float)
    u = (q - p) / np.linalg.norm(q - p)
    stand = p - u * (PUCK_R[task.goal_puck] + PADDLE)
    if np.linalg.norm(stand) < PED + PADDLE + 0.02 or np.linalg.norm(stand) > 0.42:
        return None
    xy = np.array(task.puck_xy, float).copy()
    xy[blk] = stand
    others = [j for j in range(N_PUCKS) if j not in (blk, task.goal_puck)]
    for j in others:
        if np.linalg.norm(xy[j] - stand) < PUCK_R[j] + PUCK_R[blk] + 0.03:
            return None
    if np.linalg.norm(stand - p) > PUCK_R[blk] + PUCK_R[task.goal_puck] + 0.02:
        return None          # must actually be touching the standoff region
    if any(np.linalg.norm(stand - ZONE_XY[z]) < ZONE_RADIUS + 0.04 for z in range(N_ZONES)):
        return None
    blocked = Task(xy, np.array(task.puck_theta).copy(), task.goal_puck, task.goal_zone)
    # cleared counterpart: same blocker, swung 90 degrees round the target
    perp = np.array([-u[1], u[0]])
    for side in (1.0, -1.0):
        cand = p + perp * side * (PUCK_R[task.goal_puck] + PUCK_R[blk] + 0.02)
        if not (PED + PUCK_R[blk] + 0.02 < np.linalg.norm(cand) < 0.42):
            continue
        if any(np.linalg.norm(cand - xy[j]) < PUCK_R[j] + PUCK_R[blk] + 0.03 for j in others):
            continue
        if any(np.linalg.norm(cand - ZONE_XY[z]) < ZONE_RADIUS + 0.04 for z in range(N_ZONES)):
            continue
        xy2 = np.array(task.puck_xy, float).copy()
        xy2[blk] = cand
        cleared = Task(xy2, np.array(task.puck_theta).copy(), task.goal_puck, task.goal_zone)
        if depth_of(cleared) <= 0.2:
            return blocked, cleared
    return None


def b7_standoff(n=20, cap=3000):
    """The other blocking test: block the PADDLE's stance, not the puck's route."""
    from watch import ScriptedPusher
    import time
    print(f"\nB7. paired, blocker on the STANDOFF point (n={n} per blocker identity)")
    env = PlanarPushSandbox(split="any", seed=303, render_obs=False)
    policy = ScriptedPusher(env)
    env.reset(seed=303)
    for blk in (0, 2):
        pairs, tries = [], 0
        while len(pairs) < n and tries < 60000:
            tries += 1
            t = env.sample_task()
            if t.goal_puck == blk:
                continue
            got = _place_at_standoff(t, blk)
            if got:
                pairs.append(got)
        t0 = time.time()
        sa = sb = 0
        for a, b in pairs:
            ok_a, *_ = _run(env, policy, a, cap)
            ok_b, *_ = _run(env, policy, b, cap)
            sa += ok_a
            sb += ok_b
        print(f"    blocker = puck {blk}: standoff-blocked {sa}/{len(pairs)}   "
              f"swung aside {sb}/{len(pairs)}   ({tries} draws, {time.time()-t0:.0f} s)")
    env.close()


def _sweep_time(gears, j, angle=0.5, pose=(0.6, -1.1, -0.9)):
    m, d = fresh()
    for k in range(3):
        m.actuator_gear[k, 0] = gears[k]
    qa, da = jadr(m)
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    for k in range(3):
        d.qpos[qa[k]] = pose[k]
    for i in range(N_PUCKS):
        a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_x")]
        d.qpos[a:a + 2] = [0.45, 0.45 - 0.12 * i]
    mujoco.mj_forward(m, d)
    q0 = d.qpos[qa[j]]
    d.ctrl[:] = 0.0
    d.ctrl[j] = m.actuator_ctrlrange[j, 1]
    for _ in range(int(6.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if abs(d.qpos[qa[j]] - q0) > angle:
            return d.time
    return float("nan")


def a4_ladders():
    print("\nA4. candidate gear ladders: driven sweep time per joint, and tip force")
    for gears in ((1, 1, 1), (2, 1, 0.5), (3, 1, 0.4), (4, 1, 0.35), (3, 1.2, 0.5)):
        ts = [_sweep_time(gears, j) for j in range(3)]
        good = [t for t in ts if t == t]
        ratio = max(good) / min(good) if len(good) == 3 else float("nan")
        print(f"    gear {gears}: sweep " + ", ".join(
            f"j{j} {ts[j]*1000:6.1f} ms" if ts[j] == ts[j] else f"j{j}  never" for j in range(3))
            + f"   ladder {ratio:.2f}x   tip force {a3_tip_force(gears):.2f} N")


def _sweep_time_cfg(gears, armature, j, angle=0.5, pose=(0.6, -1.1, -0.9)):
    m, d = fresh()
    qa, da = jadr(m)
    for k in range(3):
        m.actuator_gear[k, 0] = gears[k]
        m.dof_armature[da[k]] = armature[k]
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    for k in range(3):
        d.qpos[qa[k]] = pose[k]
    for i in range(N_PUCKS):
        a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_x")]
        d.qpos[a:a + 2] = [0.45, 0.45 - 0.12 * i]
    mujoco.mj_forward(m, d)
    q0 = d.qpos[qa[j]]
    d.ctrl[:] = 0.0
    d.ctrl[j] = m.actuator_ctrlrange[j, 1]
    for _ in range(int(8.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if abs(d.qpos[qa[j]] - q0) > angle:
            return d.time
    return float("nan")


def _tau_cfg(armature, j, pose=(0.6, -1.1, -0.9)):
    m, d = fresh()
    qa, da = jadr(m)
    for k in range(3):
        m.dof_armature[da[k]] = armature[k]
    return decay_tau(m, d, j, pose)


def a5_slow_and_strong():
    print("\nA5. can the shoulder be SLOW and STRONG at once?")
    print("    gear alone (gear down = slow AND weak):")
    for gears in ((0.5, 1, 2), (0.4, 1, 3)):
        ts = [_sweep_time(gears, j) for j in range(3)]
        good = [t for t in ts if t == t]
        r = max(good) / min(good) if len(good) == 3 else float("nan")
        print(f"      gear {gears}: " + ", ".join(
            f"j{j} {ts[j]*1000:6.1f} ms" if ts[j] == ts[j] else f"j{j}  never" for j in range(3))
            + f"   ladder {r:.2f}x   tip force {a3_tip_force(gears):.2f} N")
    print("    armature (rotor inertia) for the timescale, gear for the strength:")
    for arm, gears in (((0.05, 0.01, 0.002), (1, 1, 1)),
                       ((0.05, 0.01, 0.002), (2, 1, 1)),
                       ((0.12, 0.01, 0.002), (3, 1, 1)),
                       ((0.12, 0.02, 0.002), (3, 1, 1))):
        ts = [_sweep_time_cfg(gears, arm, j) for j in range(3)]
        taus = [_tau_cfg(arm, j) for j in range(3)]
        good = [t for t in ts if t == t]
        r = max(good) / min(good) if len(good) == 3 else float("nan")
        print(f"      armature {arm} gear {gears}:")
        print(f"        sweep " + ", ".join(
            f"j{j} {ts[j]*1000:6.1f} ms" if ts[j] == ts[j] else f"j{j}  never" for j in range(3))
            + f"   ladder {r:.2f}x   tip force {a3_tip_force(gears):.2f} N")
        print(f"        passive tau " + ", ".join(f"j{j} {taus[j]*1000:6.1f} ms" for j in range(3))
              + f"   passive ladder {max(taus)/min(taus):.2f}x")


# ------------------------------- B8. the POCKET: a blocker that cannot be displaced
def _pocket(task, blk):
    """Jam the target against the pedestal and jam `blk` against the pedestal too,
    immediately on the side the target must travel toward. The blocker cannot be
    pushed inward (the pedestal is there) and the target cannot leave the pocket
    without it moving.

    Paired control: the SAME blocker, same radius, same contact with the pedestal,
    rotated to the far side of the pedestal. Only the angle differs.
    """
    from sandbox_env import Task
    tgt = task.goal_puck
    q = ZONE_XY[task.goal_zone].astype(float)
    # target jammed against the pedestal, on the side away from its zone, so the
    # route to the zone runs around the pedestal rather than straight off it
    phi = math.atan2(q[1], q[0]) + math.pi
    r_t = PED + PUCK_R[tgt] + 0.002
    p = np.array([r_t * math.cos(phi), r_t * math.sin(phi)])
    # which way round the pedestal is shorter to the zone: seal that way
    turn = 1.0 if ((math.atan2(q[1], q[0]) - phi + math.pi) % (2 * math.pi) - math.pi) > 0 else -1.0
    r_b = PED + PUCK_R[blk] + 0.002
    # angular offset that just puts the blocker in contact with the target
    gap = PUCK_R[tgt] + PUCK_R[blk] + 0.004
    dphi = 2 * math.asin(min(0.999, gap / (r_t + r_b)))
    third = [j for j in range(N_PUCKS) if j not in (tgt, blk)][0]

    def build(bphi):
        xy = np.zeros((N_PUCKS, 2))
        xy[tgt] = p
        xy[blk] = [r_b * math.cos(bphi), r_b * math.sin(bphi)]
        # park the third puck out of the way, same place in both arms
        xy[third] = [0.40 * math.cos(phi + 0.9), 0.40 * math.sin(phi + 0.9)]
        for j in range(N_PUCKS):
            for k in range(j + 1, N_PUCKS):
                if np.linalg.norm(xy[j] - xy[k]) < PUCK_R[j] + PUCK_R[k] - 0.001:
                    return None
            if any(np.linalg.norm(xy[j] - ZONE_XY[z]) < ZONE_RADIUS + 0.03
                   for z in range(N_ZONES)):
                return None
        return Task(xy, np.array(task.puck_theta).copy(), tgt, task.goal_zone)

    blocked = build(phi + turn * dphi)
    cleared = build(phi - turn * dphi)      # same jam, wrong side: route is open
    if blocked is None or cleared is None:
        return None
    return blocked, cleared


def b8_pocket(n=12, cap=3000):
    from watch import ScriptedPusher
    import time
    print(f"\nB8. POCKET: target jammed on the pedestal, blocker sealing the short way "
          f"round (n={n} per blocker)")
    env = PlanarPushSandbox(split="any", seed=404, render_obs=False)
    policy = ScriptedPusher(env)
    env.reset(seed=404)
    for blk in (0, 2):
        pairs = []
        while len(pairs) < n:
            t = env.sample_task()
            if t.goal_puck == blk:
                continue
            got = _pocket(t, blk)
            if got:
                pairs.append(got)
        t0 = time.time()
        sa = sb = 0
        mov = []
        for a, b in pairs:
            start = np.asarray(a.puck_xy[blk], float).copy()
            ok_a, *_ = _run(env, policy, a, cap)
            mov.append(float(np.linalg.norm(env._puck_pose(blk)[:2] - start)))
            ok_b, *_ = _run(env, policy, b, cap)
            sa += ok_a
            sb += ok_b
        print(f"    blocker = puck {blk}: sealed {sa}/{len(pairs)}   open {sb}/{len(pairs)}   "
              f"median blocker displacement {np.median(mov):.3f} m  ({time.time()-t0:.0f} s)")
    env.close()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "a"):
        a1_passive()
        a2_inertia_and_gear()
        a3()
        a4_ladders()
        a5_slow_and_strong()
    if which in ("all", "b"):
        b1_incidental()
        b2_cost()
    if which in ("all", "b3"):
        b3_achievability()
    if which in ("all", "b4"):
        b4_paired()
    if which in ("all", "b5"):
        b5_control()
    if which in ("all", "b6"):
        b6_by_blocker()
    if which in ("all", "b7"):
        b7_standoff()
    if which in ("all", "b8"):
        b8_pocket()