"""PROTOTYPE — throwaway. Measures the puck model for issue #21.

    .venv-proto/bin/python prototypes/sandbox/dynamics_probe.py

Five questions, all about the pucks, none about the agent:

1. What does the stray `<default><joint armature="0.01"/>` actually do to a puck?
2. Do the three pucks coast differently, or are they one model with a colour on it?
3. Is joint frictionloss isotropic? (eq. 9 bounds it element-wise -> a square, not a disc.)
4. Where is the break-away force, at the joint and at the paddle?
5. Does a sub-threshold hold creep, and how far over 60 s?

Every number is printed twice: as built (armature 0.01) and with the puck dofs'
armature zeroed, which is the fix.
"""

import os
import numpy as np
import mujoco

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena.xml")
G = 9.81
PUCKS = (0, 1, 2)
FAR = np.array([[0.30, 0.0], [0.0, 0.30], [-0.30, 0.0]])   # apart, clear of the arm


def fresh(armature: bool, contact: bool = False):
    """Model with the puck dofs' armature forced to the pre-#21 value or to zero. Contact is off by
    default: tests 2-5 drive the puck joints directly, so the arm and the walls
    are only a source of collisions that would corrupt the measurement."""
    m = mujoco.MjModel.from_xml_path(XML)
    if not contact:
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    for i in PUCKS:
        for ax in "xyr":
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_{ax}")
            # the class now sets armature="0"; put the old value back to keep the
            # before/after comparison meaningful after the fix landed
            m.dof_armature[m.jnt_dofadr[j]] = 0.01 if armature else 0.0
    # park the arm out of the way and spread the pucks
    d.qpos[:3] = [np.pi / 2, 1.2, 1.2]
    for i in PUCKS:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_x")
        a = m.jnt_qposadr[j]
        d.qpos[a:a + 2] = FAR[i]
    mujoco.mj_forward(m, d)
    return m, d


def dofs(m, i):
    return [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_{ax}")]
            for ax in "xyr"]


def qadr(m, i):
    return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_x")]


# ---------------------------------------------------------------- 1. the armature
def q1_armature():
    print("\n=== 1. what armature=0.01 does to each puck =========================")
    m, _ = fresh(True)
    print(f"{'puck':>5} {'mass':>7} {'true I':>10} {'+arm m':>8} {'m err':>7} "
          f"{'+arm I':>10} {'I err':>8}")
    for i in PUCKS:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"puck_{i}")
        mass = m.body_mass[bid]
        Izz = m.body_inertia[bid][2]
        arm = m.dof_armature[dofs(m, i)]
        print(f"{i:>5} {mass:7.3f} {Izz:10.3e} {mass + arm[0]:8.3f} "
              f"{100 * arm[0] / mass:6.1f}% {Izz + arm[2]:10.3e} {(Izz + arm[2]) / Izz:7.1f}x")


# ------------------------------------------------------- 2 & 3. coast distance
def coast(m, d, i, speed, theta):
    """Launch puck i at `speed` along `theta`; return (distance, time, decel)."""
    dof, a = dofs(m, i), qadr(m, i)
    d.qpos[a:a + 2] = FAR[i]          # always launch from the same clear spot
    d.qvel[dof] = 0
    mujoco.mj_forward(m, d)
    x0 = d.qpos[a:a + 2].copy()
    d.qvel[dof[0]] = speed * np.cos(theta)
    d.qvel[dof[1]] = speed * np.sin(theta)
    t0 = d.time
    touched = False
    for _ in range(20000):
        mujoco.mj_step(m, d)
        touched |= d.ncon > 0
        if np.linalg.norm(d.qvel[dof[:2]]) < 1e-3:
            break
    assert not touched, f"puck {i} hit something at theta={np.rad2deg(theta):.0f}"
    dist = float(np.linalg.norm(d.qpos[a:a + 2] - x0))
    dt = d.time - t0
    return dist, dt, (speed / dt if dt > 0 else float("nan"))


def q2_coast():
    print("\n=== 2. free coast from 1.0 m/s, axial (mu*g = 2.94 m/s^2 predicted) ==")
    for arm in (True, False):
        m, d = fresh(arm)
        row = []
        for i in PUCKS:
            dist, t, dec = coast(m, d, i, 1.0, 0.0)
            row.append(f"puck {i}: {dist:.3f} m  {t:.2f} s  a={dec:.2f}")
        print(f"  armature={'0.01' if arm else '0   '}  " + " | ".join(row))


def q3_anisotropy():
    print("\n=== 3. coast distance vs. direction (square friction -> 45 deg dip) ==")
    angles = np.arange(0, 91, 15.0)
    for arm in (True, False):
        m, d = fresh(arm)
        print(f"  armature={'0.01' if arm else '0   '}")
        for i in PUCKS:
            out = []
            for th in np.deg2rad(angles):
                dist, _, _ = coast(m, d, i, 1.0, th)
                out.append(dist)
            worst = min(out) / max(out) - 1
            print(f"    puck {i}: " + "  ".join(f"{a:2.0f}d {v:.3f}" for a, v in zip(angles, out))
                  + f"   spread {100 * worst:+.1f}%")


# ------------------------------------------------------------- 4. break-away
def breakaway(m, d, i, hold=1.0, hi=6.0):
    """Smallest steady x-force that moves puck i more than 1 mm in `hold` s."""
    dof, a = dofs(m, i), qadr(m, i)
    lo = 0.0
    for _ in range(18):
        f = 0.5 * (lo + hi)
        d.qpos[a:a + 2] = FAR[i]
        d.qvel[dof] = 0
        mujoco.mj_forward(m, d)
        x0 = d.qpos[a]
        for _ in range(int(hold / m.opt.timestep)):
            d.qfrc_applied[dof[0]] = f
            mujoco.mj_step(m, d)
        d.qfrc_applied[dof[0]] = 0
        moved = abs(d.qpos[a] - x0) > 1e-3
        hi, lo = (f, lo) if moved else (hi, f)
    return 0.5 * (lo + hi)


def q4_breakaway():
    print("\n=== 4. break-away force at the joint (frictionloss is the prediction) =")
    for arm in (True, False):
        m, d = fresh(arm)
        out = []
        for i in PUCKS:
            fl = m.dof_frictionloss[dofs(m, i)[0]]
            out.append(f"puck {i}: {breakaway(m, d, i):.3f} N (fl {fl:.3f})")
        print(f"  armature={'0.01' if arm else '0   '}  " + " | ".join(out))


# ------------------------------------------------------------------ 5. creep
def creep(m, d, i, frac, seconds=60.0):
    dof, a = dofs(m, i), qadr(m, i)
    f = frac * m.dof_frictionloss[dof[0]]
    d.qpos[a:a + 2] = FAR[i]
    d.qvel[dof] = 0
    mujoco.mj_forward(m, d)
    x0 = d.qpos[a]
    for _ in range(int(seconds / m.opt.timestep)):
        d.qfrc_applied[dof[0]] = f
        mujoco.mj_step(m, d)
    d.qfrc_applied[dof[0]] = 0
    return abs(d.qpos[a] - x0), abs(d.qvel[dof[0]])


def q5_creep():
    print("\n=== 5. sub-threshold hold for 60 s (zone radius = 0.075 m) ===========")
    for arm in (True, False):
        m, d = fresh(arm)
        print(f"  armature={'0.01' if arm else '0   '}")
        for i in PUCKS:
            for frac in (0.5, 0.9, 0.99):
                dx, v = creep(m, d, i, frac)
                print(f"    puck {i} at {frac:4.0%} of frictionloss: "
                      f"drift {dx * 1000:8.4f} mm   final |v| {v:.2e} m/s")


# ------------------------------------- 6. is the creep physical or numerical?
def q6_creep_sensitivity():
    print("\n=== 6. what the creep responds to (puck 2, 90% of frictionloss, 60 s) =")
    base = "as built"
    knobs = [
        (base, {}),
        ("timestep 0.0005", {"timestep": 0.0005}),
        ("solver iterations 200", {"iterations": 200}),
        ("solver=CG", {"solver": mujoco.mjtSolver.mjSOL_CG}),
        ("integrator=implicit", {"integrator": mujoco.mjtIntegrator.mjINT_IMPLICIT}),
        ("frictionloss solimp[0]=0.99", {"solimp0": 0.99}),
        ("frictionloss solimp[0]=0.9999", {"solimp0": 0.9999}),
        ("noslip 20 iters", {"noslip": 20}),
    ]
    for name, k in knobs:
        m, d = fresh(False)
        if "timestep" in k:
            m.opt.timestep = k["timestep"]
        if "iterations" in k:
            m.opt.iterations = k["iterations"]
        if "solver" in k:
            m.opt.solver = k["solver"]
        if "integrator" in k:
            m.opt.integrator = k["integrator"]
        if "noslip" in k:
            m.opt.noslip_iterations = k["noslip"]
        if "solimp0" in k:
            for j in range(m.njnt):
                m.jnt_solimp[j][0] = k["solimp0"]
                m.dof_solimp[m.jnt_dofadr[j]][0] = k["solimp0"]
        dx, v = creep(m, d, 2, 0.9)
        print(f"    {name:<30} drift {dx * 1000:9.4f} mm   final |v| {v:.2e} m/s")


# --------------------------------- 7. the same two numbers, through the paddle
def ik(m, d, target, q0=(0.6, -1.1, -0.9), iters=400):
    """Gauss-Newton the tip site onto `target` (x, y), starting bent."""
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    jac, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    d.qpos[:3] = q0
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = np.asarray(target) - d.site_xpos[tip][:2]
        if np.linalg.norm(err) < 1e-6:
            break
        mujoco.mj_jacSite(m, d, jac, jacr, tip)
        J = jac[:2, :3]
        d.qpos[:3] += np.linalg.pinv(J) @ err * 0.5
    return d.site_xpos[tip][:2].copy()


def q7_through_the_paddle(seconds=20.0):
    """Press the paddle into puck 2 with a steady tip force and hold."""
    print(f"\n=== 7. steady tip force against puck 2, contact on, {seconds:.0f} s ==========")
    tip = mujoco.mj_name2id(m0 := mujoco.MjModel.from_xml_path(XML),
                            mujoco.mjtObj.mjOBJ_SITE, "tip")
    del m0
    for tipF in (0.2, 0.4, 0.6, 1.0, 1.9, 2.5):
        m, d = fresh(False, contact=True)
        a2 = qadr(m, 2)
        for i in (0, 1):                      # the other two, well out of the way
            d.qpos[qadr(m, i):qadr(m, i) + 2] = [-0.35, 0.20 - 0.40 * i]
        d.qpos[a2:a2 + 2] = [0.22, 0.0]      # 0.245 m of clear run before the wall
        # stand the paddle just off the puck's -x face: paddle r 0.030, puck r 0.055
        ik(m, d, (0.22 - 0.055 - 0.030 - 0.0005, 0.0))
        mujoco.mj_forward(m, d)
        x0 = d.qpos[a2]
        LIM = np.array([3.0, 2.0, 1.0])
        jac, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        forces, sat = [], 0
        for step in range(int(seconds / m.opt.timestep)):
            mujoco.mj_jacSite(m, d, jac, jacr, tip)
            tau = jac[:2, :3].T @ np.array([tipF, 0.0])
            sat += int(np.any(np.abs(tau) > LIM))
            d.ctrl[:] = np.clip(tau, -LIM, LIM)
            mujoco.mj_step(m, d)
            if step % 500 == 0:
                forces.append(float(d.sensordata[2]))
        print(f"    tip {tipF:3.1f} N -> puck moved {1000 * (d.qpos[a2] - x0):8.2f} mm, "
              f"final |v| {abs(d.qvel[dofs(m, 2)[0]]):.2e} m/s, "
              f"touch median {np.median(forces):5.2f} N, torque-saturated {100*sat/step:.0f}% of ticks"
              + ("   [PINNED ON THE WALL]" if d.ncon > 4 else ""))


# ------------------------- 8. the tip force at which a puck actually slides
def slides(m, d, i, tipF, seconds=6.0):
    """True if puck i leaves creep and actually travels under a steady tip force."""
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    a = qadr(m, i)
    r = m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"g_puck_{i}")][0]
    for k in PUCKS:
        if k != i:
            d.qpos[qadr(m, k):qadr(m, k) + 2] = [-0.35, 0.20 - 0.40 * k]
        d.qvel[dofs(m, k)] = 0
    d.qpos[a:a + 2] = [0.22, 0.0]
    ik(m, d, (0.22 - r - 0.030 - 0.0005, 0.0))
    mujoco.mj_forward(m, d)
    x0 = d.qpos[a]
    LIM = np.array([3.0, 2.0, 1.0])
    jac, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_jacSite(m, d, jac, jacr, tip)
        d.ctrl[:] = np.clip(jac[:2, :3].T @ np.array([tipF, 0.0]), -LIM, LIM)
        mujoco.mj_step(m, d)
    # creep runs at a few mm/s; a real slide covers centimetres in six seconds
    return (d.qpos[a] - x0) > 0.05


def q8_tip_threshold():
    print("\n=== 8. tip force that moves a puck for real (creep excluded) =========")
    print("      spec claims the heaviest puck needs more than 2 N")
    for arm in (True, False):
        m, d = fresh(arm, contact=True)
        out = []
        for i in PUCKS:
            lo, hi = 0.0, 3.0
            for _ in range(9):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if slides(m, d, i, mid) else (mid, hi)
            fl = m.dof_frictionloss[dofs(m, i)[0]]
            out.append(f"puck {i}: {0.5 * (lo + hi):.2f} N (joint fl {fl:.3f})")
        print(f"  armature={'0.01' if arm else '0   '}  " + " | ".join(out))


# ------------------------------ 9. the spec's clean-push distance, re-measured
def shove(m, d, i, tipF, drive, cap=8.0):
    """One clean push: steady tip force for `drive` seconds, then let go and coast."""
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    a, dof = qadr(m, i), dofs(m, i)
    r = m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"g_puck_{i}")][0]
    for k in PUCKS:
        d.qpos[qadr(m, k):qadr(m, k) + 2] = [-0.38, 0.22 - 0.44 * k]
        d.qvel[dofs(m, k)] = 0
    # start at 0.22: any closer and the paddle's standoff point is inside the
    # pedestal (radius 0.08), which spawns the arm in penetration and launches it
    d.qpos[a:a + 2] = [0.22, 0.0]
    ik(m, d, (0.22 - r - 0.030 - 0.0005, 0.0))
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    x0 = d.qpos[a:a + 2].copy()
    LIM = np.array([3.0, 2.0, 1.0])
    jac, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    vmax = rmax = 0.0
    for step in range(int(cap / m.opt.timestep)):
        mujoco.mj_jacSite(m, d, jac, jacr, tip)
        f = tipF if step * m.opt.timestep < drive else 0.0
        d.ctrl[:] = np.clip(jac[:2, :3].T @ np.array([f, 0.0]), -LIM, LIM)
        mujoco.mj_step(m, d)
        vmax = max(vmax, float(np.linalg.norm(d.qvel[dof[:2]])))
        rmax = max(rmax, float(np.linalg.norm(d.qpos[a:a + 2])))
        if step * m.opt.timestep > drive and np.linalg.norm(d.qvel[dof[:2]]) < 1e-3:
            break
    return float(np.linalg.norm(d.qpos[a:a + 2] - x0)), vmax, rmax


def _rmax(): pass


def q9_push_distance():
    print("\n=== 9. one clean push, then release =================================")
    print("      spec quotes 0.12-0.17 m per clean push")
    for tipF, drive in ((1.0, 0.15), (1.0, 0.25), (2.0, 0.25), (3.0, 0.40)):
        print(f"  --- {tipF} N for {drive} s")
        for arm in (True, False):
            m, d = fresh(arm, contact=True)
            out = []
            for i in PUCKS:
                dist, vmax, rmax = shove(m, d, i, tipF, drive)
                r = m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"g_puck_{i}")][0]
                out.append(f"puck {i}: {dist:.3f} m (peak {vmax:.2f} m/s)"
                           + ("*" if rmax > 0.50 - r else ""))
            print(f"      armature={'0.01' if arm else '0   '}  " + " | ".join(out))
    print("      * reached the wall, so that distance is a floor, not a measurement")


# -------------------- 10. what regime the scripted pusher actually operates in
def q10_regime():
    """Peak puck speed under the hand-written pusher that produced the spec's numbers.

    Bauza & Rodriguez put the quasi-static breakdown at 50-80 mm/s; Yu et al.'s
    dataset tops out at 500 mm/s. Which side of that is this world on?
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sandbox_env import PlanarPushSandbox
    from watch import ScriptedPusher
    print("\n=== 10. puck speed under the scripted pusher =========================")
    env = PlanarPushSandbox(split="any", seed=5, render_obs=False)
    policy = ScriptedPusher(env)
    obs, info = env.reset(seed=5)
    speeds, travels = [], []
    for _ in range(10):
        gp = info["goal_puck"]
        dof = dofs(env.model, gp)[:2]
        run, start = None, None
        for _ in range(1500):
            obs, _, _, _, info = env.step(policy(info))
            v = float(np.linalg.norm(env.data.qvel[dof]))
            xy = info["puck_pose"][gp][:2].copy()
            speeds.append(v)
            if run is None and v > 0.10:              # a push, not a nudge
                run, start = v, xy
            elif run is not None:
                run = max(run, v)
                if v < 5e-3:
                    travels.append((float(np.linalg.norm(xy - start)), run))
                    run = None
        obs, info = env.reset()
    env.close()
    s = np.array(speeds)
    t = np.array(travels)
    print(f"    {len(s)} ticks: puck speed median {np.median(s):.3f}  p90 {np.percentile(s, 90):.3f}"
          f"  max {s.max():.3f} m/s")
    print(f"    {len(t)} pushes (peak > 0.10 m/s): travel median {np.median(t[:, 0]):.3f} m"
          f"  p90 {np.percentile(t[:, 0], 90):.3f}  max {t[:, 0].max():.3f}")
    print(f"                                 peak speed median {np.median(t[:, 1]):.3f} m/s"
          f"  max {t[:, 1].max():.3f}")


# ------------------- 11. is theta load-bearing now? (the eccentric puck, #21)
def q11_theta_matters():
    """Same push, different starting theta. A circular puck cannot care; puck 1 must."""
    print("\n=== 11. does the outcome depend on the puck's hidden theta? ==========")
    for i in PUCKS:
        m, d = fresh(False, contact=True)
        ends = []
        for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            a = qadr(m, i)
            for k in PUCKS:
                d.qpos[qadr(m, k):qadr(m, k) + 2] = [-0.38, 0.22 - 0.44 * k]
                d.qvel[dofs(m, k)] = 0
            d.qpos[a:a + 2] = [0.22, 0.0]
            d.qpos[a + 2] = th
            r = m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"g_puck_{i}")][0]
            ik(m, d, (0.22 - r - 0.030 - 0.0005, 0.0))
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            x0 = d.qpos[a:a + 2].copy()
            LIM = np.array([3.0, 2.0, 1.0])
            tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
            jac, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            for step in range(int(4.0 / m.opt.timestep)):
                mujoco.mj_jacSite(m, d, jac, jacr, tip)
                f = 3.0 if step * m.opt.timestep < 0.4 else 0.0
                d.ctrl[:] = np.clip(jac[:2, :3].T @ np.array([f, 0.0]), -LIM, LIM)
                mujoco.mj_step(m, d)
            ends.append(d.qpos[a:a + 2] - x0)
        e = np.array(ends)
        lat = e[:, 1]
        print(f"    puck {i}: lateral deflection across 8 start angles "
              f"[{1000 * lat.min():+7.2f}, {1000 * lat.max():+7.2f}] mm  "
              f"spread {1000 * (lat.max() - lat.min()):6.2f} mm   "
              f"travel {1000 * e[:, 0].mean():.1f} mm")


# ---------------------- 12. does the friction field survive snapshot / restore?
def q12_restore_exact():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sandbox_env import PlanarPushSandbox
    print("\n=== 12. snapshot / restore with a position-dependent friction field ==")
    env = PlanarPushSandbox(split="any", seed=3, render_obs=False)
    env.reset(seed=3)
    rng = np.random.default_rng(7)
    plan = rng.uniform(-1, 1, (200, 3))
    for a in plan[:100]:
        env.step(a)
    n = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION)
    snap = np.zeros(n)
    mujoco.mj_getState(env.model, env.data, snap, mujoco.mjtState.mjSTATE_INTEGRATION)
    for a in plan[100:]:
        env.step(a)
    first = env.data.qpos.copy()
    mujoco.mj_setState(env.model, env.data, snap, mujoco.mjtState.mjSTATE_INTEGRATION)
    for a in plan[100:]:
        env.step(a)
    err = float(np.abs(env.data.qpos - first).max())
    print(f"    max |qpos| divergence over a replayed 100-tick tail: {err:.3e} "
          f"({'exact' if err == 0.0 else 'NOT EXACT'})")
    env.close()


if __name__ == "__main__":
    q1_armature()
    q2_coast()
    q3_anisotropy()
    q4_breakaway()
    q5_creep()
    q6_creep_sensitivity()
    q7_through_the_paddle()
    q8_tip_threshold()
    q9_push_distance()
    q10_regime()
    q11_theta_matters()
    q12_restore_exact()
