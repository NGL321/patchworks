"""PROTOTYPE — throwaway. Headless check: shapes, dynamics, sampler, continual reset.

    .venv-proto/bin/python prototypes/sandbox/probe.py

Prints the full observation state after every phase, writes frames to
prototypes/sandbox/out/ so the render can be eyeballed without a display.
"""

import os
from collections import Counter

import numpy as np

from sandbox_env import (HELDOUT_PAIRS, N_PUCKS, N_ZONES, PlanarPushSandbox,
                         _in_heldout_sector)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def show(tag, obs, info):
    print(f"\n[{tag}]  t={info['sim_time']:6.2f}s  rearrangements={info['rearrangements']}")
    print(f"  qpos  {np.round(obs['qpos'], 3)}   qvel {np.round(obs['qvel'], 3)}")
    print(f"  touch {np.round(obs['touch'], 4)}   tip  {np.round(info['tip_xy'], 3)}")
    print(f"  pucks {np.round(info['puck_pose'][:, :2], 3).tolist()}")
    print(f"  goal  puck {info['goal_puck']} -> zone {info['goal_zone']}"
          f"   d={info['goal_distance']:.3f}  satisfied={info['goal_satisfied']}"
          f"   heldout_pair={info['heldout_pair']}")


def save(env, name):
    import imageio.v3 as iio
    os.makedirs(OUT, exist_ok=True)
    big = PlanarPushSandbox.__dict__  # keep the small obs image honest; render a big one too
    iio.imwrite(os.path.join(OUT, f"{name}.png"), env.render())
    print(f"  wrote out/{name}.png")


def main():
    env = PlanarPushSandbox(split="train", seed=0)
    obs, info = env.reset(seed=0)
    print("observation_space:", {k: v.shape for k, v in env.observation_space.items()})
    print("action_space:", env.action_space)
    show("after first reset", obs, info)
    save(env, "01_reset")

    # --- the arm actually moves and actually pushes -----------------------------
    rng = np.random.default_rng(1)
    a = np.zeros(3)
    contacts = 0
    for i in range(400):
        a = 0.9 * a + 0.1 * rng.uniform(-1, 1, 3)     # smooth babble, not white noise
        obs, r, term, trunc, info = env.step(a)
        contacts += int(obs["touch"].sum() > 1e-6)
        assert r == 0.0 and term is False and trunc is False, "no reward channel, no episodes"
    show("after 400 ticks of motor babble", obs, info)
    print(f"  ticks with contact: {contacts}/400")
    save(env, "02_babble")

    # --- reset rearranges the world without resetting the agent -----------------
    arm_before = obs["qpos"].copy()
    t_before = info["sim_time"]
    obs, info = env.reset()
    same_arm = np.allclose(arm_before, obs["qpos"])
    show("after reset #2 (world rearranged)", obs, info)
    print(f"  arm configuration preserved: {same_arm}   clock monotonic: {info['sim_time'] >= t_before}")
    save(env, "03_rearranged")

    # --- the human's hand -------------------------------------------------------
    env.perturb(info["goal_puck"], (0.30, 0.30))
    env.retarget(goal_zone=(info["goal_zone"] + 1) % N_ZONES)
    obs, r, term, trunc, info = env.step(np.zeros(3))
    show("after perturb + retarget", obs, info)
    save(env, "04_perturbed")

    # --- snapshot / restore is the only reproducibility story -------------------
    snap = env.snapshot()
    for _ in range(50):
        env.step(rng.uniform(-1, 1, 3))
    drifted = env._obs()["qpos"].copy()
    env.restore(snap)
    restored = env._obs()["qpos"].copy()
    print(f"\n[snapshot]  drifted != restored: {not np.allclose(drifted, restored)}"
          f"   restored == snapshot: {np.allclose(restored, snap['qpos'][:3])}")

    # --- the sampler ------------------------------------------------------------
    for split in ("train", "heldout"):
        e = PlanarPushSandbox(split=split, seed=7)
        pairs = Counter()
        sector = 0
        for _ in range(600):
            t = e.sample_task()
            pairs[t.pair] += 1
            sector += int(_in_heldout_sector(t.puck_xy[t.goal_puck]))
        held = sum(v for k, v in pairs.items() if k in HELDOUT_PAIRS)
        print(f"\n[sampler:{split}] {len(pairs)}/{N_PUCKS * N_ZONES} pairs seen, "
              f"{held} draws on a held-out pair, {sector} draws in the held-out sector")
        print("  pair counts:", dict(sorted(pairs.items())))
        e.close()

    env.close()
    print("\nok")


if __name__ == "__main__":
    main()
