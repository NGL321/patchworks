"""If the observation render is preceded by mj_forward, does the trajectory move?"""
import numpy as np, torch, mujoco
from patchworks.agent import Agent
from patchworks.graph import build_graph
from patchworks.sandbox import PlanarPushSandbox
import patchworks.sandbox.env as envmod

def run(with_forward, ticks=20):
    dome = build_graph()
    env = PlanarPushSandbox(split="any")
    if with_forward:
        orig = envmod.PlanarPushSandbox._camera_image
        def patched(self):
            mujoco.mj_forward(self.model, self.data)
            return orig(self)
        envmod.PlanarPushSandbox._camera_image = patched
    try:
        agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(0))
        obs, _ = env.reset(seed=0)
        agent.write(obs, np.zeros(env.action_space.shape, dtype=np.float32))
        qpos, images = [], []
        for _ in range(ticks):
            out = agent.tick()
            qpos.append(env.data.qpos.copy())
            images.append(out.observation["image"].copy())
        return np.array(qpos), np.array(images)
    finally:
        if with_forward:
            envmod.PlanarPushSandbox._camera_image = orig
        env.close()

qa, ia = run(False)
qb, ib = run(True)
dq = np.abs(qa - qb)
di = np.abs(ia.astype(int) - ib.astype(int))
print(f"qpos  max abs difference over 20 ticks : {dq.max():.3e}")
print(f"image max abs difference over 20 ticks : {di.max()}")
print(f"image differing-pixel fraction (last)  : {(di[-1].sum(axis=-1) > 0).mean():.4f}")
print(f"first tick where qpos differs          : "
      f"{int(np.argmax(dq.max(axis=1) > 0)) if (dq.max(axis=1) > 0).any() else 'never'}")
