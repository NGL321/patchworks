"""Is the live frame drawn from stale kinematics? Re-render after mj_forward."""
import numpy as np, torch, mujoco
from patchworks.agent import Agent
from patchworks.graph import build_graph
from patchworks.sandbox import PlanarPushSandbox
from patchworks.surface import CAPTURE_EVERY, Recorder, Renderer

dome = build_graph()
env = PlanarPushSandbox(split="any")
agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(0))
observation, _info = env.reset(seed=0)
agent.write(observation, np.zeros(env.action_space.shape, dtype=np.float32))

recorder = Recorder(agent)
for t in range(CAPTURE_EVERY * 3):
    outcome = agent.tick()
    rec = recorder.observe()
    if rec is None:
        continue
    live = outcome.observation["image"]
    # the env's own renderer, but after mj_forward reconciles derived kinematics
    mujoco.mj_forward(env.model, env.data)
    forwarded = env._camera_image()
    with Renderer(size=env.image_size) as r:
        drawn = r.frame(rec)
    d_live = np.abs(drawn.astype(int) - live.astype(int))
    d_fwd  = np.abs(drawn.astype(int) - forwarded.astype(int))
    speed = float(np.abs(env.data.qvel).max())
    print(f"tick {t:2d}  |qvel|max={speed:8.4f}   drawn-vs-live max={d_live.max():4d}"
          f"   drawn-vs-forwarded max={d_fwd.max():4d}")
