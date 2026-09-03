"""The render test's own assertions, run on main at each capture tick."""
import numpy as np, torch
from patchworks.agent import Agent
from patchworks.graph import build_graph
from patchworks.sandbox import PlanarPushSandbox
from patchworks.surface import CAPTURE_EVERY, Recorder, Renderer

dome = build_graph()
env = PlanarPushSandbox(split="any")
agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(0))
obs, _ = env.reset(seed=0)
agent.write(obs, np.zeros(env.action_space.shape, dtype=np.float32))
recorder = Recorder(agent)

for t in range(CAPTURE_EVERY * 4):
    outcome = agent.tick()
    rec = recorder.observe()
    if rec is None:
        continue
    with Renderer(size=env.image_size) as r:
        drawn = r.frame(rec)
    live = outcome.observation["image"]
    d = np.abs(drawn.astype(int) - live.astype(int))
    max_ok = d.max() <= 1
    frac = (d.sum(axis=-1) > 0).mean()
    frac_ok = frac < 0.01
    print(f"capture tick {t:2d}: max={d.max():4d} (<=1? {max_ok})   "
          f"differing={frac:.4f} (<0.01? {frac_ok})   "
          f"{'PASS' if (max_ok and frac_ok) else 'FAIL'}")
