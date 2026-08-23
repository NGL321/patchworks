# Citation pass: the sandbox (patchworks#18)

Validates the design closed in [patchworks#3](https://github.com/NGL321/patchworks/issues/3)
(`docs/spec/03-the-sandbox.md`, `prototypes/sandbox/arena.xml`,
`docs/adr/0001-continual-learning-applies-to-the-adapting-surface.md`).

Citations validate after the fact per the map's Notes; this document does not revise the closed
design. Patchworks' side of every comparison is described in its own terms (`CONTEXT.md`); the prior
art's side in its own field's terms. Where a source could not be reached, that is stated rather than
papered over.

The pass covers the five areas the ticket names. It found **two defects in the prototype**, **one
correctness gap in the spec**, and **one place where two already-made choices point in opposite
directions**. The rest is validation or documentation.

---

## 1. Planar pushing as a manipulation testbed

### Sources

- Yu, K.-T., Bauza, M., Fazeli, N., Rodriguez, A. (2016). "More than a Million Ways to Be Pushed:
  A High-Fidelity Experimental Dataset of Planar Pushing." *IROS 2016*, 30–37. arXiv:1604.04038.
- Bauza, M., Alet, F., Lin, Y.-C., Lozano-Pérez, T., Kaelbling, L.P., Isola, P., Rodriguez, A.
  (2019). "Omnipush." *IROS 2019*. arXiv:1910.00618.
- Tassa, Y. et al. (2018). "DeepMind Control Suite." arXiv:1801.00690.
- Yu, T. et al. (2019). "Meta-World." *CoRL 2019*. arXiv:1910.10897.
- James, S. et al. (2019). "RLBench." arXiv:1909.12271.
- Zhu, Y. et al. "robosuite" (v1.5). arXiv:2009.12293.
- Zakka, K., Tabanpour, B. et al. (2025). "MuJoCo Playground." arXiv:2502.08844.
- Gymnasium `Pusher` and Gymnasium-Robotics `FetchPush` official docs.

### What the sources say

**Every RL benchmark surveyed hands the agent object state.** Gymnasium `Pusher-v5`'s 23-D
observation includes 3-D object position and 3-D goal position (indices 17–22). `FetchPush` returns
block position and Euler rotation plus `achieved_goal`/`desired_goal`. Meta-World §4: "The
observation space is always 39 dimensional" — including "3D position and quaternion of object 1" and
"the 3D goal position". RLBench is the only genuinely vision-first member of the set.

**Torque control is the minority choice but not eccentric.** Gymnasium `Pusher` is 7-DoF torque
(`Box(-2, 2, (7,))`). Meta-World, FetchPush and robosuite default to end-effector Cartesian or
operational-space control; robosuite §3.2 reports OSC learns *faster* than joint-velocity control.
MuJoCo Playground §II-C's non-prehensile Franka task uses "high frequency torque control" with
zero-shot sim-to-real.

**Pushing is mostly not a first-class task.** robosuite v1.5's nine standardized tasks contain no
pushing task at all. DMC's manipulation domain (`manipulator`) is a planar arm *with a gripper*.

**Object sets vary shape deliberately.** Yu et al. §III used 11 water-jet-cut shapes
(`rect1-3`, `tri1-3`, `ellip1-3`, `hex`, `butter`) over 4 surfaces, stating the reason: "Different
shapes can give us insights into phenomena such as the dependence of friction with variations in the
support pressure distribution." Omnipush escalates that axis to 250 objects varying "the shape of
the object and its mass distribution".

**What was reported as hard.** DMC Table 1: `manipulator:bring_ball` scores **895.9 ± 3.7** for D4PG
from state and **0.6 ± 0.1** from pixels — the largest state-vs-pixel gap in the suite. Meta-World
Table 1: MT-SAC/MT-PPO reach only ≈35–38% on MT50. Yu et al. §I state the field's judgement: "The
reality, however, is bitter. Predicting the motion of a pushed object is not trivial."

### Correspondences

- **"No object pose is ever given to the agent"** is the choice with the hardest supporting evidence.
  Every other benchmark surveyed hands it over, and the one that made agents work from pixels in this
  domain measured 0.6 out of 1000. This is a real difficulty, not a stylistic preference.
- **The pedestal making the workspace non-convex** matches Meta-World's decision to carve out
  `push-wall` as its own named task — independent evidence that routing around an obstacle is a
  distinct competence, not free once plain pushing works.
- **Torque at 50 Hz** has direct precedent in `Pusher` and MuJoCo Playground.
- **The ring wall at 0.52** has no direct analogue, but the failure it fixes — the rejected square
  arena whose corners sat at radius 0.76 — is the same class of problem as an object leaving a
  tabletop, which benchmarks handle only by keeping goals well inside the reachable region.
- **64×64** is smaller than DMC's 84×84 pixel benchmark, but the same order.

### What argues against a choice already made

**1.1 — Circular pucks make the hidden-rotation claim unsupportable.** (spec lines 86–89)

> "Each puck carries an orientation marker. **At 64×64 the marker is not resolvable**, so puck
> rotation is a genuine hidden variable — inferable only from contact dynamics, never read off."

In `arena.xml` every puck is `type="cylinder"` (line 27, class `puck`) and the marker geom carries
`contype="0" conaffinity="0"` (lines 118–119) — visual only. The puck therefore has exact rotational
symmetry in geometry, mass distribution, and contact. Its orientation θ appears in no term of its
equations of motion: it is a pure integrator of ω, dynamically decoupled from anything the arm can
do.

The claim fails in both directions. θ is not *inferable* from contact dynamics, because nothing in
the dynamics depends on it; and it is not a useful *hidden variable*, because a predictor that
ignores it loses nothing — it can never affect any future observation. "Genuine hidden variable" is
the wrong description; **unobservable and inconsequential** is the right one.

The field went the other way deliberately: Yu et al.'s 11 non-circular shapes, Bauza & Rodriguez's
parametrisation of a push by contact point *along an edge* (which does not exist for a disc),
Omnipush's 250 objects varying mass distribution. The cheapest fix consistent with the spec's own
"epistemic pressure is over dynamics" framing is to make one puck non-circular, or give a circular
puck an off-centre centre of mass — either makes θ load-bearing without touching the render, the
observation contract, or the sampler.

**1.2 — "Pucks differ in friction" is weaker than stated.** (spec lines 50–51)

All three pucks use the *same* μ = 0.3; only its product with mass differs
(0.147 = 0.3·0.05·9.81, 0.294, 0.589). For a free-sliding puck this cancels exactly:
a = frictionloss/m = μg = 2.94 m/s² for all three. **In the coasting phase the three pucks are
literally one model with a colour attached.** They differ genuinely at *contact* — the spec's
measured >2 N break-away force for puck 2 is real and follows from the differing frictionloss bounds.
The claim that survives is narrower: mass discriminates during contact, not during travel.

**1.3 — Pushes are ballistic, not quasi-static.** (spec line 102)

With deceleration μg = 2.94 m/s², a ballistic 0.12 m glide requires a launch speed of
√(2·2.94·0.12) = **0.84 m/s**, and 0.17 m requires **1.0 m/s**. Even if only 5 cm is coasting, that
is 0.54 m/s. Bauza & Rodriguez §VII-C: "the performance peaks at around 50-80mm/s and degrades after
that. We conclude that the quasi-static assumption does not hold after that." Yu et al. call 20 mm/s
"quasistatic speed"; their dataset tops out at 500 mm/s.

Patchworks runs an order of magnitude past the quasi-static regime and past the fastest data in the
MIT dataset. Nothing *requires* quasi-static pushing and the spec never claims it — but the
consequence is that this world is closer to air hockey than to pushing, and none of the quasi-static
apparatus (motion cones, stable pushing, the Lynch–Maekawa–Tanie closed form) describes it. Worth
stating so nobody reaches for it later.

**1.4 — Paddle–puck friction is more than double the measured real value.** The `<default>` geom
friction is `0.6 0.005 0.0001` (line 21) and MuJoCo combines by element-wise maximum for
non-prioritised geoms, so paddle–puck sliding friction is 0.6. Yu et al. §III measured "the friction
coefficient between the pusher and the object is approximately 0.25". Steel-on-steel is not a rounded
rubber paddle, so 0.6 is not indefensible — but it biases contact toward sticking rather than
sliding, which is the regime producing the clean repeatable pushes the spec measured. A calibration
note, not an error.

---

## 2. Push dynamics: frictionloss, zero gravity, and predictability

### Sources

- MuJoCo documentation, read directly: XML Reference (`body/joint`); Computation chapter (Constraint
  model → Friction loss; Constraint solver, eqs. 1, 7, 9); Modeling chapter (Solver parameters →
  Friction; Contact parameters).
- Yu et al. 2016 §V, §VI, §VII (above).
- Bauza, M. & Rodriguez, A. (2017). "A Probabilistic Data-Driven Model for Planar Pushing."
  *ICRA 2017*. arXiv:1704.03033.
- Ma, D. & Rodriguez, A. (2018). "Friction Variability in Planar Pushing Data." arXiv:1802.10089
  (abstract read verbatim; body not read).
- **Cited second-hand only**, as characterised by Yu et al. §II — SAGE returned HTTP 403:
  Goyal/Ruina/Papadopoulos 1991 (limit surface), Lee & Cutkosky 1991 (ellipsoidal approximation),
  Howe & Cutkosky 1996, Lynch/Maekawa/Tanie 1992.

### Is zero-gravity frictionloss a recognised trick or a liability?

**Both, and MuJoCo's own docs say so.** The Computation chapter's *Friction loss* section:

> "Friction loss is also known as dry friction, or static friction, or **load-independent friction
> (in contrast with Coulomb friction which scales with normal force)**. … it acts preemptively
> before the onset of motion, and so it cannot be modeled as a velocity-dependent force. Instead it
> is modeled as a constraint, namely an upper limit on the absolute value of the force that friction
> can generate."

And on free joints:

> "It could be argued that friction loss in free joints should not be allowed. However we allow it
> because it can model useful **non-physical** effects, such as keeping an object in place until
> something pushes it with sufficient force."

That is close to a verbatim description of the puck model, and MuJoCo's authors label the class of
use **non-physical**. It is a sanctioned trick, and the documentation tells you it is not physics.

### Correspondences

- The spec's construction is faithfully implemented and the arithmetic checks out exactly for all
  three pucks.
- μ = 0.3 sits at the top of a plausible measured band. Yu et al. §V-2 report DCoFs of 0.13–0.15
  (`abs`), 0.12–0.16 (`delrin`), 0.24–0.28 (`plywood`), 0.28–0.29 (`pu`) — roughly a
  plywood/polyurethane table.
- The rationale — "Going 3D means *adding joints and turning gravity on* — removing constraints" — is
  a real and unusual property of this construction, and nothing argues against it. The frictionloss
  rows simply disappear when the planar joints are deleted.
- MuJoCo explicitly endorses "keeping an object in place until something pushes it with sufficient
  force", which is precisely the spec's measured >2 N break-away.

### What argues against a choice already made

**2.1 — `armature` leaks onto the puck joints. This is a bug, not a design question.**

`arena.xml:22` declares `<joint armature="0.01"/>` as a **classless global default**. The `puck`
class (line 26) sets only geom attributes. The puck joints (lines 114–116, 123–125, and puck 2's)
declare no class and override only `damping` and `frictionloss` — so **they inherit
`armature="0.01"`**. MuJoCo's XML reference defines armature as inertia "usually due to a rotor …
spinning faster than the joint itself due to a geared transmission", noting it "significantly
improves simulation stability". A free-sliding puck has neither rotor nor transmission.

For the hinge joints (units kg·m²), against true disc inertia I = ½mR²:

| | true I (kg·m²) | + armature | ratio |
|---|---|---|---|
| puck 0 | 3.06e-5 | 1.003e-2 | **327×** |
| puck 1 | 1.01e-4 | 1.010e-2 | **100×** |
| puck 2 | 3.03e-4 | 1.030e-2 | **34×** |

A 1 N tangential force at puck 0's rim produces α ≈ 3.5 rad/s² instead of ≈1140 rad/s². For the slide
joints the quantity is a mass: each puck carries +0.01 kg of fictitious mass (effective
0.06/0.11/0.21 kg), giving coasting decelerations of 2.45/2.67/2.80 m/s². So the ~14% by which the
three pucks are distinguishable during coasting is *entirely an artefact of a stability default
inherited from the arm* — this is the mechanism behind 1.2.

Combined with 1.1, the hidden-rotation story is undermined twice over: orientation is dynamically
inert, *and* the spin dynamics that would have to reveal it are dominated by a fictitious inertia.

**2.2 — The friction limit set in the plane is a square, not a disc.** (spec lines 14–15)

Each puck has two independent slide joints. Computation eq. (9): "For friction loss we have the box
constraint |λ_F| ≤ η **applied element-wise**", and for joints "it is applied independently to all
degrees of freedom of the affected joint." So the admissible friction force set is `‖f‖_∞ ≤ μmg` — a
square — not the disc `‖f‖₂ ≤ μmg` that "μmg table friction" implies.

A puck sliding along +x is opposed by 0.147 N; along the 45° diagonal by 0.147·√2 = 0.208 N, **41%
more**. The anisotropy is fixed in the *world* frame with 4-fold symmetry and rotates with nothing.
Yu et al. §V-4 measured real anisotropy directly: "For pu, **the ratio between the largest friction
and the smallest is around 3/2**, which is a significant difference" — and pu was the worst of four
materials. Patchworks has silently baked in a ratio of 1.41, comparable to the worst real material
the authors singled out, into a model whose spec line reads as an isotropic scalar μ.

Ma & Rodriguez (arXiv:1802.10089) argue this exact effect matters: anisotropic friction "can
originate biases in the collected datasets, resulting in deterioration of trained models", and with
material non-homogeneity explains "a significant fraction of the observed unconventional phenomena",
including stochasticity and multi-modality.

**2.3 — The limit surface is a rectangular box in wrench space.**

The pucks have three independent friction-loss rows (`p*_x`, `p*_y`, `p*_r`), each an independent box
constraint, so the admissible friction *wrench* set is a rectangular box in (f_x, f_y, m_z). At a
corner of that box a puck simultaneously resists maximum translational friction on both axes **and**
maximum torsional friction. Physically it cannot: full translational slipping exhausts the friction
budget and leaves no torque capacity.

Yu et al. §VI is the empirical check. On maximum dissipation: "All materials except for pu, yield ΔP
very close to 0", as the principle requires. On the ellipsoid: "the real limit surface is closer to
thicker noisier ring. We can also see that the underlying curve of the data resembles an ellipse but
not exactly." So the real object is a noisy ellipsoid and the sim's is a box — a strictly *worse*
approximation than the ellipsoid the literature already treats as a computational compromise. The
regime where they differ most is coupled translation + rotation, i.e. every off-centre push, and
precisely the regime the spec nominated as carrying its hidden variable.

**2.4 — Sub-threshold "static equilibrium" is soft and may creep.** (spec lines 103–105)

MuJoCo's friction-loss constraint is soft. Modeling → Friction: for friction-loss rows the position
residual r ≡ 0, so impedance is always `solimp[0]` (default 0.9), stiffness k = 0, and "the dynamics
are first-order (exponential decay of constraint velocity, no spring)." Computation eq. (1),
`ac + d·(bv + kr) = (1 − d)·au`, means with d = 0.9 a tenth of the unconstrained acceleration leaks
through even when the constraint is active. The steady state is therefore not v = 0 but a small
nonzero creep.

**This is derived from the documented equations, not measured** — no macOS x86_64 `mujoco` wheel was
reachable at any version (the prototype README documents the same problem and pins 3.10.0). It
matters because the task horizon is 60 s: a creep of even 1 mm/s integrates to 6 cm, comparable to a
zone radius of 0.075 m. One measurement settles it — hold a constant 1.9 N tip force against puck 2
for 60 s and log displacement.

**2.5 — The held-out sector straddles the anisotropy diagonal.** The friction square's corners sit at
45°/135°/225°/315° in the world frame; the held-out wedge (30°–75°) is centred on 45°. The wedge is
defined over puck *position*, not push *direction*, so this is not asserted as a confound — but
pushes from the annulus toward a zone are broadly radial, and radial directions in that wedge do
straddle the diagonal where friction is 41% higher. Cheap to measure (sweep push direction 0°→90° at
fixed everything else, plot stopping distance), and worth measuring before the split is treated as
clean.

### Predictability of pushing

**Repeated identical pushes do not give identical outcomes.** Yu et al. §VII repeated *one* push
2,000 times (`rect1`, contact halfway to the edge, normal angle, 20 mm/s, 150 mm displacement), on
each of 4 surfaces. Table IV translation std: `abs` 5.5 mm (7.1%), `delrin` 3.4 mm (5.2%), `plywood`
8.1 mm (8.0%), `pu` 11.7 mm (12.5%); rotation std 1.3°–4.5°. Vicon accuracy is below 0.5 mm and 0.5°
and the robot's pusher pose is accurate to 0.1 mm (§IV), so **this is process noise, not sensor
noise**. And it is not Gaussian: "The distribution of final poses seems to have at least three modes,
and its shape is clearly not Gaussian." Their conclusion verbatim: "even when trying to replicate the
same initial conditions with an accurate vision system and an accurate robot, a determined pushing
interaction yields appreciable and structured uncertainty at the outcome."

**The variability is input-dependent by an order of magnitude.** Bauza & Rodriguez Fig. 1 shows three
pushes, each repeated 100×, that are "convergent", "divergent", and "multi-modal". §VI-A: "the
magnitude of the noise is in between **10% and 40% of the magnitude of the expected output**". §VII:
variability "can vary **up to an order of magnitude** with the pushing action".

**Data-driven beats analytical fast.** §VI-C: GPs and VHGPs "outperform the analytical model after
100 samples approximately", saturating at ~10³ samples — "data that can be captured in about
5 minutes".

### Correspondences

- The spec's "epistemic pressure … is over **dynamics**" is emphatically what the literature says.
  Bauza & Rodriguez's input-dependent noise is a direct argument that a predictor of pushing must
  model its own uncertainty as a function of the action — a natural thing to ask of a cell whose job
  is to advance its features one step in time, and which **disagreement**, measured in an edge stalk,
  is well positioned to expose.
- The 15/48 scripted-controller solve rate reads very differently against Yu et al.'s finding that
  identical pushes diverge: a low open-loop success rate is what the pushing literature predicts, and
  the spec is right to call it a lower bound rather than a baseline.
- Bauza & Rodriguez's ~100-sample crossover and ~10³-sample saturation are a useful order-of-magnitude
  check on how much interaction a cell needs before its local model of one puck is worth anything.

**2.6 — The sandbox is more deterministic than the phenomenon it stands for.**

The pucks are perfect uniform-density cylinders, frictionloss is an exact constant per puck, the
surface has no spatial variation, no break-in, no speed dependence, and MuJoCo's solver is
deterministic. Repeated identical pushes from an identical snapshot give bit-identical outcomes. Real
ones give 1.6–12.5% translation std with at least three modes, and per-action noise varying by an
order of magnitude.

The spec makes reproducibility a feature, and for debugging it plainly is one. But it means the
sandbox offers a *deterministic* dynamics-learning problem while being motivated by dynamics as the
site of epistemic pressure. A cell here can in principle drive its prediction error to zero; a cell
in the real version of this world provably cannot. Given `CONTEXT.md`'s **disagreement** — "never
fully cleared" — a world in which it *can* be fully cleared is a weak proving ground for that claim.

Two cheap ways to buy back irreducible uncertainty without giving up snapshot/restore: (a) resample
per-puck frictionloss from a narrow distribution at each `reset()` rearrangement, which would give
the agent a genuinely unmodellable component *and* a genuinely inferable-from-contact one — a much
better fit to what the spec wanted hidden rotation to do; or (b) make frictionloss a function of
position, which is exactly the spatial DCoF variation Yu et al. §V-1 mapped, and which would restore
some of the spatial epistemic pressure the spec notes is currently absent.

This is an unstated consequence rather than a defect, and the spec's own house style — "This is
recorded as a consequence, not a decision" — is the right register for it. Note also that at
ballistic speeds (1.3) the real phenomenon's variability is *higher*, so the gap widens rather than
narrows.

---

## 3. Goal-as-perception

### Sources

- Schaul, Horgan, Gregor, Silver (2015). "Universal Value Function Approximators." *ICML*, PMLR v37.
- Andrychowicz et al. (2017). "Hindsight Experience Replay." arXiv:1707.01495v3.
- Nair, Pong, Dalal, Bahl, Lin, Levine (2018). "Visual Reinforcement Learning with Imagined Goals"
  (RIG). arXiv:1807.04742.
- Tassa et al., DeepMind Control Suite (above); `dm_control/suite/reacher.xml` and `reacher.py`.
- Minigrid official docs (`MiniGrid-Empty`).
- Stone, Ramirez, Konolige, Jonschkowski (2021). "The Distracting Control Suite." arXiv:2101.02722.
- Torresan, Kanai, Baltieri (2025). "Prior preferences in active inference agents: soft, hard, and
  goal shaping." arXiv:2512.03293v1.
- Matsumoto & Tani (2020). "Goal-Directed Planning for Habituated Agents by Active Inference Using a
  Variational Recurrent Neural Network" (GLean). arXiv:2005.14656.
- Oliver, Lanillos, Cheng (2019/2021). "Active inference body perception and action for humanoid
  robots." arXiv:1906.03022, IEEE TCDS.
- Rao, Gklezakos, Sathish (2022). "Active Predictive Coding." arXiv:2210.13461.
- Mendonca, Rybkin, Daniilidis, Hafner, Pathak (2021). "Discovering and Achieving Goals via World
  Models" (LEXA). arXiv:2110.09514.
- Burda et al. (2018). "Large-Scale Study of Curiosity-Driven Learning." arXiv:1808.04355.
- Friston, Thornton, Clark (2012). "Free-energy minimization and the dark-room problem."
  *Front. Psychol.* 3:130.
- **Partially verified:** Sermanet et al., arXiv:1612.06699 and arXiv:1704.06888 — titles and framing
  confirmed from arXiv listings, section text not extracted; claims restricted to what abstracts
  support. Both derive a reward *from* vision rather than communicating a goal through the live
  observation, so the line is orthogonal.

### The baseline: the goal is normally a separate, addressable input

UVFA §3 fixes the field's default shape: "the most direct approach, F : S × G ↦ R simply concatenates
state and goal together as a joint input"; the alternative is "a two-stream architecture … φ : S ↦ Rⁿ
and ψ : G ↦ Rⁿ". HER §2.4 restates it — "At every timestep the agent gets as input not only the
current state but also the current goal" — with Alg. 1 explicit: `a_t ← π_b(s_t || g)`. RIG is the
visual case and still separate-channel: policy π_θ(z, z_g) with reward `r(s,g) = −‖z−z_g‖`; the goal
is a *second image*, encoded by the same VAE, concatenated in latent space. RIG contains no
discussion of goal/scene ambiguity, because the two never share a frame.

### Is the goal ever rendered into the same frame?

**Yes, and it is common — but always with a redundant separate channel.** `dm_control/suite/reacher.xml`
defines `<geom name="target" pos="0 0 .01" material="target" type="sphere" size=".05"/>` — a normal
visible geom, not an overlay — and `reacher.py`'s `initialize_episode` randomises its position. The
`pixels.Wrapper` can replace features with images entirely, so a pixel-only agent on `reacher` has
only the rendered sphere to tell it what is wanted. But `get_observation` *also* returns
`obs['to_target']`, and the reward is computed from the true target position. MiniGrid is the same
story: the green goal square is an object type inside the encoded grid, *and* there is a mission
string, "get to the green goal square".

**No primary source was found in which an in-frame goal marker is the sole goal channel** — no goal
vector, no task id, no reward. Patchworks' configuration appears genuinely unattested rather than
merely uncommon.

On disambiguation specifically: **this could not be verified to have been studied at all.** RIG does
not raise it; dm_control does not; MiniGrid sidesteps it by giving the goal tile its own object-type
index. The nearest indirect evidence is the Distracting Control Suite: "current RL methods for
vision-based control perform poorly under distractions, and … their performance decreases with
increasing distraction complexity", and "combinations of multiple distraction types are more
difficult than a mere combination of their individual effects." A lit target zone is not a distractor
— it is task-*defining* — but it is content the agent must learn to treat differently from the pucks
it must move. **Read this as "no evidence either way", not as support.**

### Clamped predictions as goals

**GLean is the real instance, and it validates the spec's phrase almost verbatim.** §2.3: GLean
"attempts to minimize the errors at the initial timestep and the goal timestep" via error regression
(Fig. 4b); the accuracy term is "calculated as the summation of prediction error in the initial
(t = 1) and distal (t = T) steps", and "except for the first timestep, the posterior distribution is
conditioned only on the prediction error at the goal." That is exactly "a goal expressed as a clamped
prediction" — and it works only because the goal lives in observation space. **Under a goal vector
there is nothing to clamp, so the spec's causal claim is correct.**

**But clamping is the non-standard end of the spectrum.** Torresan et al. Table 1 factors preferences
along two axes; on strength, hard preferences concentrate "most of the probability mass … on a single
state … thus making P*(S_t) an approximate delta distribution", versus soft goals spreading mass over
"two or more states". Their reported findings: §4.1, hard+shaped goals perform best "while sacrificing
learning about the environment's transition dynamics"; agents without goal shaping "eventually
discover all the six possible trajectories to the goal … they might be more robust to perturbations,
e.g., to one or more paths to the goal becoming obstructed"; soft goals without shaping were
"marginally more capable of reaching the final goal state". §4.2 gives a pathology unique to the
peaked end: "risk will be relatively low when computed between a low-entropy preference distribution
… and a high-entropy variational distribution Q(S_t|π_k), for a task-failing policy π_k that was
rarely attempted" — a sharply clamped goal makes under-explored, task-failing policies score *well*.

**Two corrections to what the surrounding literature might be assumed to say.** Oliver/Lanillos/Cheng
do **not** clamp: the goal enters as a gain-scaled attractor inside the dynamics prior
(`A(µ,ρ) = ρ₄[ρ₁₋₃ − g_v(µ)]`, Eq. 12), i.e. a graded preference. And **Rao's Active Predictive
Coding does not clamp either** — despite the name, its planning experiments use episodic REINFORCE
with explicit extrinsic reward ("+10 reward for goal; −0.1 per primitive action") and one-hot subgoal
vectors. It is a goal-vector architecture wearing predictive-coding clothes.

### Correspondences

- **GLean validates the spec's stated end-state**, in a published robotics system (simulated 8-DoF
  Torobo arm). The spec's claim that goal-as-perception is what *enables* clamping is correct.
- **Patchworks already has the ingredient GLean identifies as essential.** GLean's advantage over a
  plain forward model comes from a learned prior constraining the search. Patchworks' analogue is
  that disagreement is penalised rather than enforced (`01-cell-and-sheaf.md`) and never cleared — a
  clamped goal enters as a persistent disagreement term the cells descend on, not a hard projection.
  `01`'s explicit refusal of "a hard projection onto the consistent subspace" is, in active-inference
  terms, a refusal of the delta-preference limit — and Torresan et al. §4.1–4.2 supply independent
  evidence that the softer choice is right.
- **Rao's goal-changing result supports `retarget()` as a discriminating test**: flat RL "does not
  recover" after a goal change, while the hierarchical predictive agent "is able to cope with goal
  changes". That is the acceptance demo's stated purpose.
- **The `perturb`/`retarget` split is independently corroborated** by Torresan et al.'s finding that
  robustness to "paths to the goal becoming obstructed" is a property of how much transition-dynamics
  learning the preference specification permitted — the two perturbation types load different parts
  of the model.
- **UVFA §4.3 (Extrapolation)** is the closest thing in the literature to the spec's rationale:
  generalising "to unseen goals in completely new parts of space" is argued feasible "if states are
  represented with the same features as goals". Patchworks takes the limit case — state and goal are
  not merely represented with the same features, they *are* the same features, on the same node
  stalk.
- **LEXA** is direct existence-proof support for the most exposed commitment: goals specified
  perceptually, no environment reward, evaluated zero-shot, in image-based manipulation — 40 test
  tasks, first success on Kitchen.
- **Burda et al.'s stochasticity failure is anticipated rather than exposed.** Their noisy-TV result
  — "the agent will seek out transitions with the highest entropy" — is designed around: pucks are
  deterministic rigid bodies, there is no noise channel. (Though see 2.6: the design overshoots into
  *too little* stochasticity.)
- **The dark-room problem is partly designed out.** `reset()` rearranges the world without resetting
  the agent, so the environment is never allowed to become static; an agent that stops acting still
  experiences layout changes it did not predict. A structural answer, not a hope.

### What argues against a choice already made

**3.1 — The held-out slice and the clamped-goal mechanism point in opposite directions.**

GLean Fig. 13 tests "goals set in an untrained region" and reports: "GLean is not able to reach the
specified goals. In particular, it can be observed that the trajectories cannot go straight at the
branching point … the learned prior strongly prefers either turning left or right", concluding that
GLean "is more likely to generate goal-directed plan trajectories within well habituated areas." The
authors **endorse** this — plans are generated "within the boundary of well-habituated trajectories",
and the abstract credits the advantage over a forward model to exactly that confinement.

The sampler's held-out slice — pairs `(puck 0 → zone 2)`, `(puck 2 → zone 0)`, and the 30°–75° wedge
— is *definitionally* that region. If the goal mechanism is a clamp, the most closely matched primary
result predicts the held-out slice fails: not because generalisation is hard in general, but because
prior-constrained error regression is designed to refuse it. **This is the one place where two
already-made choices point in opposite directions, and neither document names the other.**

**3.2 — "Clamped prediction" is stated as if it were the standard mechanism.** The honest version is
that goal-as-perception is what makes *either* a clamp or a graded prior-preference expressible, and
the clamp specifically is the variant with the worst reported exploration and robustness profile.
`01-cell-and-sheaf.md` already refuses the hard-projection limit on parallel grounds, so the
architecture is consistent — only `03`'s justifying sentence is stronger than the architecture it
justifies. A wording fix, not a design problem.

**3.3 — Target/body resolution at 64×64.** dm_control §6 names the failure that actually killed
pixel-only runs: "some of the failure cases are likely due to the difficulty of positioning a camera
that simultaneously captures both the navigation targets as well as the details of the agents body:
e.g., in the case of swimmer:swimmer6 and swimmer15 as well as fish:swim". Patchworks asks one 64×64
top-down camera to resolve simultaneously: 3-link arm configuration, three pucks (r =
0.035–0.055 m), three zones (r = 0.075 m), *and* which zone is lit. The spec records the
orientation-marker exposure but not this one — same category, and the existing note already
establishes the right register.

**3.4 — Hallucinating the goal is a documented failure of clamping.** GLean's forward-model baseline:
"improbable trajectories that seemingly reach given goals can be generated by arbitrarily combining
motor states in sequences that happen to minimize the distal goal error." In Patchworks' vocabulary:
a clamped cell whose disagreement is minimised by the graph *re-describing* the scene rather than by
the arm moving. Nothing in `01`/`02`/`03` names this risk; `01`'s known-exposure list covers recurrent
failure modes but not this.

**3.5 — "No reward channel" conflates two claims.** No primary source demonstrates specified-goal
reaching with **no scalar objective at all**: LEXA §2.4 defines an internal cosine-similarity goal
reward plus an ensemble-disagreement exploration reward; RIG uses `−‖z−z_g‖`; GLean descends a distal
prediction-error objective; active inference uses expected free energy. Patchworks' **disagreement**
does function as an objective. So the well-attested commitment is "no *environment* reward channel";
"no scalar objective" is unattested. A documentation-precision issue — but it matters, because a
reader taking the strong reading would conclude the sandbox asks for something nobody has done.

**3.6 — Non-episodic clamping is unattested.** Every source in this area is episodic, and GLean in
particular assumes "the distal step is at a fixed point in time; in practice, if the goal is reached
early, the agent should remain stationary until the final timestep." **What "clamp the prediction at
t = T" means when there is no T is unaddressed anywhere in `02` or `03`.**

**3.7 — Precision balancing is where active-inference reaching actually breaks.** Oliver et al. §VII:
"Sensor variances, action gains and velocity limits were experimentally tuned"; §VI.A reports failure
at σ = 40° encoder noise, a finger/forearm body-illusion from mis-set relative precisions, and that
the method "is prone to local minima." Patchworks fixes four heterogeneous modalities (`qpos`,
`qvel`, `touch`, `image`) with relative precision implicit in learned restriction-map norms rather
than exposed. A defensible bet, but an unrecorded exposure with primary-source evidence behind it.

**3.8 — HER's remedy is structurally inexpressible here.** Hindsight relabelling needs `m : S → G`
(§3.2), and there is no `m` when the goal is a lit region of a uint8 render rather than a coordinate.
Not fatal — Patchworks has no replay buffer and no reward to relabel — but the single most effective
known remedy for sparse goal-reaching is unavailable, which is worth recording rather than treating
as a non-issue.

---

## 4. Continual environment contracts

### Sources

- Gymnasium source and docs, read directly: `gymnasium/core.py` (`Env.step`, `Env.reset` docstrings);
  `utils/env_checker.py` (`check_reset_seed_determinism`); `utils/passive_env_checker.py`;
  `wrappers/common.py` (`TimeLimit`, `RecordEpisodeStatistics`, `Autoreset`, `OrderEnforcing`);
  `envs/registration.py`; `vector/vector_env.py` (`AutoresetMode`).
- MuJoCo docs, read directly: `mjtState` enum (APItypes); Programming → Simulation, "State and
  control"; `mj_getState`/`mj_setState`/`mj_stateSize`.
- Sharma et al. (2021/22). "Autonomous RL: Formalism and Benchmarking" (EARL). arXiv:2112.09605.
- Gupta et al. (ICRA 2021). "Reset-Free RL via Multi-Task Learning." arXiv:2104.11203.
- Sharma et al. (NeurIPS 2021). "Autonomous RL via Subgoal Curricula" (VaPRL). arXiv:2107.12931.
- Wolczyk et al. (NeurIPS 2021). "Continual World." arXiv:2105.10919.
- Powers et al. (2022). "CORA." arXiv:2110.10067.
- Laskin et al. (2021). "URLB." arXiv:2110.15191.
- **Listed but not verified:** Zhu et al. (ICLR 2020), "The Ingredients of Real World Robotic RL",
  arXiv:2004.12570 — PDF text could not be extracted and no clean HTML rendering was found. No claim
  is made about its contents.

### What the Gymnasium contract actually requires

*On `reward`.* `Env.step` types it only as `reward (SupportsFloat)`. There is **no documented lower
bound, no requirement that it vary, and no requirement that it be non-zero.**
`env_step_passive_checker` warns on exactly three conditions: non-numeric type, NaN, and inf. A
constant `0.0` trips none. **Constant-zero reward breaks no documented Gymnasium invariant.**

*On `terminated`/`truncated`.* The passive checker warns only on non-boolean types. Permanently-False
flags break no documented invariant either — but they make several stock wrappers silently inert, and
one actively violates the spec (4.2 below).

*On `info`.* "Contains auxiliary diagnostic information (helpful for debugging, learning, and
logging)." The spec's use of `info` for privileged truth is **the documented intent of the field,
verbatim** — this deviation is not a deviation at all.

*On `reset`.* Documented as "Resets the environment to an initial internal state", and the official
custom-env tutorial says it "starts a new episode". Both phrasings assume the thing this `reset()`
deliberately does not do.

### Do reset-free methods smuggle resets back for evaluation?

**Yes, uniformly. Every source checked reintroduces a state-distribution reset for evaluation, and
says so.**

- **EARL** keeps two metrics. Deployed-policy evaluation is explicitly episodic: "Policy evaluation
  J_D(π_t) is carried out every 10000 training steps … by running the policy π_t 10 times, **starting
  from s_0 ∼ ρ for every trial**", with "these roll-outs are only used for evaluation, and are not
  provided to the algorithm." The continuing metric ℂ(𝔸) is offered *alongside*, not instead.
- **Gupta et al.** train with no episodic resets via a task-graph where tasks reset each other, but at
  evaluation "roll out its final policy starting from states randomly sampled from the distribution
  induced by all the tasks that can transition to the task under evaluation" — a learned, task-composed
  reset, but a reset.
- **VaPRL** does not even remove resets from training: "the agent was provided an environment reset
  after a few 100k steps" (H_T = 200k–400k), and evaluates by "resetting to a state from the initial
  state distribution ρ" for 10 trials. Its own framing: reset-free is a *training* property.
- **URLB** withholds extrinsic reward for the whole pre-training phase, then brings it back:
  "fine-tune the agent for 100k steps and measure its performance on the downstream task."
- **Continual World** and **CORA** both require externally-supplied task boundaries, and every
  headline metric (forgetting `F_i = p_i(i·Δ) − p_i(T)`, forward transfer) is *defined in terms of
  them*. There is no boundary-free version of the metric.

### Correspondences

- The spec's `reset()` — rearrange the world, leave the body — is the same move as Gupta et al.'s
  task-graph reset, arrived at from the other side. Both refuse the classical "teleport the robot to
  a canonical pose".
- **Patchworks goes further than any source found on one axis: nothing cited announces the change
  less.** EARL, VaPRL and Gupta all condition on an explicit goal or task command; Continual World
  and CORA hand agents explicit boundaries. "No observation component announces that anything
  happened" has no direct precedent. **That novelty claim survives the pass.**
- Monotonic physics time corresponds to EARL's continuing regime and VaPRL's H_T stretches, with
  H_T = ∞.
- Privileged `info` "for logging and evaluation only" is structurally identical to EARL's "only used
  for evaluation, and are not provided to the algorithm."
- **Snapshot/restore is MuJoCo's own first-class API**, not a workaround: `mj_getState`/`mj_setState`
  with `mj_stateSize`, and the docs note state "is entirely encapsulated in the `mjData` struct",
  which "together with the deterministic pipeline" makes resetting state well-defined. Contact forces
  are recomputed, not stored — which retires one worry the spec might otherwise have had.

### What argues against a choice already made

**4.1 — The snapshot field list is insufficient for bitwise reproducibility, and MuJoCo's docs say so
in as many words.** (spec line 138)

The list `(qpos, qvel, ctrl, clock, task, RNG)` omits **`qacc_warmstart`** (`mjSTATE_WARMSTART`).
From the simulation page:

> "The other case where warmstarts are critical is if perfect numerical reproducibility is required,
> **when loading a non-initial state** (since the initial state is always cold-started). Note that
> even though their effect on physics is negligible, many physical systems will accumulate small
> differences exponentially when time-stepping, quickly leading to divergent trajectories for
> different warmstarts."

"Loading a non-initial state" is precisely and only what this sandbox ever does — there is no episode
boundary, so **every restore is a non-initial-state load.** Warm-start is the one field whose
omission the docs specifically flag as breaking reproducibility, and it is the one field the list
omits. The spec's own framing ("Because there is no episode boundary to restart from…") makes this
the load-bearing case, not an edge case.

Also absent, in decreasing relevance: `act` (irrelevant if all actuators are direct-drive torque,
which the motor surface suggests but the spec does not say); `qfrc_applied`/`xfrc_applied` (relevant
precisely because `perturb(puck, xy)` exists — if teleporting is ever implemented by applied force,
that force is state); `eq_active`; `userdata`; `plugin_state`.

The clean fix is one line: snapshot **`mjSTATE_INTEGRATION`** — a single named signature the docs
define as "the union of all the above `mjData` fields … the entire set of inputs to the forward
dynamics" — plus task and RNG, which MuJoCo does not know about. That converts an enumeration that
can drift into a named engine constant that cannot.

**4.2 — "`terminated` and `truncated` are always `False`" is true of the env and false of `make()`.**

`registration.py` wraps in `TimeLimit` whenever `max_episode_steps` is passed to `make()` or carried
on the registered spec. `TimeLimit` sets `truncated = True` at the boundary **and calls `reset()`** —
which in this env means an unannounced world rearrangement rather than a restart. The contract holds
only if the registration sets `max_episode_steps=None` *and* callers are told never to pass it. The
spec does not name this.

Two further consequences: `RecordEpisodeStatistics` populates its info key only under
`if terminated or truncated:`, so it will **never emit a single statistic** for the entire run; and
for `VectorEnv`, with flags pinned False every `AutoresetMode` collapses to `DISABLED` in effect, so
a training loop assuming sub-envs periodically hand back fresh initial states gets none. Nothing
errors — the loop quietly runs one infinite trajectory per sub-env.

**4.3 — `check_env` will fail by construction.** `check_reset_seed_determinism` calls `reset(seed=123)`
twice and asserts observation equivalence ("Using `env.reset(seed=123)` is non-deterministic as the
observations are not equivalent"). Since `reset()` never resets the arm, the two observations differ
by design.

The documented escape hatch is worse than the problem. The check is skipped only when
`env.spec.nondeterministic is True`, and `EnvSpec.nondeterministic` is documented as "If the
observation of an environment cannot be repeated with the same initial state, random number generator
state and actions." This env **is** repeatable given the same initial state and RNG state — that is
the entire point of snapshot/restore. Setting the flag would assert something false about the physics
to silence a check about `reset()`. Either the spec accepts that `check_env` fails and says so, or it
misuses a registration flag.

**4.4 — `info` is named as being "for evaluation" but no evaluation protocol is defined.** Every
primary source reintroduces a state-distribution reset *for evaluation specifically*. The spec
provides the mechanism (`reset_arm`) and then rules it out of "normal operation" without defining
what replaces it. Combined with `reward` always `0.0`, there is **no scalar at all** on which two
runs can be compared: no return, no episode, no success-per-trial denominator. `info` does carry
"whether the goal is satisfied", which is the raw material for a success rate — but a success rate
needs trials, and trials need a defined start. This is the most substantive gap the pass found in
this area.

**4.5 — The boundary-agnostic regime has no inherited metric shape.** Not an error, but worth
recording in the same register as the 64×64 note: continual-RL benchmarks explicitly avoid this
regime partly because their metrics stop being computable without boundaries. Patchworks retains the
boundaries in `info`, so the metrics remain *computable* — the exposure is that the agent is asked to
do something no cited benchmark asks of its agents, with no established metric shape to inherit.

---

## 5. Held-out combinatorial slices

### Sources

- Ruis, Andreas, Baroni, Bouchacourt, Lake (NeurIPS 2020). gSCAN. arXiv:2003.05161.
- Lake & Baroni (ICML 2018). SCAN. arXiv:1711.00350.
- Keysers et al. (ICLR 2020). CFQ. arXiv:1912.09713.
- Johnson et al. (CVPR 2017). CLEVR, incl. CoGenT. arXiv:1612.06890.
- Mendez, van Seijen, Eaton (NeurIPS D&B 2022). CompoSuite. arXiv:2207.04136.
- Yu, T. et al., Meta-World (above).
- Zeng et al. (CoRL 2020). Transporter Networks / Ravens. arXiv:2010.14406.
- Pumacay et al. (2024). THE COLOSSEUM. arXiv:2402.08191.
- **Listed but not verified:** Xing et al., KitchenShift — OpenReview served a bot-check page and the
  abstract does not state whether shift factors are applied singly or jointly. No claim is made.

### What the sources say

**The near-universal shape is one axis per split, reported separately.**

- **gSCAN** is the most informative source, because it contains both a spatial holdout and
  attribute-pair holdouts in the same benchmark, reported side by side. Split B holds out yellow
  squares as target; C holds out red squares (a novel colour-shape combination); D holds out targets
  "located to the south-west". Baseline exact-match: **B = 54.96 ± 39.39, C = 23.51 ± 21.82,
  D = 0.00 ± 0.00.** The spatial holdout is not merely harder — it is *categorically* different, a
  total failure against partial successes. Their diagnosis: the model knew *where* but not *how* —
  "the agent usually walks all the way west (or south) and then fails to turn to the target object …
  the agent knows where to go … just not how to get there", reaching the correct row or column
  63.10% of the time. **A spatial holdout and a combinatorial holdout demonstrably do not test the
  same thing.**
- **SCAN** shows the same within a single split *type*: two instances of the primitive-command split
  give 90.3% ("turn left") and 1.2% ("jump").
- **CFQ** states the principle explicitly: maximise compound divergence "while guaranteeing a small
  atom divergence" (𝒟_A ≤ 0.02), because principle 1 "aims to guarantee that the experiment is
  exclusively measuring the effect of the difference in the way atoms are composed to form compounds
  (**rather than some related but different property such as domain adaptation on the distribution of
  the atoms**)."
- **CLEVR CoGenT** holds out along exactly one axis (a shape×colour pairing), reported separately as
  A→A vs A→B.
- **CompoSuite** has four axes (4⁴ = 256 tasks) and holds out **one element of one axis at a time**.
- **Meta-World** keeps the two kinds in *separate benchmarks* — ML1 is the parametric/spatial axis
  ("50 held-out positions"), ML10/ML45 the non-parametric/identity axis — and names the distinction
  explicitly.
- **Transporter Nets / Ravens** is the cautionary outlier: held-out objects exist (14 training, 6
  held out) but Table 2 gives unified success percentages, so the seen/unseen distinction survives
  only in prose. **Exactly the failure mode a unioned split invites.**
- **THE COLOSSEUM** does both and shows why both are needed: 14 factors applied individually *and* an
  "All Perturbations" condition, with individual factors costing 30–50% and all-perturbations "≥75%
  decrease". The combined condition is reported **in addition to**, not instead of, the per-factor
  ones.

### Correspondences

- A task as a **(layout, target puck, target zone)** triple with a 3×3 pair grid is structurally the
  same object as CompoSuite's element grid and CLEVR CoGenT's shape×colour grid. Holding out two
  cells while training on the rest is the standard compositional-holdout shape and satisfies CFQ's
  atom-coverage principle cleanly: every puck and every zone is seen, only two *compounds* withheld.
  **This half of the design is textbook.**
- The spec's insistence that pucks differ so "the dynamics are not one model with a colour attached"
  is precisely what CompoSuite argues for in choosing objects requiring "orthogonal grasping
  orientations" rather than recoloured copies. Both refuse a holdout a colour-invariant model could
  pass trivially. (Though see 1.2 and 2.1: the prototype currently undercuts this.)
- The sector holdout corresponds to gSCAN's split D and Meta-World's ML1 parametric holdout — a
  recognised split type with strong precedent, **as a split of its own**.

### What argues against a choice already made

**5.1 — `split="heldout"` as a *union* yields an unattributable number.** (spec line 119)

Because it is a union rather than an intersection, a draw from `heldout` lands in one of three
distinct regimes — pair-violating only, sector-violating only, or both — and a single aggregate
averages across all three. gSCAN's 54.96/23.51/0.00 and SCAN's 90.3/1.2 are direct evidence that
these regimes can differ by tens of points or by everything; COLOSSEUM's ≥75% vs 30–50% makes the
same point in manipulation. **No cited benchmark reports a single aggregate over heterogeneous
holdout types**, and the one that comes closest (Transporter Nets' unified Table 2) is the one whose
seen/unseen result is hardest to read.

**5.2 — The two axes are not the same kind of holdout.** Under CFQ's framing the pair holdout raises
**compound** divergence with atom divergence at zero — every puck, zone and region seen. The sector
holdout raises **atom** divergence: a target-puck position atom is entirely absent from training.
CFQ's principle 1 exists specifically to keep these apart. Unioning them means the `heldout` number
confounds compositional generalisation with spatial domain adaptation — the exact confound CFQ names.

Two mitigations are real and should be stated in the design's favour. First, the holdout constrains
only the *target* puck: distractors still spawn in the wedge (the spawn annulus 0.15–0.36 overlaps
it) and the arm still moves through it, so the region is *seen*, just never as a goal-relevant
location. Atom divergence is therefore milder than gSCAN's split D, where the south-west region was
absent outright, and the catastrophic 0.00 there is probably not the number to expect. Second, `info`
already carries goal identity and puck poses, so **every heldout episode can be stratified post hoc
into the three regimes with no change to the env.** The problem is in the reporting contract, not the
sampler.

**5.3 — Unverifiable from the spec:** it gives zone radius (0.30) but not zone angles, so it could
not be checked whether any target zone falls inside the 30°–75° wedge. If one does, the sector
holdout also silently restricts approach geometry for pair-holdout tasks involving that zone,
coupling the axes more tightly than the union already does. Worth a check, not a claim.

---

## Verdict

The sandbox design survives the pass. Its most exposed commitments are the ones that came back
cleanest:

- **"No object pose is ever given to the agent"** — the strongest-supported choice in the spec
  (DMC's 895.9-vs-0.6 gap; every other benchmark hands object state over).
- **Constant `reward = 0.0`** — breaks no documented Gymnasium invariant.
- **Snapshot/restore** — MuJoCo's own first-class mechanism, not a workaround.
- **`reset()` that never resets the agent, announcing nothing** — has partial precedent in
  reset-free RL and goes *further* than anything cited. A genuine novelty claim, supported.
- **Goal-as-perception** — three legs, three published sources: dm_control for a rendered target as
  goal channel, GLean for the clamped-prediction end-state, LEXA for perceptually-specified goals
  with no environment reward.

### Revision tickets warranted

1. **`armature` leaks onto the puck joints** (2.1). A prototype bug, verified directly in
   `arena.xml`: a classless `<default><joint armature="0.01"/>` inherited by every puck joint. Puck
   rotational inertia is 34–327× too large and translational mass 5–20% too large. Fixing it moves
   both numbers the spec quotes — the 0.12–0.17 m push distance and the >2 N break-away force.
2. **The hidden-rotation claim is unsupportable** (1.1). Circular pucks with a non-colliding marker
   make θ unobservable *and* inconsequential. Either drop the claim or make one puck
   non-circular / off-centre.
3. **The snapshot field list is incomplete** (4.1). Omits `qacc_warmstart`, which MuJoCo's docs
   identify as required for reproducibility when loading a non-initial state — and here every restore
   is one. Recommend naming `mjSTATE_INTEGRATION` plus task and RNG.
4. **`split="heldout"` needs a stratified reporting contract** (5.1, 5.2). A union of a compound
   holdout and an atom holdout produces an unattributable aggregate. `info` already makes
   stratification possible without touching the sampler.

### Documentation, in the spec's existing "Known exposure" register

- Friction is anisotropic and box-shaped: 41% stronger along world diagonals, limit surface a box
  rather than an ellipsoid (2.2, 2.3). Silence is the only option worth arguing against, because the
  current line reads as isotropic Coulomb friction.
- Pushes are ballistic, not quasi-static, and the world is exactly repeatable where the phenomenon it
  models is not (1.3, 2.6).
- `check_env` fails `check_reset_seed_determinism` by construction, and `EnvSpec.nondeterministic` is
  the wrong flag to silence it with (4.3); `gymnasium.make(..., max_episode_steps=N)` reimposes
  `truncated=True` via `TimeLimit` (4.2).
- No evaluation protocol is defined (4.4).
- The held-out slice and the clamped-goal mechanism point opposite ways (3.1); 64×64 target/body
  resolution (3.3); goal hallucination as a clamping failure mode (3.4); precision balancing across
  four modalities (3.7).

### Measurements to run

- Sub-threshold creep: hold 1.9 N against puck 2 for 60 s, log displacement (2.4).
- Push distance vs. push direction, 0°→90°, before the sector split is treated as clean (2.5).

### Wording fixes

- "This is what lets a goal be expressed as a clamped prediction later" states the delta-limit variant
  as if it were the standard mechanism; a graded prior preference is (3.2).
- "There is no reward channel" conflates "no environment reward" (attested) with "no scalar objective"
  (unattested) — disagreement *is* an objective (3.5).

### Honest gaps

- Whether an agent can disambiguate a goal marker from world state when they share a channel could
  not be verified to have been studied at all. Every system found that renders a goal into the scene
  also supplies a redundant goal vector, task id, or mission string. Patchworks' sole-channel
  configuration appears unattested — **read as "no evidence either way", not as support.**
- Sub-threshold creep (2.4) is derived from documented equations, not measured: no macOS x86_64
  `mujoco` wheel was reachable.
- Not verified: Zhu et al. arXiv:2004.12570, KitchenShift, Howe & Cutkosky 1996, Goyal/Ruina/
  Papadopoulos 1991, Lee & Cutkosky 1991, Lynch/Maekawa/Tanie 1992 (the last four cited second-hand
  via Yu et al. §II). Sermanet et al. partially verified from abstracts only.
