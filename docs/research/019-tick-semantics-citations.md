# Citation pass: tick semantics (patchworks#19)

Validates the design closed in patchworks#4 (`docs/spec/02-tick-semantics.md`,
`docs/spec/01-cell-and-sheaf.md`, `docs/adr/0002-message-passing-is-one-step-not-a-solve.md`).
Citations validate after the fact per the map's Notes; this does not revise the closed design.
Vocabulary follows `CONTEXT.md`: Patchworks' side of every comparison is described in its own
terms (chart, node stalk, edge stalk, disagreement, reconciliation, restriction map); the prior
art's side is described in its own field's terms.

## Predictive coding

**Source:** Rao, R.P.N. & Ballard, D.H. (1999). "Predictive coding in the visual cortex: a
functional interpretation of some extra-classical receptive-field effects." *Nature
Neuroscience* 2(1), 79–87.

Read directly from the paper (Methods): the model is a hierarchy of levels, each holding a
vector of "causes" `r` and a basis matrix `U` such that a lower level's signal is generated as
`I = f(Ur) + n` (their Eq. 1). Higher levels predict a lower level's `r` via `rtd = f(Uh·rh)`
(their Eq. 3). **Feedback connections carry the prediction; feedforward connections carry the
residual error `(I − f(Ur))` and the between-level error `(r − rtd)` as their own dedicated
neural populations** — "error-detecting neurons" are a named, first-class part of the
architecture (Fig. 1b). Estimating `r` for a given input is not a single-shot computation: it is
literal continuous-time gradient descent on the combined-error objective `E`,
`dr/dt = k₁·[Uᵀ(I − f(Ur))/σ² + (rtd − r)/σtd² − g′(r)]` (their Eq. 7), which the paper says
"converges" to a stable estimate before that estimate is used. Dimensionally, `r` is a real
compression relative to what it predicts in their simulations (32 level-1 causes over a 256-pixel
patch; 128 level-2 causes over three level-1 modules' worth of output).

**Correspondences:**
- Rao & Ballard's causes vector `r` and Patchworks' **chart** play the same structural role: a
  compressed, private coordinate system (`k < n` in Patchworks' terms) that a linear-plus-static-
  nonlinearity map turns into a prediction of the lower/adjacent representation (`Ur`/decode).
  `k=32 < n=256` in their level-1 simulation is a direct numerical instance of Patchworks' fixed
  `k < n` invariant, arrived at independently.
- Neither paper's "recognition" step is a separately-trained network: Rao & Ballard invert their
  one generative model (`U`) by gradient descent on the same objective that trains it, rather than
  learning a distinct encoder. Patchworks similarly has one algorithm per cell (`encode`/`step`/
  `decode`), not an opposed pair of a generative model and a separately-optimized recognition
  model — both designs collapse "prediction" and "inference" into one machine rather than two.

**Divergence, precisely stated:** Rao & Ballard's error is a **transported, first-class signal**
— it has its own neurons, its own connections, and is explicitly the thing carried feedforward.
Patchworks' **disagreement is the opposite of a transported error channel by design**: the edge
stalk "carries no error channel; error is derived, never transported" (`01-cell-and-sheaf.md`),
and disagreement is computed locally at reconciliation time from two independently-restricted
node-stalk values, never itself broadcast onward as a signal in its own right. This is a precise,
named point of divergence from the founding predictive-coding paper on exactly the question of
whether error is a channel — not an oversight, since CONTEXT.md explicitly instructs avoiding
"error, residual, loss, surprise" as names for disagreement. Rao & Ballard's within-presentation
convergence (their `dr/dt` settles to a fixed point before the estimate is used) is also precisely
the "iterate-to-convergence" pattern ADR-0002 names and rejects for the message-passing phase —
though note it settles their **chart-analogue**, across what Patchworks would call multiple
sub-tick relaxation steps, which the tick-semantics design instead spreads across real ticks
(cross-tick settling via the persisted chart) rather than settling within one.

**Related:** Bogacz, R. (2017). "A tutorial on the free-energy framework for modelling
perception and learning." *Journal of Mathematical Psychology* 76, 198–211. Confirms the same
shape at the level Friston's free-energy framework generalizes it to: perception is inversion of
one generative model by gradient flow (`φ̇ = ∂F/∂φ`, their Eq. 9) with dedicated prediction-error
neural populations `εp`, `εu` (Eq. 11–13) that are explicitly propagated between levels — the same
transported-error commitment as Rao & Ballard, and the same divergence point from Patchworks'
derived-never-transported disagreement. Friston's wider active-inference framework unifies
perception and *action* under one free-energy objective (a single scalar minimized by both
updating beliefs and acting on the world) rather than running two opposed processes; this is a
looser but real correspondence to Patchworks running one uniform cell algorithm rather than
splitting prediction and correction into separate opposed systems.

## Kalman-filter predict/update

**Source:** Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems."
*ASME Journal of Basic Engineering* 82(1), 35–45.

The classical discrete Kalman recursion, as codified in every standard treatment of the 1960
paper, runs each time step in two named stages, in this fixed order: **(1) time update
("predict")** — project the prior state and covariance forward through the state-transition model
to get the *a priori* estimate, `x̂(k|k−1) = A·x̂(k−1|k−1)`; **(2) measurement update ("correct")**
— incorporate the new observation to get the *a posteriori* estimate, `x̂(k|k) = x̂(k|k−1) +
K·(z(k) − H·x̂(k|k−1))`. Predict always precedes correct within a cycle.

**Investigated, not assumed, per the ticket's instruction — here is exactly how Patchworks'
order aligns and inverts:**

- **Within one tick, Patchworks' order is inverted relative to Kalman's per-cycle order.**
  `encode` fuses the persisted chart with the node stalk left behind by the previous
  message-passing phase — this is the *correction* operation (it incorporates new evidence into
  the internal estimate) — and it runs **before** `step`, which is the *prediction* operation
  (it advances the fused chart one tick). So, read as one cycle, Patchworks does correct-then-
  predict where Kalman does predict-then-correct.
- **Across the tick boundary, the two orders converge rather than stay inverted.** What actually
  gets corrected in a Kalman cycle is the *state estimate itself*, in the same step the
  observation arrives. Patchworks explicitly refuses to do that: reconciliation "edits the node
  stalk only... never reaches into the chart" (`01-cell-and-sheaf.md`) — the correction is
  computed as disagreement during message-passing at tick `t`, but is not incorporated into the
  chart (the actual recurrent state) until `encode` at tick `t+1`. So if the unit of comparison is
  "when is the internal state actually corrected relative to when a prediction is made and
  compared against incoming evidence," the sequence is: predict & expose (this tick's `step` +
  `decode` + restriction), compare against a delayed neighbour value and derive disagreement
  (this tick's reconciliation — structurally Kalman's innovation computation, `z − H·x̂`), then
  correct the state (next tick's `encode`) — which **is** predict-then-correct, just with the
  correct half deliberately deferred by exactly one tick rather than folded into the same step.
  This is a genuine and load-bearing design choice, not an approximation of Kalman: Patchworks
  never lets an external correction reach into the chart directly (`01-cell-and-sheaf.md`, "the
  correction re-enters inference on the next tick as ordinary evidence... not as an imposed
  correction to internal state"), where a Kalman filter's measurement update is precisely such a
  direct, same-step correction to the state estimate.
- Net assessment: **encode-before-step is an inversion of Kalman's textbook per-step order when
  compared naively; it is a one-tick-delayed reproduction of Kalman's predict → correct order when
  compared at the level of "when does a correction actually reach the recurrent state."** Both
  readings are correct depending on which unit of time is treated as one cycle — this is the
  precise shape of the correspondence, not a simple match or a simple inversion.

## Synchronous / Jacobi-style message passing

**Sources:**
- Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., & Dahl, G.E. (2017). "Neural Message
  Passing for Quantum Chemistry." *ICML 2017* (PMLR 70), arXiv:1704.01212.
- Scarselli, F., Gori, M., Tsoi, A.C., Hagenbuchner, M., & Monfardini, G. (2009). "The Graph
  Neural Network Model." *IEEE Transactions on Neural Networks* 20(1), 61–80.

Gilmer et al.'s MPNN formalism runs message passing for **`T` time steps**, all nodes updating in
parallel each step from the previous step's values — `m_v^{t+1} = Σ_{w∈N(v)} M_t(h_v^t, h_w^t,
e_vw)`, `h_v^{t+1} = U_t(h_v^t, m_v^{t+1})` (their Eqs. 1–2) — synchronous exactly in the Jacobi
sense (every node reads only last-round values; there is no visiting order). Critically, `T` is
**a tuned hyperparameter with no principled derivation**: their own hyperparameter search
constrained it to `3 ≤ T ≤ 8`, chosen empirically per dataset. Scarselli et al.'s earlier GNN
model goes further and treats the per-node state-update function as a **contraction map**, so
that iterating it is formally guaranteed (via the Banach fixed-point theorem) to converge to a
unique equilibrium — i.e., they iterate to convergence rather than to a fixed round count, using a
convergence check as the stopping rule.

**Correspondence and named divergence:** These two papers are, between them, exactly the two
alternatives ADR-0002 names and rejects for Patchworks' message-passing phase: Gilmer et al. is
the "fixed round count `R`, chosen as an unmotivated constant" branch (their own `3–8` sweep is
direct evidence that `T` has no principled value — it is tuned empirically, precisely the failure
mode ADR-0002 calls out), and Scarselli et al. is the "iterate to convergence" branch, which
ADR-0002 rejects because any legitimate stopping rule requires reading total disagreement across
the graph — an all-to-all aggregate that graph-locality rules out. Patchworks' reconciliation
being exactly one synchronous Jacobi-style step per tick is the same update rule as either paper's
single round, just never repeated within a tick; **the settling those papers get from many rounds
inside one forward pass, Patchworks gets instead from many ticks, each contributing one round,
with the persisted chart and one-tick edge delay standing in for the "more rounds" these papers
use.**

On the Jacobi/Gauss–Seidel distinction itself: classical iterative-solver literature (e.g.
comparisons of Jacobi and Gauss–Seidel parallel iteration schemes) defines Jacobi as every
unknown updating from the *previous* full iterate (synchronous, no intra-round dependency) and
Gauss–Seidel as each unknown updating from whatever values are *already newest* within the same
round (sequential, visiting-order-dependent). Patchworks' message-passing phase is precisely
Jacobi in this sense — "all cells read the same prior round's incoming values and update at once…
there is no visiting order to define" (`02-tick-semantics.md`) — and this is a correct, literal
application of the classical term, not a loose analogy.

## Cellular sheaf Laplacians

**Sources:**
- Hansen, J. & Ghrist, R. (2019). "Toward a Spectral Theory of Cellular Sheaves." *Journal of
  Applied and Computational Topology* 3, 315–358. arXiv:1808.01513.
- Hansen, J. & Ghrist, R. (2021). "Opinion Dynamics on Discourse Sheaves." *SIAM Journal on
  Applied Mathematics* 81(5), 2033–2060. arXiv:2005.12798.
- Hansen, J. & Gebhart, T. (2020). "Sheaf Neural Networks." arXiv:2012.06333.
- Bodnar, C., Di Giovanni, F., Chamberlain, B.P., Liò, P., & Bronstein, M.M. (2022). "Neural
  Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs."
  *NeurIPS 2022*. arXiv:2202.04579. (Note: the paper's actual subtitle is
  **"...Oversmoothing..."**, not "oversquashing" as informally referenced in the ticket brief —
  corrected here against the primary source.)

Hansen & Ghrist define the degree-0 sheaf Laplacian blockwise directly from restriction maps:
diagonal blocks `Δ⁰_{v,v} = Σ_{v⊴e} F*_{v⊴e}F_{v⊴e}`, off-diagonal blocks `Δ⁰_{u,v} =
−F*_{u⊴e}F_{v⊴e}` for each edge `e` between `u` and `v`. Bodnar et al.'s Definition 14 gives the
resulting quadratic form explicitly as a **sum over edges of squared norms of restriction
differences**: `E_F(x) = xᵀΔ_F x = ½ Σ_{e=(v,u)} ‖F_{v⊴e}x_v − F_{u⊴e}x_u‖²` — this sum is what
"Dirichlet energy" names. Hansen & Ghrist's Opinion Dynamics paper explicitly calls this quantity
"discord" ("the sheaf Laplacian `L_F` … registers the discord in the system") and shows the
harmonic space `ker L_F = H⁰(G; F)` — the zero-discord, fully-consistent subspace — is exactly
what a continuous-time diffusion `dx/dt = −α·L_F·x` (their Eq. 13) converges toward. Bodnar et
al.'s Neural Sheaf Diffusion discretizes that same diffusion as a residual update, `X_{t+1} = X_t
− σ(Δ_F(t)(I⊗W₁ᵗ)X_t·W₂ᵗ)` (their Eq. 6), run for `T` layers — again a **tuned hyperparameter**,
not a convergence-derived count (their Figure 5 sweeps performance "as a function of diffusion
time").

**Correspondences, stated precisely:**
- Patchworks' restriction maps (linear, one per cell per incident edge, independent at the two
  ends) are the same object Hansen & Ghrist's `F_{v⊴e}` and Bodnar et al.'s `F_{v⊴e}` name — this
  is a direct 1:1 match, not an analogy, down to the requirement that they be linear (Bodnar et
  al.'s central innovation is *learning* these maps rather than hand-specifying them, whereas
  Patchworks' spec masks them by a fixed structural mask that "closes and never reopens" — the
  learned/masked split is Patchworks' own choice, not a divergence from the sheaf-theoretic
  object itself).
- **Disagreement, summed in squared form over every edge, is exactly the Dirichlet energy** —
  Patchworks' per-edge disagreement (a vector: the difference of the two endpoints' restricted
  node-stalk values) is precisely the term `F_{v⊴e}x_v − F_{u⊴e}x_u` inside the sum above. This
  correspondence is exact and well-founded — see Flagged inconsistencies below for the one
  place its phrasing overstates it.
- Hansen & Ghrist's continuous-time diffusion toward the harmonic (zero-discord) space is the
  same direction of travel as Patchworks' reconciliation, which is "penalised, not enforced" —
  neither design projects onto the consistent subspace in one shot; both move toward it by a
  bounded step and let disagreement persist as signal. Patchworks' further restriction to
  *exactly one* step per tick (never accumulated into a many-layer diffusion block the way Bodnar
  et al.'s `T`-layer stack does) is the same locality-driven choice already covered under
  Synchronous/Jacobi message passing above — the sheaf-diffusion literature treats `T` as a tuned
  hyperparameter exactly as the MPNN literature does, and Patchworks rejects that move here for
  the identical reason (ADR-0002).

## Flagged inconsistencies

One genuine, precisely-locatable imprecision was found — not a design flaw, but a wording gap
between the spec's claim and the cited mathematical object:

- **`01-cell-and-sheaf.md` states disagreement "is the sheaf's Dirichlet energy."** Per Hansen &
  Ghrist and Bodnar et al., the Dirichlet energy `E_F(x) = ½ Σ_e ‖F_{v⊴e}x_v − F_{u⊴e}x_u‖²` is a
  **single global scalar**: a sum over every edge in the graph. Patchworks' disagreement, as
  defined in the same document and in CONTEXT.md, is a **local, per-edge, vector-valued**
  quantity — "the difference, in an edge stalk, between the two endpoint cells' restrictions of
  their node stalks." One cell's disagreement on one edge is therefore not the Dirichlet energy;
  it is **one squared-norm term inside the sum that defines the Dirichlet energy**. Summed over
  every edge in the graph (a computation no single cell ever performs, and which graph-locality
  explicitly forbids any cell from performing), the disagreements would recover the Dirichlet
  energy exactly. The claim is directionally correct and the underlying architecture is
  consistent with the sheaf-theoretic object — cell-local disagreement genuinely is a real,
  well-defined summand of Dirichlet energy — but the sentence's "is," read literally, equates a
  local part with the global whole. A more precise phrasing consistent with the rest of the
  document (which is careful everywhere else to keep cell-local and graph-local distinct, per
  CONTEXT.md's own two separate glossary entries for those terms) would be: *disagreement on one
  edge is the local summand of the sheaf's Dirichlet energy* — same relationship the document
  clearly intends, one word away from being exact.

No other contradiction was found. The remaining divergences documented above (Rao & Ballard's
transported error channel vs. Patchworks' derived-never-transported disagreement; Kalman's
same-step correction vs. Patchworks' one-tick-deferred correction; Gilmer/Scarselli/Bodnar's
tuned-or-converged round counts vs. Patchworks' fixed one-step rule) are deliberate,
already-documented design choices (ADR-0002, CONTEXT.md's "avoid" lists), not contradictions of
anything Patchworks' own spec claims about itself.

## Sources

- Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex: a functional
  interpretation of some extra-classical receptive-field effects. *Nature Neuroscience* 2(1),
  79–87. https://www.nature.com/articles/nn0199_79
- Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and
  learning. *Journal of Mathematical Psychology* 76, 198–211.
  https://doi.org/10.1016/j.jmp.2015.11.003
- Kalman, R.E. (1960). A new approach to linear filtering and prediction problems. *ASME Journal
  of Basic Engineering* 82(1), 35–45.
- Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., & Dahl, G.E. (2017). Neural message
  passing for quantum chemistry. *Proceedings of the 34th International Conference on Machine
  Learning* (PMLR 70), 1263–1272. arXiv:1704.01212.
- Scarselli, F., Gori, M., Tsoi, A.C., Hagenbuchner, M., & Monfardini, G. (2009). The graph
  neural network model. *IEEE Transactions on Neural Networks* 20(1), 61–80.
- Hansen, J. & Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. *Journal of
  Applied and Computational Topology* 3, 315–358. arXiv:1808.01513.
- Hansen, J. & Ghrist, R. (2021). Opinion dynamics on discourse sheaves. *SIAM Journal on Applied
  Mathematics* 81(5), 2033–2060. arXiv:2005.12798.
- Hansen, J. & Gebhart, T. (2020). Sheaf neural networks. arXiv:2012.06333.
- Bodnar, C., Di Giovanni, F., Chamberlain, B.P., Liò, P., & Bronstein, M.M. (2022). Neural sheaf
  diffusion: a topological perspective on heterophily and oversmoothing in GNNs. *Advances in
  Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2202.04579.
