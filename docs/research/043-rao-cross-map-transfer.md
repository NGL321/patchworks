# Citation pass: Rao's cross-map transfer — timescales or rollout? (patchworks#43)

A targeted re-read of one paper — Rao, Gklezakos & Sathish (2022), *Active Predictive Coding*,
arXiv:2210.13461v1 — covering the sections [#26](https://github.com/NGL321/patchworks/issues/26) did
not examine (§3, §4, §5.2, §6.1, Supplementary §2.2 and §2.4). §5.1, §6, §6.2's planning mechanism
and Supplementary §1.1.1 are covered in `026-action-boundary-citations.md`, cited here rather than
re-derived. Per the map's citation sequencing this validates a committed design; it does not reopen it.

## Headline verdict, stated plainly

**The recalled result is real but far smaller than [#25](https://github.com/NGL321/patchworks/issues/25)
remembered it, and the paper never separates the two candidate causes. It is unattributable.**

1. **The cross-map navigation transfer is a three-sentence supplementary note with no numbers.** It
   is not in the main paper. It is Supplementary §2.4, "Additional Results: Zero-Shot Transfer to New
   Environments" — reproduced *in full* below, because its full length is the finding. No metric, no
   baseline, no success rate, no count of environments beyond "a variety," no control. The evidence
   is four rows of a qualitative figure.
2. **Every navigation result in the paper, that one included, runs the random-sampling shooting MPC
   loop.** There is no navigation result anywhere in the paper with the rollout removed, so the
   transfer cannot be attributed to the timescale separation by the design of the experiment.
3. **There *is* a two-level-versus-one-level ablation in a no-planning setting (§5.2, Table 1) — but
   it is on the vision task, not navigation, and it is not clean.** Closest the paper comes to the
   decisive experiment; close enough to state precisely rather than dismiss.

**One-sentence reasoning:** the paper demonstrates that a library of reusable macro-actions plus a
4-step rollout composes into new environments, and separately that two levels help a reconstruction
task generalize to unseen classes — but it never runs the experiment that would tell you whether the
*navigation* transfer survives without the rollout, so #25's attribution to the multi-timescale
separation is not supported by this paper and cannot be repaired by reading it more carefully.

## 1. What the transfer result actually is

### The navigation result (Supplementary §2.4) — quoted in full

> We also tested the APC model on navigation problems in a variety of other large compositional
> building environments. The same eight higher-level "macro-actions" (options) learned for the
> environment in Figure 8 (a) in the paper were used by the APC model to successfully compose
> macro-action sequences to navigate to different goals in new environments, an example of zero-shot
> transfer. Example results showing hierarchical planning by the APC model in these new large
> environments are shown in Figure 2.

Supplementary Fig. 2's caption: "(a)-(d) show hierarchical planning by the APC model using the same
set of higher-level actions as in the paper [...] Each row is a different new environment. [...]
Navigation (red dots) to two randomly chosen goals (green) are shown." Four environments, two goals
each — eight demonstrated trajectories, shown as pictures.

**What is held out:** the *building layout* only. The eight macro-actions `A1…A8` are the carried-over
artifact, trained (REINFORCE, §6.2) to reach the four corners of room types `R1` and `R2`; the new
environments are new arrangements of *the same two room types*, so nothing about the room-type
vocabulary is held out. This is compositional recombination over a fixed, small, hand-specified
component set, not generalization to unfamiliar parts. **How much transferred:** unquantified;
"successfully compose" is the whole claim.

### The vision result (§5.2) — the one with numbers

> We trained a two-level APC model to reconstruct examples from 85% of classes from each Omniglot
> alphabets. The rest of the classes were used to test transfer: the trained model had to generate
> new programs (via the state and action hypernets) to predict parts for new character classes for
> each alphabet. The model successfully performed this transfer task (Table 1 and Figure 7b).

Table 1 (recon. MSE/pixel, lower better), transfer column `Om-Trn`: `RB 0.0301`, `APC-1 0.0323`,
`APC-2 0.0226`. No variance, repeats, or error bars anywhere in the table.

## 2. Attribution: timescales, rollout, or neither?

**Unattributable. Stated plainly, per the ticket: the paper never separates them.**

**Against the timescale reading.** The navigation transfer is carried by the *reuse of a trained
option library defined in a local reference frame*, and the paper says so in its own voice.
§6.1: "Defining these policies to operate within the local reference frame of the higher-level state
R1 or R2 (regardless of global location in the building) confers the APC model with enormous
flexibility because **the same policy can be re-used at multiple locations** to solve local tasks."
The operative property is reference-frame locality — a policy that does not know where in the
building it is — not a rate ratio between levels. A schedule with `T1 = 1` would preserve
reference-frame locality entirely.

**For the rollout reading.** Composition of those reused options into a route is done by the MPC loop
and by nothing else (§6.2, quoted in full in #26's pass). Supplementary §2.2.2 confirms this runs in
the transfer setting: "`Fs` was unrolled with the random action sequence (`At, At+1, At+2, At+3`).
`N` such random trajectories were unrolled and the sequence reaching the goal was chosen." And the
planner is explicitly *not* given a heuristic: "the APC planner in this experiment had no access to
heuristics that would inform the location of goal and relied only on feedback regarding whether an
action sequence reached the goal." Search over sampled futures does the whole job of deciding *which*
macro-actions to string together in the new building.

**The one experiment that comes closest to separating them.** §5.2's ablation compares APC-2 (two
levels) against APC-1 (one level) and a randomized-sampling baseline RB on reconstruction — **a
setting where no planning loop runs at all**. On the Omniglot transfer set APC-2 (0.0226) beats both
APC-1 (0.0323) and RB (0.0301): a genuine two-levels-help-generalization result obtained with the
rollout absent, and the strongest thing in the paper for #25's reading. Three reasons it does not
carry the claim:

- **Wrong task.** Part-whole reconstruction of unseen character classes, not navigation across maps.
  Nothing licenses transporting it to the navigation claim; the paper does not transport it either.
- **APC-1 is worse than the random baseline on Omniglot** (0.0323 vs 0.0301) on both the test and the
  transfer column. The ordering is not monotone in levels; a single-point comparison with no variance
  reported cannot distinguish a levels effect from a capacity or optimization effect.
- **Levels are not timescales here anyway.** APC-2 differs from APC-1 in having a hypernetwork
  generating a second network (§3 below). The ablation varies *architecture depth*, not the
  macro/micro step ratio `T1`. The paper contains no ablation on `T1`.

**Therefore:** no experiment in this paper observes the cross-map navigation transfer with the
rollout removed, and no experiment varies the macro/micro rate ratio at all. The result is real and
unattributable — the same shape as [#23](https://github.com/NGL321/patchworks/issues/23)'s `heldout`
correction, exactly as the ticket anticipated.

## 3. Does the macro/micro relationship match ADR-0005?

**No, on both halves of ADR-0005's decision, and the paper concedes the second half itself.**

**Not persistence — an explicit schedule.** §4: "In our current implementation, the lower level RNNs
execute for a **fixed number of time steps** before returning control back to the higher level"
(§4.1.1 names the count `T1`: "after `T1` bottom-level micro-steps have finished executing"). §7, in
the paper's own words: "Current limitations of the model include **the fixed number of time steps
used at each level** and using only a two-level hierarchy." Footnote 2 names the escape hatch:
"Future implementations will explore the use of **termination functions** [...] to allow a variable
number of time steps at each level" — precisely ADR-0005's *termination-driven abstract steps*
alternative, rejected there as new mechanism and held as its named escape hatch. The same next move,
identified independently from the same dissatisfaction, and Rao has not built it either.

**Not one substrate — separate networks per level.** §4: "with each level employing a state network
and an action network," each generated by its own hypernetwork. ADR-0005's mechanism is a *frozen
shared body* whose per-cell fold offsets select a regional Jacobian, with every cell running every
tick; Rao's higher level is a different RNN that does not run on micro-steps at all. The two designs
share no mechanism, no substrate, and no definition of what makes a level slow.

**Overlap with [#29](https://github.com/NGL321/patchworks/issues/29), not duplication.** #29 owns "is
APC's timescale really a relative action count per level, with a fixed `K`". The quotes above confirm
it incidentally — `T1` is the `K` — and this pass stops there.

## Verdict

The paper does not support the sentence #25 wanted. Nor does it refute Patchworks: it fails to run
the experiment, which is a different thing. What it positively establishes is narrower and mostly
about options — a library of reference-frame-local macro-actions, learned once, recombines into new
layouts of the same room types when a 4-step random shooting search does the recombining. It is a
precedent for *reusable local policies* and for *rollout as the composer*; as a precedent for
timescale separation producing transfer, it is not evidence at all.

### The sentence `04-action-and-the-boundary.md` (*Horizon*) may claim

**Nothing.** There is no honest sentence available from this paper claiming Rao as a transfer
precedent for timescale separation. If one is wanted anyway — defensible in the *Known exposure*
register, never as support — this is the strongest form the record permits:

> Rao's active predictive coding does demonstrate zero-shot transfer of learned macro-actions to new
> compositional environments (arXiv:2210.13461, Supplementary §2.4), but it demonstrates this
> qualitatively, over recombinations of the same two hand-specified room types, and only with its
> random-sampling shooting MPC loop running — the paper never separates the contribution of its
> timescale hierarchy from the contribution of the rollout, so it is not a precedent for getting
> transfer without one.

**The recommendation is still to claim nothing** — the spec does not currently lean on the precedent
and gains nothing by naming one it must immediately disown.

### Revision tickets warranted

**One, against [#25](https://github.com/NGL321/patchworks/issues/25) — low severity: correction of a
recalled premise, not of a decision.** #25's reasoning cites a Rao result the paper does not contain
at the strength recalled — supplementary, unquantified, unattributed between timescales and rollout.
#25's *decision* (two routes, no plan — the world selects) does not depend on the precedent, and #26
already recorded Rao as a precedent for the thing Patchworks declines. Correct the citation in #25's
rationale; do not reopen route selection. No spec text changes, since the spec never took the claim up.

**None against ADR-0005 or `05-timescales.md`.** The macro/micro mismatch in §3 is confirmatory of
what #29 is already chartered to examine, and ADR-0005 never claimed Rao as precedent.

### Honest gaps

- **The *Neural Computation* 36(1) (2024) version could not be read.** MIT Press returned HTTP 403 to
  both `curl` and WebFetch; PubMed returned a cookie wall; the Scholar Gateway full-text corpus
  returned twelve results for a targeted query, none of them this paper. **This is the one material
  gap**, because the journal version is retitled "…Active Perception, **Compositional Learning**, and
  Hierarchical Planning" — the compositional claim was promoted into the title between versions,
  exactly the kind of change that accompanies added experiments, and arXiv has only v1 (single
  submission, 23 Oct 2022) so it cannot reveal what changed. **Every finding here is a finding about
  arXiv v1.** A journal-version quantified compositional experiment or `T1` ablation could change
  findings 1 and 2; finding 3 (the schedule) is architectural and could not.
- No further sources consulted, per the ticket's budget — in particular no options/HRL literature to
  contextualize the macro-action reuse claim; that is #29's and #7's territory.

## Sources

- Rao, R.P.N., Gklezakos, D.C., Sathish, V. (2022). Active Predictive Coding: A Unified Neural
  Framework for Learning Hierarchical World Models for Perception and Planning. arXiv:2210.13461v1.
  https://arxiv.org/abs/2210.13461 — read from the PDF (§3, §4, §4.1.1, §5.2, §6.1, §7, Supp. §2.2,
  §2.2.1, §2.2.2, §2.4).
- Same authors (2024). *Neural Computation* 36(1), 1–43. https://doi.org/10.1162/neco_a_01627 —
  **not obtained; see Honest gaps.**
- `docs/research/026-action-boundary-citations.md` — prior pass on §5.1, §6, §6.2, Supp. §1.1.1.
