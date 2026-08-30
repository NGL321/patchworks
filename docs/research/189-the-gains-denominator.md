# The gain's denominator: the ledger

[#189](https://github.com/NGL321/patchworks/issues/189). An audit of every quantity in

```
gain_v  =  γ / max( Σ_{e∋v} m_e , ρ² · deg(v) )
```

against the record that put it there and everything measured since. **Decision-free by
construction**: the ruling is [#190](https://github.com/NGL321/patchworks/issues/190)'s, and where
this document has a view it says whose call it is rather than making it.

Readings of its own — the construction-time ones, and the initialisation `λ_max` — come from

    docker run --rm -v "$PWD:/work" -w /work --entrypoint python patchworks:189 \
        prototypes/gain-denominator-189/audit.py     # 8 s wall

in the supported container ([ADR-0012](../adr/0012-a-container-is-the-supported-execution-target.md)),
on the real dome: 414 cells (150 predicting, 264 boundary), 682 edges, seed 0. Trained readings are
[#182](https://github.com/NGL321/patchworks/issues/182)'s and
[#178](https://github.com/NGL321/patchworks/issues/178)'s, cited rather than re-run.

---

## The verdict, in five lines

1. **The fourth one is at the boundary.** `Σ_e m_e` and `ρ²·deg(v)` are both written for interior
   maps in the gauge band, and both are applied unchanged to the 264 **boundary** cells, whose maps
   carry the *exact* gauge `‖F‖_F = 1`. The bound that holds there is `deg(v)`; the denominator
   applied is `8·deg(v)`. That looseness is **permanent** — an exact gauge is an equality, not a
   band training tightens into — and the record's tables have never contained a boundary cell.
2. **It lands on the one cell the demo reads.** Of the 264, exactly one has a reconciliation step
   that survives the tick: the actuator's *commanded* components, which `Agent.write` deliberately
   does not write. The return path's last step into the arm is taken at **1/24** where the provable
   bound permits **1/3**.
3. **The composition is an artifact of an additive edit, and the record names the moment.**
   `Σ_e m_e` was the *whole* denominator from `b3a60fe`; `ρ²·deg(v)` and the `max` arrived a day
   later with ADR-0010 (`c382882`, #37), which added a provable bound *beside* the old proxy instead
   of replacing it — under a sentence, *"so the gain is in practice what it always was"*, that made
   the change look inert and stopped anyone re-reading the argument it superseded.
4. **The equaliser does not equalise the quantity it claims to.** `02` defends `Σ_e m_e` as giving
   "roughly the same descent on its own local energy" per cell. Measured as `gain_v · λ_max`, the
   descent in units of the largest stable one, the spread across the taper is **3.57x at
   initialisation and ~2.6x taught** — and it grades with depth, largest at the apex.
5. **The dimensional pass finds one defect and one conservatism.** #181's `hop ≥ floor` is the
   defect and it is the only one: the fold-margin comparison is well-formed. But it is measured in
   `encode`'s whole input space while reconciliation displaces only the node-stalk block, making it
   **~1.176x tighter than it needs to be** — and #178 reads a permitted factor off exactly that
   tightness.

---

## 1. The entries

Five quantities, one composition rule. Each entry is *claim / provenance / job / check / couplings*,
as the ticket asked.

### 1.1 `Σ_{e∋v} m_e` — the sum of incident mask widths

**The claim.** [`02-tick-semantics.md:101`](../spec/02-tick-semantics.md): *"`Σ_e m_e` tracks the
largest eigenvalue of the cell's local Laplacian block, so this normalisation **equalises** the
effective step across the graph: every cell takes roughly the same descent on its own local energy
regardless of how many edges it sits on. It removes a degree artifact; it is not a timescale knob and
must not become one."* Restated verbatim in `tick.py:230`.

**The provenance.** `b3a60fe` (2026-08-22), the original spec batch, where it was the entire
denominator: `gain_v = γ / Σ_{e∋v} m_e`. The "tracks the largest eigenvalue" clause is asserted
there with **no derivation**, and none has been supplied since. It predates every transmission
measurement in the record.

**The job it does.** **Two**, and this is the ticket's §3 case. (a) A stand-in for `λ_max` — the job
#142 and #155 argue about. (b) A degree normaliser — defended in its own right, against ADR-0005.
Only (b) has an argument behind it in the text.

**The check.** As (a) it **fails**: `02` derives the provable bound as `ρ²·deg(v)` and never shows
`Σ_e m_e` bounds `λ_max(Σ_e F_evᵀF_ev)` at all (#182 §5). As (b) it **does not do what it says** —
see §3 below. What it *is*, unambiguously, is the binder: it takes the `max` at 70 rim cells outright
and ties at 72 more (§2).

**The couplings.** `m` is set at `graph.py:132` and is the same number the rim's width buys rank
with; #142 priced widening at `m^-3/2` and #182 found the sign — the width that lets the world in is
the number that throttles the step carrying it inward. It is also `H⁰`'s divisor:
`private = max(0, n − Σ_e m_e)`, so anything that reasons about this quantity is reasoning about
private dimension too (#150: `interior_m = 8` leaves the graph none).

### 1.2 `ρ² · deg(v)` — the gauge bound

**The claim.** [`02:94`](../spec/02-tick-semantics.md): *"The denominator bounds the largest
eigenvalue of the cell's local Laplacian block **provably**, not by proxy"*, via
`λ_max(Σ_e F_evᵀF_ev) ≤ Σ_e ‖F_ev‖_F² ≤ ρ²·deg(v)`.

**The provenance.** `c382882` (2026-08-23), ADR-0010 / #37. **Note the subject of that sentence.**
The property belongs to `ρ²·deg(v)`; the sentence attributes it to *"the denominator"*, which by then
was the whole `max`. That is the textual moment the two jobs fused, and it is one clause long.

**The job it does.** The stability bound proper, and the only argued one.

**The check.** True, and **loose**. #142: `bound / λ_max` is 41.8 untrained and **5.585** taught over
the 150 predicting cells; this audit reads 41.68 untrained at seed 0, reproducing it. The inequality
is tight only if a cell's incident maps load the same input direction, and they do not. #182 split
the surviving slack into width × rank × spread and found **rank spent** (1.02–1.06) and **spread**
holding 2.08–2.69x of headroom, largest at the rim.

**The couplings.** `ρ` is ADR-0010's band edge and appears in the numerator through both map norms;
#150 showed the two `ρ²` **cancel identically**, so `ρ = 16` buys 1.008x. Raising `ρ` cannot buy
transmission and is not a lever on this entry.

### 1.3 `deg(v)` — the incident degree

**The claim.** Implicit: that charging per *edge* is the right unit for the bound.

**The provenance.** ADR-0010's consequence bullet, same edit.

**The job it does.** Counts the terms in `Σ_e ‖F_ev‖_F²`.

**The check.** Correct as arithmetic. What it costs is what #142 named: the bound charges for
degree at full price and gives back nothing for the maps being mutually incoherent, which is where
the 5.585x lives. #142 **retired attention over exactly this** — the `deg` the bound refuses to
discount is bought by reading `λ_max` instead, and no per-edge gain or attention mechanism is needed.
Settled; not reopened here.

**The couplings.** Degree runs 8.41 at the rim to 5.00 at the apex (#182), so any change here is
graded by depth whether or not it intends to be — which is #190's decision 2 and ADR-0005's
boundary.

### 1.4 `γ` — the global scalar

**The claim.** `02:114`: *"Exactly two things"* constrain it — the global cap of 1.0, and the
provable bound. The fold margin is **demoted, not deleted** (#140).

**The provenance.** `tick.py:71`, `DEFAULT_GAMMA = 1.0`.

**The job it does.** The one global knob; per-cell variation is entirely structural.

**The check.** It is at its ceiling, and **nothing is pushing it down**. #178 read the floor to
100,000 ticks: the mid-depth tail that capped `γ` at 0.085 at 30k is a **30k artifact** — by 75k the
worst cell in the graph is 0.263 and `γ_cap` is **1.0**. So `γ` is under no pressure and buys
nothing, which is #120's finding that "any hope that removing the fold-margin bound frees it is worth
zero", now with the trajectory behind it.

**One unmeasured claim, and it is a live defect.** `tick.py`'s own comment says `γ` sits at 1.0
*"until [#85's] check exists"* — **#85 is closed, `FoldMarginCheck.gamma_cap` exists, and nothing
wires it.** That is [#180](https://github.com/NGL321/patchworks/issues/180)'s subject and this audit
confirms it stands: the value is right for a reason the code cannot state.

**The couplings.** `γ × floor` against the fold margin (§4), and the transient breach #159 recorded
— at 2,000 ticks the permitted factor is **0.90x**, so the bound is already breached at `γ = 1.0`
with no raise at all. That is #160's, not #190's.

### 1.5 The fold-margin `floor`

**The claim.** `02` binds `γ × floor` below each cell's distance to its nearest activation boundary;
`bias_selection.FoldMarginCheck` divides `product_cap` by it.

**The provenance.** ADR-0007, amended by #37 and demoted by #140.

**The job it does.** A construction-time diagnostic carrying ADR-0005's falsification duty. **Not a
bound on `γ`** since #140.

**The check.** #158 measured it and found the name wrong: the quantity is the **per-tick
reconciliation step magnitude at the running operating point**, dominated at construction by model
error — not ADR-0007's static floor, whose direct reading is a Dirichlet energy of 30.79 against
32,773 and whose gradient at its own minimiser is zero. It moves **144x** over a run and, per #178,
not monotonically: a 3.8x scatter with no trend before it settles at 0.087. `02` treats it as a
construction-time constant. **It is not one**, and that is #160.

**The couplings.** `gain_v` multiplies it, so every entry above is in this comparison; and #181's
`hop ≥ floor` is not well-formed, which is §4.

---

## 2. What actually binds, and the two readings that looked like a contradiction

Construction-time, exact, no training involved:

| population | cells | `Σ_e m_e` binds | exact tie | `ρ²·deg(v)` binds |
|---|---|---|---|---|
| predicting | 150 | 70 | **72** | 8 |
| boundary | 264 | 263 | 0 | 1 |
| all | 414 | 333 | 72 | 9 |

Per level, predicting cells — reproducing #182's table exactly:

| level | cells | deg | `Σ_e m_e` | `ρ²·deg(v)` | binds |
|---|---|---|---|---|---|
| 1 | 70 | 8.41 | 48.8 | 33.7 | `Σ_e m_e` |
| 2 | 20 | 7.50 | 30.0 | 30.0 | **tie** |
| 3–6 | 52 | 6.00 | 24.0 | 24.0 | **tie** |
| 7 | 8 | 5.00 | 17.0 | 20.0 | `ρ²·deg(v)` |

**#150 and #182 do not disagree.** #150 reported *"80 of 150 receiving cells are already on the
`ρ²·deg` arm"*; #182 reported *"it binds at 142 of 150 cells"* on `Σ_e m_e`. Both are right and
neither states its tie-break: `70 + 72 = 142` and `72 + 8 = 80`. **The 72 exact ties are the whole
of the difference**, and naming the tie column is what makes the two readings one reading. Anyone
carrying 142 or 80 forward into #190 should carry `70 / 72 / 8` instead.

The two exceptions are the two places the edge dimension is not 4 — `m = 8` on boundary-incident
edges and `m = 1` on the drive edge (`graph.py:132`) — which is exactly what makes the `max` a no-op
in the interior and live only at the two rims.

---

## 3. The composition: what the `max` guards, and what the normalisation normalises

**The guard.** `02:97` writes the `max` *"so that a later change to `ρ` cannot silently loosen the
bound below"*. Against a change to `ρ` **upward** it is a no-op — `ρ²·deg` grows and takes the max
on its own. What it guards against is `ρ` moving **down**, where `Σ_e m_e` holds the floor. Since
#150 closed raising `ρ` (self-cancelling) and nothing proposes lowering it, the guard protects
against a move nobody has a reason to make, at the price of §1.1's second job riding along with it.
That is #190's decision 3 and the ledger takes no view.

**The normalisation, checked.** `02`'s defence is that every cell takes "roughly the same descent on
its own local energy". The descent on a cell's own local energy, in the only units that make the
phrase mean something — the step as a fraction of the largest stable one — is `gain_v · λ_max`:

| level | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| `gain · λ_max`, init | 0.0177 | 0.0309 | 0.0380 | 0.0355 | 0.0379 | 0.0350 | **0.0630** |
| vs level 1 | 1.00x | 1.75x | 2.16x | 2.01x | 2.14x | 1.98x | **3.57x** |

**3.57x across the taper at initialisation**, and on #182's taught ratios (6.11 at the rim, 2.35 at
the apex) **~2.6x** taught. The raw `gain_v` spread is 2.41x, so equalising `Σ_e m_e` does not
remove the gradient in the quantity the sentence is about — it **inverts and shrinks** it. And the
residue grades with depth, largest at the apex: a step graded by depth, which is what the same
paragraph invokes ADR-0005 to forbid.

This is a check, not a ruling. It says the trade #190 faces is not *equalisation versus
transmission* but *this partial, depth-graded equalisation versus transmission* — which is a
different and cheaper thing to give up. #190 decides.

---

## 4. The dimensional pass

Every comparison the denominator participates in, checked for #181's defect rather than waiting to
trip over it:

| comparison | where | units | verdict |
|---|---|---|---|
| `gain_v ≤ 1/λ_max` | `02:94`, ADR-0010 | both `1/‖F‖²` | **well-formed** |
| `γ × floor < margin_v` | `02:122`, `bias_selection.py:900` | both node-stalk distances | **well-formed, but see below** |
| `hop ≥ floor` | #142 §5, #155 falsification 1 | dimensionless gain vs magnitude | **ill-formed — #181** |
| `Σ_e m_e` vs `ρ²·deg(v)` | `02:91` | mask widths vs `‖F‖²` | **commensurable only by coincidence** — see below |

**The fold margin is well-formed and needlessly tight.** `_fold_margin` reads
`min_i |z_i| / ‖∇z_i‖` over `encode`'s **whole** input, `R^k × R^n` — chart and node stalk together
— while reconciliation displaces the node stalk alone (`tick.py` edits `x_v`, never the chart). A
distance measured over more directions than the displacement can use is an under-estimate, so the
check is **conservative, never unsafe**. Its size, measured on the built body:

```
‖row‖ over all 44 inputs / ‖row‖ over the 32 stalk inputs
mean 1.1759, median 1.1585, max 1.3793      (isotropic expectation √(44/32) = 1.1726)
```

**~1.176x**, and it matters because #178's permitted factor of 3.79x is read straight off this
check's cap. Whether to spend it is #190's or #160's; that it exists is this ledger's.

**The `max`'s two arguments are not the same kind of number.** `Σ_e m_e` counts mask *widths*
(integers, dimensionless); `ρ²·deg(v)` is a bound on a sum of squared Frobenius norms. They are
comparable at all only because the gauge is fixed at `ρ = 2` and the interior `m = 4` makes them
numerically equal — which `02` states as a convenience and #182 found to be the whole story. This is
not ill-formed the way `hop ≥ floor` is, but it is a coincidence rather than a commensuration, and
any candidate replacing one arm inherits the question of what the other is now being compared to.

---

## 5. The fourth quantity: the boundary cells

**This is what the ticket was opened to find, and it was found by looking.**

Every table in the record — #142's, #158's, #178's, #182's — is over the **150 predicting cells**.
The graph has 414. The gain is computed for all of them (`reconciliation_gain` reads `dome.degrees`
and `dome.stalk_sums`, which are indexed by cell id) and applied to all of them
(`message_passing_phase` scatters into the whole stalk buffer). `tick.py:238` says so and prices it
in one clause:

> *"Boundary cells are included on the same formula. Their maps carry the tighter exact gauge,
> `Σ_e ‖F‖_F² = deg(v)`, so `ρ² · deg(v)` is a valid bound for them too and merely a looser one."*

**"Merely a looser one" is a factor of 8**, and it is not the `ρ² = 4` that clause implies, because
at every boundary cell `Σ_e m_e = 8·deg(v)` takes the max instead:

| kind | cells | deg | `Σ_e m_e` | `ρ²·deg` | exact bound | applied / provable |
|---|---|---|---|---|---|---|
| patch | 256 | 1 | 8 | 4 | 1 | **8.00x** |
| proprioceptive | 3 | 1 | 8 | 4 | 1 | **8.00x** |
| touch | 3 | 1 | 8 | 4 | 1 | **8.00x** |
| actuator | 1 | 3 | 24 | 12 | 3 | **8.00x** |
| drive | 1 | 8 | 8 | 32 | 8 | **4.00x** |

**And unlike the interior's, this looseness is permanent.** ADR-0010 pins a boundary cell's own maps
at the *exact* gauge `‖F‖_F = 1` (`restriction.py:23`), so `Σ_e ‖F‖_F² = deg(v)` is an **equality
that holds for the life of the run**, not a band the transport rule grows into. The interior's
41.8 → 5.585 fall (#142) is mostly gauge headroom being spent; **there is no headroom here to
spend.** The 8x is there at tick 0 and at tick 100,000.

### 5.1 It lands on the one boundary cell whose step reaches the world

Of the 264, **263 have their stalk overwritten by the external write at the end of the same tick**.
`agent.py:19` states the exception in its own words:

> *"The actuator boundary cell's three **commanded** components appear nowhere in `Agent.write` —
> nobody outside writes them — so reconciliation fills them and the world reads them."*

So the actuator is the **only** cell outside the 150 whose gain has any effect at all — and it is
the last step of the return path, the one the acceptance demo's onset-latency instrument reads:

```
stalk 6, degree 3, Σ_e m_e 24
map Frobenius norms   1.000, 1.000, 1.000     (pinned, exact gauge)
denominator applied   24.0
provable bound         3.0                    (Σ_e ‖F‖_F², an equality here)
true λ_max             0.9517                 (seed 0, at construction)
applied / provable     8.00x                  -- permanent, by construction
applied / true        25.22x                  -- at initialisation
```

**#182's conclusion is unaffected and its scope is one cell short.** It ruled there is no
*rim-specific* remedy because the rim's problem is the graph's problem, and that holds. But its
survey stopped at the 150, and the return path's final step is taken outside them, at a denominator
loose by a factor the interior's remedies do not reach: no incoherence term applies (the maps are
pinned), and `ρ²·c` does not bind here (`Σ_e m_e` does).

**What it is worth is not this ticket's to say** — an 8x at one cell is not an 8x on the hop, and
whether the demo's instrument is sensitive to it needs the return path's arithmetic, which is #190's
and #181's. What the ledger asserts is narrower and firm: **it is a fourth quantity inside `gain_v`
that is not what the record takes it for, it is at the cell the demo reads, and no ticket has
looked at it.**

### 5.2 The drive cell, and why it is the same finding in the other direction

The drive boundary cell is the one place `ρ²·deg` takes the max on the boundary: degree 8, `m = 1`
per #183, so `Σ_e m_e = 8` against `ρ²·deg = 32`. Its stalk is one-dimensional, so
`λ_max = Σ‖F‖² = 8` **exactly**, and the applied denominator is 4x that — the pure `ρ²` overcharge,
with no incoherence to recover because there is only one direction. It is written back by
`Agent.write` every tick, so nothing propagates from it; the finding is that #183's *"the transport
rule has nothing to align there"* has a denominator-side twin, and #188 (widening the drive edge)
should know that widening also multiplies this cell's `Σ_e m_e`.

---

## 6. What the audit collapses — reported, not decided

**#181 (the missing target): untouched, and reinforced.** Nothing here supplies an amplitude
convention, rules among ADR-0007's three floors, or settles per-hop versus cumulative. §4 adds one
data point in its favour — `hop ≥ floor` is the *only* ill-formed comparison the denominator
participates in, so #181 is a specific defect rather than an instance of a general sloppiness, and
fixing it fixes the whole family. **Do not close it.**

**#184 (the cover): reshaped, not collapsed.** Its 2.15x is cross-edge alignment between two cells'
maps; every quantity here is within-cell. The two are independent factors and neither substitutes
for the other. But two entries change its arithmetic: the shortfall it is sized against is #178's
1.47x, and §5 adds a factor at the return path's last cell that **no cross-edge pressure can reach**
— the actuator's maps are pinned and its neighbours' alignment does not enter its own denominator.
So #184's target may be over- or under-sized depending on what #190 does to the denominator first.
**Do not close it; re-size it after #190.**

**Nothing else collapses.** ADR-0002, topology, the gauge's scale and attention are closed on
measurements (#142, #150) and this audit found nothing that reopens any of them — it reproduced
#142's untrained 41.8 and #182's binder table, which is evidence for those closures rather than
against.

---

## 7. Unmeasured claims, listed as findings

Per the ticket: an unmeasured claim is a finding, not a gap to fill silently.

1. **"`Σ_e m_e` tracks the largest eigenvalue"** (`b3a60fe`, still in `02:101` and `tick.py:230`).
   Never derived, never measured as such. §3 measures the closest well-defined version and finds a
   3.57x residual gradient.
2. **The equalisation property has never been measured at all** before §3 — it is defended by
   argument in `02`, cited by ADR-0005's boundary, and load-bearing in #190's decision 2.
3. **Boundary cells appear in no table in the record.** §5.
4. **`γ = 1.0`'s stated justification is stale** — `#85`'s check exists and is unwired (#180).
5. **The `max`'s guard has never been exercised.** `ρ` has been 2 for the life of the project and
   #150 closed moving it.

---

## 8. What this ledger does not do

It rules on nothing. It does not choose a denominator, does not decide equalisation's fate, does not
touch the `max`, and does not close #181 or #184. Those are
[#190](https://github.com/NGL321/patchworks/issues/190)'s, with this in hand.

Rig: `prototypes/gain-denominator-189/audit.py`, branch `worktree-ticket-189-gain-ledger`.
