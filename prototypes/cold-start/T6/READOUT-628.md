# B56 (#628): the two-sided channel-return term on a trained surface

`reserve_p12` seed 42, `worktree-b56-channel-holonomy-628`. Four arms to 2,000
ticks plus two construction sweeps. **Nothing here is shipped and nothing here is
a decision** — #628 is a `wayfinder:prototype` ticket and #532's standing note is
*plan, don't do*.

Instruments, all in `prototypes/cold-start/T6/`:

| file | what it does | cost |
| --- | --- | --- |
| `b56_channel.py` | trains the four arms, reports every column #628 names at every rung | 11–13 min/arm |
| `b56_analyse.py` | stamps each run's own stall horizon; applies B48's joint rule | seconds |
| `b56_stagger6.py` | every stagger `0..k_v−1` on three seeds — #628's cheap question | 0.4 min |
| `b56_stagger_width.py` | the same sweep at `p ∈ {8, 12, 16}` | 0.4 min |
| `b42_stagger.py` | **vendored unmodified** from `worktree-b42-flat-bundle-605` at `7e02410`; not on `main` | — |

---

## §0 — the cheap question first, because it came back bigger than asked

#628: *"If this rig can cheaply say whether stagger 6 is a finding or an
artifact, that is worth more than another arm."*

**It is a finding, it is not about stagger 6, and it is not about the seed.**

B42 sampled staggers `{0, 1, 2, 3, 4, 6, 8}`. Sweeping **all** of `0..19` on
**three** seeds, the staggers reading `sigma_max` 1.000 at `channel_return`
1.0000 are

> `{0, 1, 2, 6, 7, 10, 13, 14, 18, 19}` — **identical on all three seeds**

and `edge_overlap` agrees to four decimals across seeds. Only `identification`
and the *collapsed* arms' `channel_return` move with the seed. **Exactness and
lane overlap are functions of the row arithmetic, not of the random frame.**
Stagger 6 looked isolated because the sampled set hit one member of the far
family and missed the rest.

The differentiation available at perfect channel return is much higher than
B54's 0.5444:

| stagger | `identification` | `channel_return` | differentiation |
| ---: | ---: | ---: | ---: |
| 1 | 0.6075 | 1.0000 | 0.3000 |
| 2 | 0.8656 | 1.0000 | 0.3333 |
| 6 | 0.9531 | 1.0000 | 0.5444 |
| 13 | 0.9359 | 1.0000 | 0.6692 |
| 18 | 0.8535 | 1.0000 | 0.6778 |
| **19** | 0.6388 | 1.0000 | **0.7321** |

### §0.1 — and it is arithmetic in `k_v = n − p`

`b56_stagger_width.py`, one seed, `p ∈ {8, 12, 16}`:

| `p` | `k_v` | exact staggers | best differentiation at `channel_return` 1.0000 |
| ---: | ---: | ---: | ---: |
| 8 | 24 | 5 of 24 | **0.8036** (`identification` 0.8665) |
| 12 | 20 | 10 of 20 | 0.7321 (`identification` 0.6388) |
| 16 | 16 | **16 of 16** | 0.6061 (`identification` 0.5869) |

**A larger reserve buys exactness everywhere and lowers the differentiation
ceiling; a smaller one raises the ceiling and makes exactness rare.** Since `p`
is an initialisation and not a constant ([B42](https://github.com/NGL321/patchworks/issues/605)),
this is a lever with a priced trade rather than a fact about the dome. It is
**construction only** — no arm here was trained, and under
[B49](https://github.com/NGL321/patchworks/issues/616) that is what these numbers
are on.

---

## §1 — the trained surface: clause 1 works, and it works on a dead world

Live horizon **100 ticks** on every run, stamped from that run's own MuJoCo
`std_max` and inherited from nothing (`std_max` 1.28 at 100, 6.0e-04 at 150).
Rungs past it are marked `*`.

`channel_return` and `identification`, median over the 45 `wide` cycles:

| ticks | baseline chan | channel chan | unbounded chan | baseline diff | channel diff | unbounded diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.2006 | 0.2006 | 0.2006 | 0.3683 | 0.3683 | 0.3683 |
| 100 | 0.2571 | 0.2811 | 0.2591 | 0.3585 | 0.3585 | 0.3585 |
| 150\* | 0.2308 | 0.2976 | 0.2911 | 0.3567 | 0.3567 | 0.3566 |
| 500\* | 0.2341 | 0.4608 | 0.3860 | 0.3486 | 0.3436 | 0.3428 |
| 2000\* | 0.3223 | **0.7899** | 0.6768 | 0.3354 | **0.3122** | 0.3117 |

**Clause 1 is a real lever**: `channel_return` 0.2006 → 0.7899 against baseline's
own drift to 0.3223, at unit-normalised weight 0.05.

**And it never needed the world.** All of that movement is past the stall stamp.
That is not the usual caveat: clause 1's gradient is computed from the maps
alone, and [B17](https://github.com/NGL321/patchworks/issues/565) established the
transport operator never touches a stalk — so this term trains on a motionless
body by construction. **Both its gain and its cost below are computed from the
maps and are not artifacts of the stall.** What the stall does mean is that the
*shipped* transport rule was seeing a frozen stimulus over the same rungs, so
`baseline` past 150 is drift under constant input rather than learning.

---

## §2 — clause 2 does nothing, in either form

### §2.1 — the hinge is **bit-identical** to clause 1 alone

`628-channel-hinge-seed42-2000.json` and `628-channel-channel-seed42-2000.json`
agree on **every field at every rung apart from the label string.** `ident_loss`
is exactly `0.0` at all seven rungs. The `identification` surrogate's **minimum
over the entire run is 0.9267**, against `GAMMA = 0.6075`.

**`GAMMA` re-anchored, which #628 asked for.** 0.6075 was read off B42's
stagger-1 row — a **construction** sweep. On the trained surface `identification`
sits at **0.97–0.99** from tick 0 and never falls below **0.9267** under any arm.
The construction anchor is not slightly off; it is **below the trained surface's
entire operating range**, so the hinge is inert by more than 50%.

**And re-anchoring it does not rescue it.** A `GAMMA` high enough to fire — 0.95
or above — is active at essentially every point the surface visits, which is the
`unbounded` clause with a cap rather than a floor that is *"inert above it"*. §2.2
is then what it inherits.

### §2.2 — the unbounded clause moves `identification` and not differentiation

`identification` 1.0017 → **1.0412**, i.e. pushed **past chance**, exactly the
risk B54 Q8 named. What it bought:

| | differentiation at 2000\* | vs baseline |
| --- | ---: | ---: |
| baseline | 0.3354 | — |
| channel (clause 1 alone) | 0.3122 | −0.0232 |
| **unbounded** (both clauses) | **0.3117** | **−0.0238** |

**Clause 2 maximised leaves differentiation indistinguishable from clause 1
alone, and marginally worse.** It also costs clause 1: `channel_return` 0.6768
against 0.7899.

**B33's 950× is not reproduced here and could not be.** Unit-normalising each
clause's gradient before weighting removes exactly the scale asymmetry that
produced it (`chan_grad_raw` 0.715 against `ident_grad_raw` 0.275 at 2,000 — a
factor of 2.6, not 950). What this rig can say is the *direction* the surface
moves, and it moves toward chance without protecting the thing clause 2 exists to
protect.

### §2.3 — so the two-sided term is one-sided in practice

Clause 2 was what discharged
[B42 (#605)](https://github.com/NGL321/patchworks/issues/605)'s standing
constraint — *any objective carrying a local holonomy term states what stops its
optimum being collapsed lanes*. Measured, it stops nothing. **The discharge does
not hold as specified.**

---

## §3 — B48's joint rule, applied

*Agreement rising while differentiation stays nonzero*, against `baseline` at the
same rung:

| arm | rung | chan(w) | Δ chan | diff | Δ diff | Δ `k_v` | scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| channel / hinge | 100 | 0.2811 | +0.0239 | 0.3585 | −0.0000 | −0.02 | no |
| channel / hinge | 2000\* | 0.7899 | +0.4676 | 0.3122 | −0.0232 | −0.01 | **bought out of differentiation** |
| unbounded | 100 | 0.2591 | +0.0019 | 0.3585 | +0.0000 | −0.01 | yes, on a +0.0019 move |
| unbounded | 2000\* | 0.6768 | +0.3545 | 0.3117 | −0.0238 | +0.04 | **bought out of differentiation** |

**Two readings, and the honest one is both.**

- **At the live rung the term is not measurable.** +0.0239 on `channel_return`,
  differentiation unmoved to four decimals. Nothing here scores and nothing here
  fails; B33's practice was to read the verdict at the last live rung, and read
  that way this ticket has no result.
- **Past the stall the term is large and it is paid for.** Every arm that raises
  `channel_return` lowers differentiation below baseline's own drift.

**But it is not the full-stalk collapse.** Differentiation lands at 0.3122, not
0.0000. The full-stalk form's optimum **is** differentiation 0.0000 by
construction; here a 3.9× rise in `channel_return` costs 7% of differentiation
relative to baseline. **The channel form is quantitatively different in kind from
what was suspended**, which is what B54's §1 claimed and this is the trained
check of it.

### §3.1 — the exposure it cost: none

Rank-measured `k_v` is **20.0 on every arm at every rung**, and its participation
ratio tracks baseline to within 0.04 (9.0 → 8.7 on all four arms alike). **The
differentiation the term spends does not come out of exposure** — it is a
reallocation inside a window whose size never moves, which is
[B22](https://github.com/NGL321/patchworks/issues/571)'s construction-time clamp
showing up again.

---

## §4 — the ceded question (B54 Q5), checked

B54 Q5 was **ceded**: that this term does not inherit
[#610](https://github.com/NGL321/patchworks/issues/610)'s standing constraint,
on the argument that retention is temporal on `K` while holonomy is spatial on
the transport operator. Measured against baseline at the same rung:

| arm | rung | Δ `rho(used)` | Δ `tau` |
| --- | ---: | ---: | ---: |
| channel / hinge | 100 | −0.00011 | −0.51 |
| channel / hinge | 2000\* | −0.00104 | −2.80 |
| unbounded | 100 | −0.00011 | −0.49 |
| unbounded | 2000\* | +0.00025 | +0.72 |

**At the live rung, nothing.** Both arms sit half a tick of `tau` from baseline
against baseline's own drift of 68.3 → 53.1 across the run. Past the stall the
largest effect is `channel`'s −2.80 on a median of 50.3, and it is **not
consistent in sign** — `unbounded` moves it the other way. **The ceded argument
survives its first measurement**: on this surface the two terms are not competing
in any direction this rig can see. It is one seed and one surface, and the check
is a null rather than a confirmation.

---

## §5 — B19's `corr(state, emitted)`, and B50's advisory

**B54 Q4's refusal is discharged by joint report.** `corr_learned` against the
Haar control at every rung, all four arms: learned **+0.10 to +0.45**, Haar
**+0.89 to +0.97**. B19's degeneracy is intact on every arm and the term neither
worsens it (`channel` reads +0.437 at 2,000 against baseline's +0.358) nor fixes
it. The column is noisy enough across rungs — baseline alone runs
+0.243/+0.252/+0.340/+0.184/+0.136/+0.450/+0.358 — that **no arm difference here
is a signal**, which is itself the answer: **no CycleGAN side-channel is visible,
and none is excluded.**

B33's scale-free representation floor is carried as a diagnostic per B54 Q9 and
does not ship in the term.

**B50's advisory, from #624's amendment, is netted out by the trained surface
itself.** The confound was that `channel_return` 1.0000 might be the forced
per-hop intersection `f` — geometry rather than agreement — in which case a term
driving it would be inert. On the trained surface `channel_return` starts at
**0.2006**, not 1.0000, and the term moves it to 0.7899. **A saturated quantity
cannot be moved 3.9×**, so whatever `f` explains at construction, it is not what
clause 1 is pushing on here.

---

## §6 — what is not settled

- **One seed, one surface.** Everything in §1–§5 is `reserve_p12` seed 42. B38's
  13× horizon spread between seeds of one arm is reason enough not to read a
  second seed off this one.
- **The live window has no result.** The term acts on the maps and not the world,
  so it trains past the stall — but *whether the shipped rule would have gone
  somewhere different on a moving body* is not answerable from here.
- **`GAMMA` has no defensible trained anchor yet.** §2.1 says where it would have
  to sit to fire; it does not say that is the right place for it, and §2.1's own
  argument is that anywhere it fires it stops being a hinge.
- **The stagger family's mechanism is measured, not explained.** §0 establishes
  the exact set is arithmetic in `k_v` and seed-invariant. *Why* `{0,1,2,6,7,10,
  13,14,18,19}` and not another ten is not derived here.
