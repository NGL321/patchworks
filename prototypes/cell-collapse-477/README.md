# #477: what collapses a tenth of the cells by 100k

Four reads, resolving [#477](https://github.com/NGL321/patchworks/issues/477). Run each from the
repository root.

## The problem these exist for

[#132](https://github.com/NGL321/patchworks/issues/132)'s committed 100k JSON carries the per-cell
arrays this question needs, but its `levels` and `p_v` fields are **all-zero**: `read.py`'s
`cell_context` raises, and the `except Exception` fallback at `read.py:265` writes zeros without
saying so. So the dead cells could be *counted* from that file but not *identified*.

The fix is cheap and is what `structure.py` does: the dome is deterministic given `DEFAULT_SPEC`, so
rebuilding it recovers every construction property. Predicting-cell ids are contiguous — 263 to 412
— so per-cell array index `i` is cell `263 + i`.

## The reads

| script | what it answers |
|---|---|
| `structure.py` | Which cells are dead at each checkpoint, and their level, degree, bus, `p_v` and drive-adjacency. Also whether death is one-way. |
| `columns.py` | The dead rate per **column**, and the `R²` of construction against the collapse — the [#233](https://github.com/NGL321/patchworks/issues/233) standard. |
| `trajectory.py` | `ρ(K)`, `nonnormality` and `stable_rank` per construction group across all ten checkpoints. |
| `band_faces.py` | Splits ADR-0015's projection into its two faces. Takes `ticks` and `seed`: `python prototypes/cell-collapse-477/band_faces.py 20000 0`. Runs live, ~2h at 20k. |

The first three read #132's committed JSON and take seconds. Only `band_faces.py` runs the agent.

## Why `band_faces.py` exists

`CellOperators.project` (`body.py:730`) clamps into `[1/ρ_K, 1]` at **both** ends, and the mask it
returns — `target != norms` — does not say which face fired.
`benchmarks/projection_firing.py` therefore reports a rate that conflates *retention was cut* with
*retention was restored*, and [#335](https://github.com/NGL321/patchworks/issues/335)'s mechanism
claim is specifically about the upper face.

The reading: **the lower face never fires, at any level.** #335's mechanism is intact.

## The horizon matters, and the rig's default is short

`benchmarks/projection_firing.py` defaults to `TICKS = 3000`, which is before the apex knee. At 3000
the cut reads **0.397 CLEAR**; at 20000 the same rig reads **1.34 CROSSED**. Any future read of that
cut should say which horizon it was taken on.

## Surface

All readings are post-[ADR-0031](../../docs/adr/0031-the-sparsity-pressure-is-deleted.md) and
post-floor: #132's `132-postfloor-real-train-seed{42,43,44}-100000.json` for the per-cell arrays, and
`main` at `18f6781` for the two live runs. The projection read measures the post-step `project()`
that `learning.py:381` still calls; [#466](https://github.com/NGL321/patchworks/issues/466) owns
moving it into the forward path.
