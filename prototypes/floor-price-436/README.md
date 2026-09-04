# The runs behind #436

ADR-0032's third pre-registration, taken on `benchmarks/floor_price.py`. Raw
console output, kept because the resolution quotes medians out of it and a
quoted median with no run behind it is the shape this repository has had to
repair before.

| file | command | what it is |
|---|---|---|
| `read-30k.txt` | `floor_price.py price --learn 30000` | the full read at 30k |
| `read-100k.txt` | `floor_price.py price --learn 100000` | the same at 100k, because ADR-0032 names the 30k mistake |
| `map-effective-rank-30k.txt` | `floor_price.py ranks --learn 30000` | the premise alone: map effective rank with no floor |

Seed 42, `real` dome, `train` split, on the branch of
[#434](https://github.com/NGL321/patchworks/pull/434) merged into `main` — the
floor is not on `main`, so both surfaces are trained in one process against one
harness, the *before* one with `RestrictionMaps._flatten` patched out.

**`read-100k.txt` predates the control and premise sections** and does not carry
them. The control (`floor_price.py control`) needs no training and is horizon
independent — it is exactly-flat maps drawn at random, so there is nothing in it
a training horizon could move — and the premise is in
`map-effective-rank-30k.txt`. Re-running `price` today reports all of it in one
pass.

The verdict is on the issue, not here.
