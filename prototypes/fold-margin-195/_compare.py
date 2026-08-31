import numpy as np
from pathlib import Path

here = Path(__file__).resolve().parent
a = np.load(here / "surface-75000.npz")
b = np.load(here / "surface-100000.npz")
c = np.load(here / "surface-30000.npz")
for key in ("floors", "full", "stalk"):
    print(f"{key}: 75k vs 100k identical={np.array_equal(a[key], b[key])}  "
          f"maxdiff={np.abs(a[key]-b[key]).max():.3e}   |   "
          f"30k vs 75k maxdiff={np.abs(c[key]-a[key]).max():.3e}")
print()
f = b["floors"]
s = b["stalk"]
print(f"margin (stalk) quantiles: "
      + "  ".join(f"p{p}={np.percentile(s, p):.5f}" for p in (0, 1, 5, 25, 50, 75, 95, 100)))
print(f"floor quantiles:          "
      + "  ".join(f"p{p}={np.percentile(f, p):.5f}" for p in (0, 1, 5, 25, 50, 75, 95, 100)))
