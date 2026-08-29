# ADR-0013: The CPU build of torch is what CI tests and what the image ships

**Status:** accepted

**Amends [ADR-0012](./0012-a-container-is-the-supported-execution-target.md), decision 4.** That
decision's sentence — *the image's Python environment is exactly what CI tested* — stands, and is
the reason this change touches `ci.yml` at all rather than only the `Dockerfile`.

## Context

Raised by [#133](https://github.com/NGL321/patchworks/issues/133), out of the measurement
[#131](https://github.com/NGL321/patchworks/issues/131) took after building the image.

`torch==2.2.2` resolved from PyPI is the **CUDA build**, and on `linux/amd64` it brings the CUDA 12.1
runtime with it: `site-packages/nvidia/` is 2852 MB and `libtorch_cuda.so` and
`libtorch_cuda_linalg.so` inside the wheel are another 897 MB — **~3.7 GB of a 5.19 GB image**, and
the same download on every CI run.

**Nothing in this repository wants it, and three places say so.**
[`09-the-build-stack.md`](../spec/09-the-build-stack.md)'s *The compute target* declares CPU the
target and says the AMD desktop "is not a GPU target"; ADR-0012 declined GPU support and
`MUJOCO_GL=egl` with it, on the grounds that a driver is something the image cannot assume; and
`tests/test_package.py::test_torch_is_pinned_and_cpu_only` **already asserts**
`torch.cuda.is_available() is False`. That test is named for a property the install line did not
hold. The suite has been asserting the absence of a capability that 3.7 GB was being spent to
provide.

**Why this was not simply done in the `Dockerfile`.** Because that would have been the one thing
ADR-0012 exists to prevent. Its argument for a container over Windows-native support is evidence —
*"a container inherits the evidence that already exists instead of acquiring more"* — and an image
built on a torch binary CI has never run spends exactly that inheritance, while the sentence "what
the container runs is what CI tested" quietly stops being true. This repository's standing failure
mode is prose claiming something it has not got, and that would have been a fresh instance of it.

**Two facts made the change cheap.** The CPU index carries **both** architectures for this pin —
`torch-2.2.2+cpu-…-linux_x86_64.whl` and the plain `torch-2.2.2-…-manylinux_2_17_aarch64.whl`, which
is the same wheel PyPI serves — so one install line works everywhere. And `pyproject.toml` needs no
edit: under PEP 440 a `torch==2.2.2` specifier with no local version of its own is satisfied by
`2.2.2+cpu`, and `test_torch_is_pinned_and_cpu_only` asks `startswith("2.2.2")`. The pin does not
move. Where the wheel comes from does.

## Decision

**Both `ci.yml` and the `Dockerfile` install torch from `download.pytorch.org/whl/cpu`, in the same
change**, so that ADR-0012's decision 4 is preserved rather than broken: the image's environment
still equals CI's, and CI now exercises the build the image ships.

1. **Two install lines, not one line with an extra index.** `pip install --index-url <cpu>
   torch==2.2.2` first, then the editable install, which finds the requirement already satisfied. A
   single line carrying `--extra-index-url` would work only through pip's index ordering and through
   `2.2.2+cpu` sorting above `2.2.2` — a resolution that depends on two coincidences is not a thing
   to pin.
2. **`pyproject.toml` does not move.** The dependency set is unchanged, which is why
   `PERMITTED_DEPENDENCIES` and `PERMITTED_EXTRAS` in `TestBothChecksRunInCI` are unchanged too.
3. **The guard is edited, not widened.** `PERMITTED_RUN_STEPS` pins every `run:` by value, so this
   reddens it; the class gains one more pinned line and loosens no rule. `ci.yml`'s own comment
   already calls that "the whitelist working rather than a false positive", and this is the first
   occasion to act on it.
4. **arm64 is untouched by construction.** torch gates every `nvidia-*` requirement behind
   `platform_machine == "x86_64"`, so the arm64 image never carried CUDA and no figure here is an
   arm64 figure.

## Consequences

- **The amd64 image loses ~3.7 GB, and every CI run loses the same download.** The front door of a
  README is a `docker run` line, and what that line pulls is now most of an order of magnitude
  smaller.
- **A second index is now in the trusted set.** `download.pytorch.org` is PyTorch's own, and it is
  named on one pinned line installing one pinned package — but it is a supply-chain surface that did
  not exist before, and it is named here rather than left to be discovered in a diff.
- **`torch.__version__` becomes `2.2.2+cpu` on amd64 and stays `2.2.2` on arm64.** Anything that
  compares the version exactly rather than by prefix will find that; `test_package.py` asks by
  prefix, and `doctor` only reports the string.
- **A GPU is now a rebuild away rather than an install away.** If the rented NVIDIA box in
  [`09-the-build-stack.md`](../spec/09-the-build-stack.md) is ever reached — it is gated on
  measurement and nothing has fired it — that machine needs an install line of its own. That is the
  right shape: it is a target with no automated evidence behind it, and it should have to say so.

## Alternatives considered

- **Changing only the `Dockerfile`.** The cheap version, and it loses the sentence ADR-0012 was
  written to protect. Rejected above at length.
- **One install line with `--extra-index-url`.** Rejected in decision 1: correct only by resolution
  order.
- **Editing `pyproject.toml` to pin `torch==2.2.2+cpu`.** It would put a platform-specific local
  version in the dependency set every consumer of the package resolves, including the macOS
  development laptop, where no `+cpu` wheel exists at all. The pin is a statement about *which
  torch*, not about which build of it for which machine.
- **Uninstalling the `nvidia-*` packages after the fact.** It would leave the 897 MB of CUDA inside
  the torch wheel itself, and leave the environment in a state no `pip install` produces — which is
  worse than either end of the choice.
- **Shrinking the image by other means** — a slimmer base, a multi-stage copy of `site-packages`, a
  non-editable install. Each is a real argument and none of them is this one: this removes only what
  the repository already says it does not want.
