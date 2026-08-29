# ADR-0012: A container is the supported execution target

**Status:** accepted

## Context

Raised by [#125](https://github.com/NGL321/patchworks/issues/125), which was opened for a concrete
reason: the owner wanted to show the current state on an unrelated Windows machine after a fresh
clone, and the honest answer was *probably, but do a dry run first*. That ticket named two ways to
close the gap and deliberately refused to pick between them, because they solve overlapping problems
at very different costs and doing both is probably wrong.

**What was already portable by construction, and is not what this decides.** The platform question is
asked in exactly one place — `needs_mjpython(platform)` returns `platform == "darwin"`;
`surface/window.py` spawns its child with `subprocess.Popen` rather than `fork`, and its docstring
already claims the arrangement "is the same on Linux and Windows"; and the pinned wheels
(`torch==2.2.2`, `mujoco==3.10.0`) exist for Windows as well as for Linux on both x86_64 and aarch64.
This is not a port. It is closing the gap between *designed to work there* and *known to work there*.

**The asymmetry that decides it is evidence, not effort.** CI has run the Ubuntu path on every push
for the life of the repository. Every claim about any other platform is untested — macOS is the
development machine and is equally unverified in CI. Windows support would mean acquiring that
evidence for a third platform and then holding it: a `windows-latest` job, `os.execv` semantics that
differ there (it spawns rather than replaces, so a console returns to its prompt while a window is
still coming up, and the CLI's documented exit codes stop meaning what they say), and every command in
the README and in `cli.py`'s remedies rewritten to print what runs on the platform it is printed on. A
container inherits the evidence that already exists instead of acquiring more.

**Three facts found while grilling this, each of which moved the decision:**

- **`--replay` is not display-free.** #125 assumed `patchworks check` and `--replay` "would work
  trivially" without a display. `watch.py`'s `replay` opens no *scene* window, but it does open a
  `FrameWindow` — GLFW, in a child process. The genuinely headless surface is `doctor`, `check`,
  `dome`, `pytest` and the benchmarks, and nothing else.
- **`demo` is one process doing two GL jobs.** `gestures.drive` runs `mujoco.viewer` *and* the
  environment's `mujoco.Renderer` for the 64×64 observation. One process gets one `MUJOCO_GL` backend,
  and the viewer requires the GLFW/GLX one — so a container defaulting to CI's `osmesa` breaks `demo`
  unless the display path overrides it, and the observation render then goes through mesa's software
  GLX too, not only the window.
- **The CI guard is a whitelist over one file.** `TestBothChecksRunInCI` pins every `run:`, every
  `uses:`, the top-level keys, every `env:`, and `pyproject.toml`'s `[project]` and `[tool.hatch]`
  tables of `.github/workflows/ci.yml`, by value. It cannot see a second workflow, and it cannot see a
  Dockerfile.

## Decision

**A container is the supported execution target.** `linux/amd64` and `linux/arm64`, published, with
the README's front door a `docker run` line. Windows-native support is not pursued;
[#125](https://github.com/NGL321/patchworks/issues/125) is superseded rather than deferred, and its
two platform-neutral criteria are carried forward.

1. **One Dockerfile, two modes.** A `headless` stage that is a complete image on its own — `doctor`,
   `check`, `dome`, the benchmarks, the suite — and a `desktop` stage built `FROM headless` that adds
   Xvfb, a window manager, x11vnc and noVNC, so the two windows open against the container's own X
   server and a human watches them in a browser. Both published as tags. The headless tier is the
   guaranteed floor; the desktop tier is what makes *demonstration* portable, which is the need behind
   the ticket.
2. **`ENTRYPOINT` is `patchworks`**, so `docker run … check` reads the way the CLI does. The desktop
   tag wraps that in a script which starts the display stack, unsets `MUJOCO_GL`, and execs the same
   command.
3. **`MUJOCO_GL=osmesa` is the image default** — CI's value, the one backend this repository has
   actually exercised — overridden in the one place that opens windows. `egl` is declined: it needs a
   driver the image cannot assume.
4. **The image's Python environment is exactly what CI tested.** `pip install -e ".[dev]"` and nothing
   else; the whole display stack comes from apt, where Debian ships `novnc` and `websockify`. The base
   image is pinned by digest; apt packages are named and unpinned.
5. **`ci.yml` is untouched.** A separate workflow builds both architectures on their native runners —
   `ubuntu-latest` and `ubuntu-24.04-arm`, free for public repositories — runs `doctor` and `check` in
   the built image, and pushes a manifest to GHCR from the default branch only.

## Consequences

- **"Supported" now means a platform with automated evidence behind it.** The container is the only
  target that will have it on more than one architecture. The Intel-macOS laptop stays what
  [`09-the-build-stack.md`](../spec/09-the-build-stack.md) already calls it — a correctness-only target
  checked by hand — and running natively on a host is best-effort rather than claimed. That spec
  section is amended in the same change: it enumerates the laptop, the AMD desktop and a rented NVIDIA
  box, and a target that goes unmentioned there is the spec drifting from the build.
- **A Windows user is served, but not natively.** Docker Desktop is the dependency this decision adds,
  and it is a real cost to someone who has none. What they get in exchange is a command with no venv,
  no `mjpython`, no clone and no interpreter version in it.
- **The demo runs on software GL.** Mesa's llvmpipe carries both the viewer and the 64×64 observation
  render. That it is fast enough is a claim to be **measured in the container and recorded**, not
  assumed — and the resulting figures are a different machine's, not a correction to spec 09's host
  table.
- **One run exists that the CI guard cannot see.** The docker workflow is outside
  `TestBothChecksRunInCI`'s whitelist by construction. This does not weaken the guard's premise — that
  premise is that the pinned suite runs whole on every push, and a workflow running no `pytest` narrows
  nothing. **If that workflow ever grows a `pytest`, the guard has to grow a Dockerfile reader in the
  same change**, or the sentence "what the container runs is what CI tested" stops being true while
  everything stays green. Recorded here because it is precisely this repository's standing failure
  mode: prose claiming an enforcement it has not got.
- **`doctor`'s GL remedy becomes situational.** Its current text — install osmesa, or check the
  platform's GL drivers — is advice for a host, and a human reading it inside the image would go
  looking for a driver that is not the problem. The container is detected for the remedy's wording
  only; it is not a new check, because being in a container is not a failure and every line `doctor`
  prints is an observation with a verdict.

## Alternatives considered

- **Windows-native support** — the loser, and it lost on evidence rather than on difficulty. It would
  need a `windows-latest` job and everything that job turned red fixed, a resolution of `os.execv`'s
  different meaning there, and every command in the README and in `cli.py`'s remedies made
  platform-aware. All of that buys one more platform; the container buys every platform that runs one.
  Not closed forever: if the container proves insufficient, that is a fresh ticket with this ADR as its
  context, not a standing obligation.
- **Doing both.** Refused for the reason #125 gave when it raised the fork.
- **Host display passthrough** (`-e DISPLAY -v /tmp/.X11-unix`, WSLg, XQuartz) instead of a display
  inside the image. Cheaper, and it fails at the thing this is for: it delivers a window on hosts that
  already had a display server reachable from a container, which is three different incantations and is
  flakiest on exactly the borrowed laptop the demo happens on.
- **Running the suite inside the container in CI.** Rejected: it moves the pinned `pytest` invocation
  inside a Dockerfile the whitelist does not read, so the guard would have to grow a Dockerfile reader
  to stay true — a large widening to buy nothing the native job already proves.
- **A guard test over the Dockerfile**, pinning its install lines the way `ci.yml`'s are pinned.
  Declined *for now*, and conditionally: the guard's whitelists are load-bearing because the pinned
  suite runs through them, and under this decision no suite runs through the Dockerfile. The condition
  under which it becomes required is stated in *Consequences* above.
- **Publishing only on release tags.** The repository has no releases and the README says so; this
  would publish nothing.
- **Building arm64 under QEMU emulation** on an x86_64 runner. Unnecessary since GitHub's
  `ubuntu-24.04-arm` hosted runners are free for public repositories, and a torch install under
  emulation is slow enough to make the workflow a nuisance.
