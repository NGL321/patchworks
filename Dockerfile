# The supported execution target (ADR-0012, docs/adr/0012-a-container-is-the-supported-execution-target.md).
#
# One file, two stages. `headless` is a complete image on its own -- `doctor`,
# `check`, `dome`, the benchmarks and the suite -- and is the guaranteed floor:
# it is the Ubuntu path CI has run on every push, packaged. `desktop` is
# `FROM headless` plus a display, and is what makes *demonstration* portable,
# which is the need the decision was made for.
#
# Both stages take the same commands, because both end in `patchworks "$@"`.

# Pinned by digest rather than by tag: a tag moves, and an image whose Python
# environment is "exactly what CI tested" cannot rest on a base that changes
# underneath it. This is the manifest-list digest, so it resolves to the right
# image on linux/amd64 and on linux/arm64 alike.
FROM python:3.12-slim-bookworm@sha256:0f5b26b9518d002b6173fd61daad821fa340635ebfec5bba471013f9ca114579 AS headless

# Unbuffered, because every command in this image prints a report a human is
# reading -- `doctor` a line per check, `check` its numbers -- and a buffered
# stream through `docker run` delivers them all at the end or not at all.
ENV PYTHONUNBUFFERED=1

# The same pair `.github/workflows/ci.yml` installs, and for the same reason:
# libgl1 is MuJoCo's runtime GL library and libosmesa6 is the software
# offscreen backend the render below goes through. Named and unpinned -- the
# base image is pinned by digest, which is what fixes the archive these resolve
# against; pinning Debian versions on top of that buys nothing and goes stale
# the first time the archive rebuilds a security update.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# **The dependency metadata first, the tree last.** `pip install -e` needs the
# source present, so the whole tree used to be copied here, above the installs
# -- and the cost was priced as "a rebuild after a one-line change reinstalls
# torch". That was accepted to avoid "a second place where the dependency set is
# written down", which is a real thing to protect and is protected here too:
# `pyproject.toml` is still the only file that names a dependency. Copying it
# early is not a second list, it is the same list, earlier.
#
# **What the original pricing could not see is the disk.** Docker keeps every
# invalidated layer as build cache, and on the WSL2 backend that cache lives in
# a virtual disk that grows and never shrinks on its own. Each rebuild wrote a
# fresh ~5.6 GB entry (torch ~0.8 GB, the dev extras ~4.8 GB), so a single day
# of ordinary edits left ~20 GB of cache behind permanently. It is not a slower
# rebuild; it is a few gigabytes of disk per edit, kept forever.
#
# The stub package is what lets the editable install run before its own source
# exists: hatchling needs `packages = ["src/patchworks"]` to be *there* to build
# the metadata, not to be complete. `COPY . /app` below replaces it with the
# real tree, and the `.pth` the editable install wrote already points at
# `/app/src`, so it resolves the real package from that point on.
WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
RUN mkdir -p src/patchworks && touch src/patchworks/__init__.py

# These two lines and nothing else reach pip, in this stage or the next, and
# they are `ci.yml`'s two install steps -- `--no-cache-dir` apart, which keeps
# pip's download cache out of the layer and installs nothing extra. That is what
# makes the sentence "the image's Python environment is bit-for-bit the set CI
# tested" true rather than nearly true, and it is why the desktop stage's
# display packages come from apt: Debian ships `novnc` and `websockify`, so
# nothing about the display has to be bought with a pip install. Do not tidy
# that into a `pip install novnc` -- it would put a package in this environment
# that no CI run has ever had in it.
#
# **The two files move together or not at all.** The torch line is here because
# it is there (ADR-0013): PyPI's `torch==2.2.2` is the CUDA build, ~3.7 GB of
# CUDA in an image whose suite asserts `torch.cuda.is_available() is False`.
# Changing it in only one of the two files would buy the same gigabytes and
# spend the sentence above, which is the one thing ADR-0012 exists to protect.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
RUN pip install --no-cache-dir -e ".[dev]"

# The tree itself, after the installs, so that editing a source file
# invalidates this layer and nothing below it -- which is the whole of the
# change described above. The image's contents are unaffected: the same two pip
# commands run against the same `pyproject.toml`, and the same files land in
# `/app`.
COPY . /app

# CI's value, and the one backend this repository has actually exercised. The
# desktop stage's entrypoint unsets it: `demo` runs the viewer and the
# environment's own `mujoco.Renderer` in one process, one process gets one
# backend, and the viewer needs the GLFW/GLX one.
ENV MUJOCO_GL=osmesa

# The mount point for run artifacts -- `run.npz`, snapshots -- so that a run in
# a container does not lose its output when the container exits:
#
#     docker run -v patchworks-work:/work ghcr.io/ngl321/patchworks \
#         check --ticks 2000
#
# Made a directory rather than a `VOLUME`: a declared volume is created
# anonymously on every run that does not mount one, which litters the host with
# dangling volumes holding nothing. The path exists either way, so a `--save`
# into it works whether or not anything was mounted -- with a mount the file
# outlives the container, and without one it does not.
RUN mkdir -p /work

# So that `docker run ... check` reads the way the CLI does. `--entrypoint
# pytest` is how the suite is run in here, and `--entrypoint python` reaches
# the benchmarks.
ENTRYPOINT ["patchworks"]
CMD ["--help"]


FROM headless AS desktop

# The display stack, all from apt (see the pip comment above).
#
#   xvfb                     an X server with no screen behind it
#   x11vnc                   that X server, served as VNC
#   novnc + websockify       that VNC, served to a browser over HTTP
#   openbox                  a window manager: `demo` opens two windows, and
#                            without one they cannot be raised, moved or
#                            focused, which is half of driving them by hand
#   x11-utils                xdpyinfo, which the entrypoint waits on rather
#                            than sleeping a guessed number of seconds
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        xvfb x11vnc novnc websockify openbox x11-utils \
    && rm -rf /var/lib/apt/lists/*

# chmod here rather than relying on the checkout's mode bit: the build
# context is a working tree, and a Windows one carries no execute bit to
# copy.
COPY docker/desktop-entrypoint.sh /usr/local/bin/desktop-entrypoint
RUN chmod 755 /usr/local/bin/desktop-entrypoint

# The image's screen, named here rather than left to whatever Xvfb defaults to,
# because `--pitch` and `--scale` size the panel against the screen it opens
# on: a geometry that varied would make those flags mean something different
# from one run to the next. Overridable with `-e` for a screen of another size
# -- what matters is that it is the same on every run that does not say
# otherwise.
ENV PATCHWORKS_DISPLAY=:0 \
    PATCHWORKS_GEOMETRY=1600x1000x24 \
    PATCHWORKS_NOVNC_PORT=6080

EXPOSE 6080

ENTRYPOINT ["desktop-entrypoint"]
CMD ["demo"]
