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

# The whole tree, before the install, because the install is an editable one:
# `pip install -e` needs the source present and leaves a path to it rather than
# a copy. The cost is that any edit invalidates the layer below, so a rebuild
# after a one-line change reinstalls torch. Accepted rather than worked around
# with a two-stage requirements copy: the alternative is a second place where
# the dependency set is written down, and one of the things this image is for
# is that there is exactly one.
WORKDIR /app
COPY . /app

# `pip install -e ".[dev]"` and nothing else reaches pip, in this stage or the
# next. That is what makes the sentence "the image's Python environment is
# bit-for-bit the set CI tested" true rather than nearly true, and it is why
# the desktop stage's display packages come from apt: Debian ships `novnc` and
# `websockify`, so nothing about the display has to be bought with a pip
# install. Do not tidy that into a `pip install novnc` -- it would put a
# package in this environment that no CI run has ever had in it.
RUN pip install --no-cache-dir -e ".[dev]"

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
