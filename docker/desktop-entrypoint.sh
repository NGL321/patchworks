#!/bin/sh
# The desktop tag's entrypoint: a display, then the same command the headless
# tag takes (ADR-0012, decision 2).
#
# Three jobs, in order, and nothing else:
#
#   1. Start an X server the container owns, a window manager over it, and a
#      route from that server to a browser (x11vnc -> websockify -> noVNC).
#   2. Unset MUJOCO_GL. The image's default is `osmesa`, which is right for
#      every headless command and wrong here: `demo` runs `mujoco.viewer` and
#      the environment's own `mujoco.Renderer` in one process, one process gets
#      one backend, and the viewer needs the GLFW/GLX one. Unset, MuJoCo picks
#      GLX, which mesa serves in software -- for the window and for the 64x64
#      observation render alike.
#   3. Exec the command, so that it is PID 1's successor and a `docker stop`
#      reaches it rather than this script.
#
# Arguments are a `patchworks` command, as on the headless tag:
#
#     docker run -p 6080:6080 ...:desktop demo
#
# unless the first is `--`, in which case the rest is exec'd verbatim. That is
# the route to the two-window surface, which is a module until #122 gives it a
# subcommand:
#
#     docker run -p 6080:6080 ...:desktop -- \
#         python -m patchworks.surface.watch --ticks 2000 --save /work/run.npz
set -eu

display="${PATCHWORKS_DISPLAY:-:0}"
geometry="${PATCHWORKS_GEOMETRY:-1600x1000x24}"
port="${PATCHWORKS_NOVNC_PORT:-6080}"

Xvfb "$display" -screen 0 "$geometry" -nolisten tcp &
export DISPLAY="$display"

# Waited for rather than slept on: Xvfb is ready when a client can talk to it,
# and how long that takes is a property of the machine, not a number to guess.
attempt=0
until xdpyinfo -display "$display" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt 100 ]; then
        echo "desktop-entrypoint: Xvfb did not come up on $display" >&2
        exit 1
    fi
    sleep 0.1
done

# A window manager, because `patchworks.surface.watch` opens two windows and
# without one they cannot be raised, moved or focused -- which is half of
# driving them by hand.
openbox &

x11vnc -display "$display" -rfbport 5900 -forever -shared -nopw -quiet &
websockify --web=/usr/share/novnc "$port" localhost:5900 >/dev/null 2>&1 &

echo "desktop-entrypoint: $display at $geometry; noVNC on http://localhost:$port/vnc.html" >&2

# The GL remedy `doctor` prints inside this image reads MUJOCO_GL, so unsetting
# it here is visible there: on the desktop tag the backend really is MuJoCo's
# own default, and the report says so rather than naming a value that is not in
# force.
unset MUJOCO_GL

if [ "${1:-}" = "--" ]; then
    shift
    exec "$@"
fi
exec patchworks "$@"
