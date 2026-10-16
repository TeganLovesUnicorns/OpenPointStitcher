#!/bin/sh
# Thin front door so the common jobs are one word instead of a remembered
# incantation. Anything unrecognised is passed straight through, so
# `docker run ... bash` and `docker run ... python -c ...` still work.
set -e

APP_USER=stitch

# --- make /out usable, then stop being root ---------------------------------
if [ "$(id -u)" = "0" ]; then
    # Empty /scans is the most common first mistake. Say so plainly rather than
    # letting it surface later as a stack trace. printf, not echo: dash's echo
    # eats backslashes, which turns a Windows path into nonsense.
    if [ -d /scans ] && [ -z "$(ls -A /scans 2>/dev/null)" ]; then
        printf 'Note: /scans is empty. Mount your .pts folder there, for example:\n' >&2
        printf '  docker run --rm -v "C:/path/to/scans:/scans:ro" -v "C:/path/to/out:/out" open-point-stitcher inspect\n' >&2
    fi

    # Match the owner of the mounted output folder so results belong to whoever owns it. Do not test [ -w /out ]:
    # root can always write, so that test would always pass.
    OUT_UID=$(stat -c %u /out 2>/dev/null || echo "")
    OUT_GID=$(stat -c %g /out 2>/dev/null || echo "")
    if [ -n "$OUT_UID" ] && [ "$OUT_UID" != "0" ]; then
        exec gosu "$OUT_UID:$OUT_GID" "$0" "$@"
    fi

    # /out is root owned or absent. Hand it to the unprivileged user. This does
    # change the ownership of the mounted host folder on Linux; on Windows and
    # macOS the mount ignores ownership and it is a no-op.
    [ -d /out ] && chown "$APP_USER:$APP_USER" /out 2>/dev/null || true

    # /projects gets the same ownership fix as /out.
    [ -d /projects ] && chown "$APP_USER:$APP_USER" /projects 2>/dev/null || true
    exec gosu "$APP_USER" "$0" "$@"
fi

case "$1" in
  help|"")
    cat <<'TXT'
OpenPointStitcher

  stitch       run the full pipeline over /scans, write to /out
  inspect      load every scan and print its stats, change nothing
  gpu-info     report what this build of Open3D can actually do
  test         run the quick and real-scan test layers
  hunt         run the same alignment in many processes and compare the results
  build-bench  build a standalone placement_bench.html from /scans, write to /out
  place-merge  apply a saved placement to /scans and merge, write to /out
  bench        the placement bench: a project picker, then the open project.
               Projects live under /projects, each with its own copy of its
               scans, its saved placement and its output. /scans is only an
               import source for bench.
  qa           run every test layer (Python, real scans, browser) and print
               one verdict. Needs the QA image (Dockerfile.qa).
  shell        open a shell inside the container

Your scans are mounted read only at /scans; nothing here ever writes to them.
stitch, inspect, hunt and place-merge read them from there and write to /out.
bench copies scans into a project under /projects on import, so /projects
must be mounted read-write.

bench listens on 0.0.0.0 inside the container's own network. The launchers
publish it on 127.0.0.1:8080 only, so nothing else on your network can reach it.
TXT
    ;;
  stitch)   shift; exec python -m open_point_stitch.cli stitch /scans -o /out/merged.pts --report /out/report.json "$@" ;;
  inspect)  shift; exec python -m open_point_stitch.cli inspect /scans "$@" ;;
  gpu-info) shift; exec python -m open_point_stitch.cli gpu-info "$@" ;;
  # quick + slow layers from the one list in tools/qa_layers.py; no browser in this image
  test)     shift; exec python tools/qa.py --quick --slow --no-canaries --out /out "$@" ;;
  hunt)     shift; exec python tools/hunt_nondeterminism.py --scans /scans --log /out/hunt_log.jsonl "$@" ;;
  build-bench) shift; exec python tools/build_bench.py /scans -o /out/placement_bench.html "$@" ;;
  place-merge) shift; exec python -m open_point_stitch.cli place-merge /scans -o /out/merged.pts --report /out/place_merge_report.json "$@" ;;
  bench)    shift; exec python -m open_point_stitch.server --projects /projects --scans /scans --out /out "$@" ;;
  qa)       shift; exec python tools/qa.py --scans /scans --out /out "$@" ;;
  shell)    shift; exec /bin/bash "$@" ;;
  *)        exec "$@" ;;
esac
