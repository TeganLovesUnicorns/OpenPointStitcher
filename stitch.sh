#!/usr/bin/env bash
# Linux twin of stitch.ps1: same commands, same folders, same image tags.
#
#   ./stitch.sh build
#   ./stitch.sh inspect
#   ./stitch.sh qa | qa-quick | qa-medium | qa-slow
#   ./stitch.sh stitch
#   ./stitch.sh hunt --runs 40
#   ./stitch.sh bench | bench-update | bench-stop | bench-logs   (add -Dev for dev mode)
#   ./stitch.sh gpu-info
#
# Reads ./scans and writes ./out unless SCANS / OUT are set.
# OPENPOINT_NO_GPU=1 runs without --gpus all, for a machine with no usable NVIDIA card.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmd="${1:-help}"
[ $# -gt 0 ] && shift
rest=("$@")

scans="${SCANS:-$root/scans}"
out="${OUT:-$root/out}"
scans_regression="${SCANS_REGRESSION:-$root/scans-regression}"
projects="${PROJECTS:-$root/projects}"
golden="$root/golden"

# The version lives only in open_point_stitch/__init__.py.
version="$(sed -n 's/^__version__[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$root/open_point_stitch/__init__.py")"
[ -n "$version" ] || { echo "Could not find __version__ in open_point_stitch/__init__.py" >&2; exit 1; }
image="open-point-stitcher:$version"
bench_name="open-point-bench"

if [ "${OPENPOINT_NO_GPU:-0}" = "1" ]; then gpu_args=(); else gpu_args=(--gpus all); fi

build_time() { date -u +%Y-%m-%dT%H:%M:%SZ; }

do_build() {
  local t; t="$(build_time)"
  echo "Building $image (build time $t)..."
  docker build -t "$image" --build-arg "BUILD_TIME=$t" "$root"
  echo "Built $image"
}

has_flag() { local f="$1"; shift; for a in "$@"; do [ "$a" = "$f" ] && return 0; done; return 1; }
without_flag() { local f="$1"; shift; for a in "$@"; do [ "$a" != "$f" ] && printf '%s\n' "$a"; done; return 0; }

if [ "$cmd" = "build" ]; then do_build; exit 0; fi

if [ "$cmd" = "help" ]; then sed -n '2,14p' "$0"; exit 0; fi

# A missing mount source becomes an empty root-owned folder, which looks like "my scans vanished".
[ -d "$scans" ] || { echo "No scans folder at $scans. Create it and put your .pts exports in it, or set SCANS." >&2; exit 1; }
mkdir -p "$out" "$projects"

threads="${THREADS:-32}"   # 32 means "don't cap it"
mounts=(-v "$scans:/scans:ro" -v "$out:/out"
        -e "OMP_NUM_THREADS=$threads" -e "OPENBLAS_NUM_THREADS=$threads"
        -e "MKL_NUM_THREADS=$threads" -e "NUMEXPR_NUM_THREADS=$threads")
[ -d "$scans_regression" ] && mounts+=(-v "$scans_regression:/scans-regression:ro")

golden_mounts=()
[ -d "$golden" ] && golden_mounts=(-v "$golden:/golden:ro" -e OPENPOINT_GOLDEN_DIR=/golden)

# Dev mode: source bind-mounted read only, container open-point-bench-dev.
is_dev=0; has_flag -Dev "${rest[@]+"${rest[@]}"}" && is_dev=1
bench_container="$bench_name"; bench_extra=()
if [ "$is_dev" = "1" ]; then
  bench_container="$bench_name-dev"
  bench_extra=(-v "$root/open_point_stitch:/app/open_point_stitch:ro" -v "$root/tools:/app/tools:ro"
               -v "$root/tests:/app/tests:ro" -v "$root/placement_bench.html:/app/placement_bench.html:ro"
               -e OPENPOINT_DEV_MODE=1)
fi

run_bench() {
  # Loopback only, detached and named.
  docker run -d "${gpu_args[@]+"${gpu_args[@]}"}" --name "$bench_container" \
    -v "$scans:/scans:ro" -v "$out:/out" -v "$projects:/projects" \
    -p 127.0.0.1:8080:8080 "${golden_mounts[@]+"${golden_mounts[@]}"}" "${bench_extra[@]+"${bench_extra[@]}"}" "$image" bench
}

case "$cmd" in
  bench)
    if [ -n "$(docker ps -a -q -f "name=^$bench_container\$")" ]; then
      echo "A container named $bench_container already exists. Use './stitch.sh bench-update' or './stitch.sh bench-stop' first." >&2
      exit 1
    fi
    echo "OpenPointStitcher ($version): http://localhost:8080"
    run_bench
    echo "running as '$bench_container'. './stitch.sh bench-logs' to watch it, './stitch.sh bench-stop' to stop it."
    exit 0 ;;
  bench-stop)
    docker rm -f "$bench_container" >/dev/null 2>&1 || true
    echo "stopped and removed '$bench_container' (if it existed)."
    exit 0 ;;
  bench-logs)
    exec docker logs -f "$bench_container" ;;
  bench-update)
    do_build
    docker rm -f "$bench_container" >/dev/null 2>&1 || true
    # The bench's name before 0.50.0; remove it so the two do not fight over port 8080.
    docker rm -f "${bench_container/open-point-bench/nomad-bench}" >/dev/null 2>&1 || true
    run_bench
    echo "replaced '$bench_container': now running $version. Reload the browser tab to pick it up."
    exit 0 ;;
esac

# QA writes to out-qa and projects-qa, never the real ones; venues are mounted read only.
qa_mounts() {
  local with_venues="$1"
  local out_qa="${OUT_QA:-$root/out-qa}" projects_qa="${PROJECTS_QA:-$root/projects-qa}"
  mkdir -p "$out_qa" "$projects_qa"
  qa_m=(-v "$scans:/scans:ro" -v "$out_qa:/out" -v "$projects_qa:/projects"
        -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1)
  [ -d "$scans_regression" ] && qa_m+=(-v "$scans_regression:/scans-regression:ro")
  [ -d "$golden" ] && qa_m+=(-v "$golden:/golden:ro" -e OPENPOINT_GOLDEN_DIR=/golden)
  [ "$with_venues" = "1" ] && qa_m+=(-v "$projects:/venues:ro")
  return 0
}

# Same hash as stitch.ps1's Get-ImageInputsHash, so both scripts share the cached QA images.
inputs_hash() {
  (cd "$root" && cat Dockerfile Dockerfile.qa requirements-lock.txt docker-entrypoint.sh) | sha256sum | cut -c1-12
}

case "$cmd" in
  qa-quick|qa-medium|qa-slow)
    speed="${cmd#qa-}"
    tag="inputs-$(inputs_hash)"
    qa_image="open-point-stitcher-qa:$tag"
    if [ -z "$(docker images -q "$qa_image")" ]; then
      echo "Image inputs changed (or first run): building $qa_image once..."
      docker build -t "open-point-stitcher:$tag" --build-arg "BUILD_TIME=$(build_time)" "$root"
      docker build -t "$qa_image" --build-arg "STITCH_VERSION=$tag" -f "$root/Dockerfile.qa" "$root"
    fi
    src=(-v "$root/open_point_stitch:/app/open_point_stitch:ro" -v "$root/tools:/app/tools:ro"
         -v "$root/tests:/app/tests:ro" -v "$root/placement_bench.html:/app/placement_bench.html:ro")
    mapfile -t qa_args < <(without_flag -Changed "${rest[@]+"${rest[@]}"}")
    qa_args=("--$speed" "${qa_args[@]+"${qa_args[@]}"}")
    if has_flag -Changed "${rest[@]+"${rest[@]}"}"; then
      changed="$( { git -C "$root" diff --name-only HEAD; git -C "$root" ls-files --others --exclude-standard; } | paste -sd, -)"
      qa_args+=(--touched "$changed")
    fi
    echo "QA $speed on the working tree ($qa_image)..."
    venues=0; [ "$speed" = "slow" ] && venues=1
    qa_mounts "$venues"
    exec docker run --rm "${gpu_args[@]+"${gpu_args[@]}"}" --shm-size=1g "${qa_m[@]}" "${src[@]}" "$qa_image" python tools/qa.py "${qa_args[@]}" ;;
  qa)
    # The developer's docs check runs first when present (it is not part of the public repository).
    if [ ! -f "$root/tools/check_docs.py" ]; then
      :
    elif command -v python3 >/dev/null 2>&1; then
      python3 "$root/tools/check_docs.py" || { echo "QA stopped before building: the docs are out of step with the code (listed above)."; exit 1; }
    else
      echo "NOTE: python3 is not on this machine's PATH, so the docs check was skipped (it is not part of this run)."
    fi
    do_build
    qa_image="open-point-stitcher-qa:$version"
    echo "Building $qa_image on top of $image..."
    docker build -t "$qa_image" --build-arg "STITCH_VERSION=$version" -f "$root/Dockerfile.qa" "$root"
    qa_mounts 1
    echo "Running qa: all three speeds on the image just built (scans and venues read only)..."
    set +e
    docker run --rm "${gpu_args[@]+"${gpu_args[@]}"}" --shm-size=1g "${qa_m[@]}" "$qa_image" qa "${rest[@]+"${rest[@]}"}"
    qa_exit=$?
    set -e
    exit "$qa_exit" ;;
esac

# Anything else (inspect, stitch, hunt, gpu-info, test...) goes straight to the image.
exec docker run --rm "${gpu_args[@]+"${gpu_args[@]}"}" "${mounts[@]}" "$image" "$cmd" "${rest[@]+"${rest[@]}"}"
