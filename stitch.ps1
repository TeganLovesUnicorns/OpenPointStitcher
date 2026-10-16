<#
  Windows convenience wrapper. Saves remembering the docker flags.

    .\stitch.ps1 build
    .\stitch.ps1 inspect
    .\stitch.ps1 test
    .\stitch.ps1 qa               all three speeds on a fresh image: the one to run before bench-update
    .\stitch.ps1 qa-quick         pure checks on the working tree, a minute or two (-Changed: only what changed reaches)
    .\stitch.ps1 qa-medium        the browser suite on the working tree
    .\stitch.ps1 qa-slow          the real scans and both venue canaries on the working tree
    .\stitch.ps1 stitch
    .\stitch.ps1 hunt --runs 40
    .\stitch.ps1 bench
    .\stitch.ps1 bench -Dev
    .\stitch.ps1 bench-update
    .\stitch.ps1 bench-stop
    .\stitch.ps1 bench-logs

  Reads .\scans and writes .\out unless you set $env:SCANS / $env:OUT.
#>
param(
  [Parameter(Position = 0)][string]$Command = 'help',
  [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$root  = $PSScriptRoot
$scans = if ($env:SCANS) { $env:SCANS } else { Join-Path $root 'scans' }
$out   = if ($env:OUT)   { $env:OUT }   else { Join-Path $root 'out' }
$scansRegression = if ($env:SCANS_REGRESSION) { $env:SCANS_REGRESSION } else { Join-Path $root 'scans-regression' }
# Project folders for bench: read-write, unlike $scans.
$projects = if ($env:PROJECTS) { $env:PROJECTS } else { Join-Path $root 'projects' }

# The version lives only in open_point_stitch/__init__.py; the image tag, badge and output all read it.
$initPy = Get-Content (Join-Path $root 'open_point_stitch\__init__.py') -Raw
$versionMatch = [regex]::Match($initPy, '__version__\s*=\s*"([^"]+)"')
if (-not $versionMatch.Success) { Write-Error "Could not find __version__ in open_point_stitch\__init__.py" }
$stitchVersion = $versionMatch.Groups[1].Value
$image = "open-point-stitcher:$stitchVersion"
$benchName = 'open-point-bench'

# BUILD_TIME is a build ARG so it changes on every build (a RUN in the Dockerfile would be cached).
function New-BuildTime { [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ') }

function Invoke-Build {
  $buildTime = New-BuildTime
  Write-Host "Building $image (build time $buildTime)..."
  docker build -t $image --build-arg "BUILD_TIME=$buildTime" $root
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host "Built $image"
}

if ($Command -eq 'build') {
  Invoke-Build
  exit $LASTEXITCODE
}

# Docker will happily create a missing mount source as an empty root-owned
# directory, which then looks like "my scans vanished". Better to say so.
if (-not (Test-Path $scans)) {
  Write-Error "No scans folder at $scans. Create it and put your .pts exports in it, or set `$env:SCANS."
}
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
if (-not (Test-Path $projects)) { New-Item -ItemType Directory -Path $projects | Out-Null }

# Every container asks for the GPU unless OPENPOINT_NO_GPU=1. Without NVIDIA container support, --gpus all fails
# loudly on purpose rather than silently running on the CPU.
$gpuArgs = if ($env:OPENPOINT_NO_GPU -eq '1') { @() } else { @('--gpus', 'all') }

# 32 means "don't cap it", not a real core count.
$threads = if ($env:THREADS) { $env:THREADS } else { '32' }
$mounts = @('-v', "${scans}:/scans:ro", '-v', "${out}:/out",
            '-e', "OMP_NUM_THREADS=$threads", '-e', "OPENBLAS_NUM_THREADS=$threads",
            '-e', "MKL_NUM_THREADS=$threads", '-e', "NUMEXPR_NUM_THREADS=$threads")
if (Test-Path $scansRegression) {
  $mounts += @('-v', "${scansRegression}:/scans-regression:ro")
}

# Dev mode bind-mounts the source read-only in a container named open-point-bench-dev, so code changes show on reload.
function Get-BenchArgs([bool]$dev) {
  if ($dev) {
    # Use string interpolation, not + concatenation, inside @(...): PowerShell splits the latter into separate args.
    $openPointMount = "$(Join-Path $root 'open_point_stitch'):/app/open_point_stitch:ro"
    $toolsMount = "$(Join-Path $root 'tools'):/app/tools:ro"
    $testsMount = "$(Join-Path $root 'tests'):/app/tests:ro"
    $benchHtmlMount = "$(Join-Path $root 'placement_bench.html'):/app/placement_bench.html:ro"
    return @{
      Name = "$benchName-dev"
      ExtraMounts = @(
        '-v', $openPointMount,
        '-v', $toolsMount,
        '-v', $testsMount,
        '-v', $benchHtmlMount,
        '-e', 'OPENPOINT_DEV_MODE=1'
      )
    }
  }
  return @{ Name = $benchName; ExtraMounts = @() }
}

$isDev = $Rest -contains '-Dev'

# Read only: saved versions the golden folder names are never tidied away.
$golden = Join-Path $root 'golden'
$goldenMounts = @()
if (Test-Path $golden) { $goldenMounts = @('-v', "${golden}:/golden:ro", '-e', 'OPENPOINT_GOLDEN_DIR=/golden') }

if ($Command -eq 'bench') {
  $b = Get-BenchArgs $isDev
  $benchContainerName = $b.Name
  $benchExtraMounts = $b.ExtraMounts
  $existing = docker ps -a -q -f "name=^$benchContainerName`$"
  if ($existing) {
    Write-Error "A container named $benchContainerName already exists. Use '.\stitch.ps1 bench-update' to replace it, or '.\stitch.ps1 bench-stop' to remove it first."
  }
  Write-Host "OpenPointStitcher ($stitchVersion$(if($isDev){', dev mode'})): http://localhost:8080"
  # Published on 127.0.0.1 only, so nothing else on the network can reach it. Detached and named.
  docker run -d @gpuArgs --name $benchContainerName -v "${scans}:/scans:ro" -v "${out}:/out" -v "${projects}:/projects" `
    -p 127.0.0.1:8080:8080 @goldenMounts @benchExtraMounts $image bench
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host "running as '$benchContainerName'. '.\stitch.ps1 bench-logs' to watch it, '.\stitch.ps1 bench-stop' to stop it."
  exit 0
}

if ($Command -eq 'bench-stop') {
  $benchContainerName = (Get-BenchArgs $isDev).Name
  docker rm -f $benchContainerName 2>$null | Out-Null
  Write-Host "stopped and removed '$benchContainerName' (if it existed)."
  exit 0
}

if ($Command -eq 'bench-logs') {
  $benchContainerName = (Get-BenchArgs $isDev).Name
  docker logs -f $benchContainerName
  exit $LASTEXITCODE
}

if ($Command -eq 'bench-update') {
  # Rebuild and replace the running container (a build alone leaves it untouched).
  Invoke-Build
  $b = Get-BenchArgs $isDev
  $benchContainerName = $b.Name
  $benchExtraMounts = $b.ExtraMounts
  docker rm -f $benchContainerName 2>$null | Out-Null
  # The bench's name before 0.50.0; remove it so the two do not fight over port 8080.
  docker rm -f ($benchContainerName -replace '^open-point-bench', 'nomad-bench') 2>$null | Out-Null
  docker run -d @gpuArgs --name $benchContainerName -v "${scans}:/scans:ro" -v "${out}:/out" -v "${projects}:/projects" `
    -p 127.0.0.1:8080:8080 @goldenMounts @benchExtraMounts $image bench
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host "replaced '$benchContainerName': now running $stitchVersion. Reload the browser tab to pick it up."
  exit 0
}

# QA folders and mounts shared by every speed. QA writes to out-qa and projects-qa, never the real ones; venue
# projects are mounted read only for the canaries.
function Get-QaMounts([bool]$withVenues) {
  $outQa = if ($env:OUT_QA) { $env:OUT_QA } else { Join-Path $root 'out-qa' }
  $projectsQa = if ($env:PROJECTS_QA) { $env:PROJECTS_QA } else { Join-Path $root 'projects-qa' }
  foreach ($d in @($outQa, $projectsQa)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
  $m = @('-v', "${scans}:/scans:ro", '-v', "${outQa}:/out", '-v', "${projectsQa}:/projects",
         '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1', '-e', 'NUMEXPR_NUM_THREADS=1')
  if (Test-Path $scansRegression) { $m += @('-v', "${scansRegression}:/scans-regression:ro") }
  if (Test-Path $golden) { $m += @('-v', "${golden}:/golden:ro", '-e', 'OPENPOINT_GOLDEN_DIR=/golden') }
  if ($withVenues) { $m += @('-v', "${projects}:/venues:ro") }
  return ,$m
}

# What the images are built from. The same hash means the same images, so the speed commands reuse them.
function Get-ImageInputsHash {
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $bytes = @()
  foreach ($f in @('Dockerfile', 'Dockerfile.qa', 'requirements-lock.txt', 'docker-entrypoint.sh')) {
    $bytes += [System.IO.File]::ReadAllBytes((Join-Path $root $f))
  }
  return ([System.BitConverter]::ToString($sha.ComputeHash([byte[]]$bytes)) -replace '-', '').Substring(0, 12).ToLower()
}

if ($Command -in @('qa-quick', 'qa-medium', 'qa-slow')) {
  $speed = $Command.Substring(3)
  $tag = "inputs-$(Get-ImageInputsHash)"
  $qaImage = "open-point-stitcher-qa:$tag"
  # `docker images -q` prints nothing for a missing image; `inspect` writes to stderr, fatal under 'Stop' in 5.1
  if (-not (docker images -q $qaImage)) {
    Write-Host "Image inputs changed (or first run): building $qaImage once..."
    docker build -t "open-point-stitcher:$tag" --build-arg "BUILD_TIME=$(New-BuildTime)" $root
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    docker build -t $qaImage --build-arg "STITCH_VERSION=$tag" -f (Join-Path $root 'Dockerfile.qa') $root
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  # The working tree, read only, over the image's copy: no rebuild for a code change.
  $src = @('-v', "$(Join-Path $root 'open_point_stitch'):/app/open_point_stitch:ro", '-v', "$(Join-Path $root 'tools'):/app/tools:ro",
           '-v', "$(Join-Path $root 'tests'):/app/tests:ro",
           '-v', "$(Join-Path $root 'placement_bench.html'):/app/placement_bench.html:ro")
  $qaArgs = @("--$speed") + @($Rest | Where-Object { $_ -ne '-Changed' })
  if ($Rest -contains '-Changed') {
    $changed = @(git -C $root diff --name-only HEAD) + @(git -C $root ls-files --others --exclude-standard)
    $qaArgs += @('--touched', ($changed -join ','))
  }
  Write-Host "QA $speed on the working tree ($qaImage)..."
  $qaMounts = Get-QaMounts ($speed -eq 'slow')
  docker run --rm @gpuArgs --shm-size=1g @qaMounts @src $qaImage python tools/qa.py @qaArgs
  exit $LASTEXITCODE
}

if ($Command -eq 'qa') {
  # Build a fresh image and run every test layer on it.
  # The developer's docs check runs first when present (it is not part of the public repository).
  $checkDocs = Join-Path $root 'tools\check_docs.py'
  if (-not (Test-Path $checkDocs)) {
    # nothing to check
  } elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python $checkDocs
    if ($LASTEXITCODE -ne 0) {
      Write-Host "QA stopped before building: the docs are out of step with the code (listed above). Fix them and run qa again."
      exit $LASTEXITCODE
    }
  } else {
    Write-Host "NOTE: python is not on this machine's PATH, so the docs check was skipped (it is not part of this run)."
  }
  Invoke-Build
  $qaImage = "open-point-stitcher-qa:$stitchVersion"
  Write-Host "Building $qaImage on top of $image..."
  docker build -t $qaImage --build-arg "STITCH_VERSION=$stitchVersion" -f (Join-Path $root 'Dockerfile.qa') $root
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  $qaMounts = Get-QaMounts $true
  Write-Host "Running qa: all three speeds on the image just built (scans and venues read only)..."
  # --shm-size=1g: headless Chromium crashes on Docker's default 64 MB /dev/shm.
  docker run --rm @gpuArgs --shm-size=1g @qaMounts $qaImage qa @Rest
  $qaExit = $LASTEXITCODE

  $toolSize = docker image inspect $image --format '{{.Size}}'
  $qaSize = docker image inspect $qaImage --format '{{.Size}}'
  $toGb = { param($bytes) [math]::Round([double]$bytes / 1GB, 2) }
  Write-Host ""
  Write-Host "image sizes: $image = $(& $toGb $toolSize) GB, $qaImage = $(& $toGb $qaSize) GB"
  exit $qaExit
}

# Same --gpus all as bench, so gpu-info and stitch --gpu can reach the card.
docker run --rm @gpuArgs @mounts $image $Command @Rest
exit $LASTEXITCODE
