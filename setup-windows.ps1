<#
    OpenPointStitcher - one-shot setup for Windows 11

    WHAT THIS DOES
      Checks what you already have, installs only what is missing, and tells you
      what it is doing at every step. Nothing is uninstalled. Nothing is
      overwritten. Run it twice and the second run should just say "already
      there" all the way down.

    HOW TO RUN IT
      1. Right-click the Start button, choose "Terminal (Admin)".
      2. Allow scripts for your account, once:
             Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
      3. cd into this folder, then:
             .\setup-windows.ps1

      If a step fails, it says which one and what to do, then stops.
#>

[CmdletBinding()]
param(
    # Skip the Docker image build at the end. Useful if you just want the tools.
    [switch]$SkipBuild,
    # Say yes to every install without being asked.
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$script:NeedsReboot = $false
$script:Problems    = @()

# ---------------------------------------------------------------------------
# small helpers, so the output reads like a person talking rather than a log
# ---------------------------------------------------------------------------
function Step($n, $text) { Write-Host "`n[$n] $text" -ForegroundColor Cyan }
function Good($text)     { Write-Host "      OK    $text" -ForegroundColor Green }
function Note($text)     { Write-Host "      ..    $text" -ForegroundColor Gray }
function Warn($text)     { Write-Host "      WARN  $text" -ForegroundColor Yellow
                           $script:Problems += $text }
function Fail($text)     { Write-Host "      STOP  $text" -ForegroundColor Red
                           $script:Problems += $text }

function Ask($question) {
    # -Yes turns every prompt into an automatic yes, for unattended runs.
    if ($Yes) { return $true }
    $a = Read-Host "      $question [Y/n]"
    return ($a -eq '' -or $a -match '^[Yy]')
}

function Have($cmd) {
    # Get-Command throws when the command does not exist, hence the try.
    try { $null = Get-Command $cmd -ErrorAction Stop; return $true }
    catch { return $false }
}

Write-Host "`n=====================================================" -ForegroundColor White
Write-Host " OpenPointStitcher - Windows setup" -ForegroundColor White
Write-Host "=====================================================" -ForegroundColor White

# ---------------------------------------------------------------------------
Step 1 "Are you running this as Administrator?"
# WSL and Docker Desktop both need admin.
$isAdmin = ([Security.Principal.WindowsPrincipal] `
            [Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) { Good "yes" }
else {
    Fail "no. Close this, right-click Start, pick 'Terminal (Admin)', and run it again."
    return
}

# ---------------------------------------------------------------------------
Step 2 "Windows version"
$os = Get-CimInstance Win32_OperatingSystem
$build = [int](Get-CimInstance Win32_OperatingSystem).BuildNumber
Note "$($os.Caption), build $build"
# Docker Desktop's WSL2 backend and winget both want a reasonably current
# Windows. 19041 is the WSL2 floor.
if ($build -lt 19041) {
    Fail "Windows is too old for WSL2 (need build 19041+). Update Windows first."
    return
}
Good "new enough"

# ---------------------------------------------------------------------------
Step 3 "Is hardware virtualization turned on?"
# This is the single most common reason WSL2 and Docker fail on a fresh
# machine, and the error message you get later is unhelpful, so check it here
# where we can say something useful.
$virt = (Get-CimInstance Win32_ComputerSystem).HypervisorPresent
if ($virt) { Good "yes" }
else {
    Warn "cannot confirm it. Open Task Manager, Performance tab, click CPU, and look for 'Virtualization'. If it says Disabled, reboot into your BIOS and turn on SVM Mode (AMD) or VT-x (Intel). Docker will not work without it."
}

# ---------------------------------------------------------------------------
Step 4 "winget, the Windows package installer"
if (Have 'winget') {
    Good "present ($(winget --version))"
    # A stale package index is a common cause of "no package found" later.
    Note "refreshing its package list"
    try { winget source update | Out-Null; Good "list refreshed" }
    catch { Warn "could not refresh the package list, carrying on anyway" }
} else {
    Fail "not found. winget ships inside 'App Installer'. Open the Microsoft Store, search App Installer, and install or update it, then run this script again."
    return
}

# ---------------------------------------------------------------------------
Step 5 "WSL2 (Docker needs it, and it is how the GPU reaches containers)"
$wslOk = $false
if (Have 'wsl') {
    try {
        # 'wsl --status' prints the default version among other things.
        $status = (wsl --status 2>&1 | Out-String)
        Note ($status.Trim() -replace "`r`n", "; ")
        $wslOk = $true
    } catch { Note "wsl.exe is present but did not answer, will try to update it" }
}
if (-not $wslOk) {
    if (Ask "Install WSL2 now?") {
        Note "running: wsl --install"
        wsl --install
        # wsl --install almost always wants a reboot before anything works.
        $script:NeedsReboot = $true
        Good "WSL install started. A reboot will be needed."
    } else { Warn "skipped. Docker will not work without it." }
} else {
    Note "updating the WSL kernel"
    try { wsl --update | Out-Null; Good "WSL is up to date" }
    catch { Warn "could not update WSL, carrying on" }
    # Make sure new distributions default to version 2, not 1. Docker needs 2.
    try { wsl --set-default-version 2 | Out-Null; Good "default version set to 2" }
    catch { Note "could not set the default version, usually harmless" }
}

# ---------------------------------------------------------------------------
Step 6 "Docker Desktop"
if (Have 'docker') {
    Good "already installed ($((docker --version)))"
} else {
    Note "not found"
    if (Ask "Install Docker Desktop?") {
        Note "running: winget install -e --id Docker.DockerDesktop"
        # --accept-*-agreements stops it stalling on a licence prompt.
        winget install -e --id Docker.DockerDesktop `
            --accept-package-agreements --accept-source-agreements
        $script:NeedsReboot = $true
        Good "installed. It needs a reboot, and you must launch Docker Desktop once by hand afterwards so it can finish setting itself up."
    } else { Warn "skipped. Nothing else here will run without it." }
}

# ---------------------------------------------------------------------------
Step 7 "Your graphics card"
# The GPU is optional. Everything works without it, just slower. So this
# reports rather than blocks.
if (Have 'nvidia-smi') {
    $smi = (nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | Out-String).Trim()
    Good $smi
    Note "For containers to see this card you need Docker Desktop on the WSL2 backend, which is its default."
} else {
    Warn "nvidia-smi not found. Either there is no NVIDIA card or the driver is old. The stitcher runs fine on CPU; the GPU is a speed-up, not a requirement."
}

# ---------------------------------------------------------------------------
Step 8 "Project folders"
$root = $PSScriptRoot
foreach ($d in @('scans', 'out', 'projects')) {
    $p = Join-Path $root $d
    if (Test-Path $p) { Note "$d already there" }
    else { New-Item -ItemType Directory -Path $p | Out-Null; Good "created $d" }
}
# .env carries your settings. Copy the example if you have not got one.
$envFile = Join-Path $root '.env'
if (-not (Test-Path $envFile)) {
    $example = Join-Path $root '.env.example'
    if (Test-Path $example) {
        Copy-Item $example $envFile
        Good "created .env from the example"
    }
}

# ---------------------------------------------------------------------------
Step 9 "Build the container"
if ($script:NeedsReboot) {
    Warn "skipping the build because a reboot is needed first."
} elseif ($SkipBuild) {
    Note "skipped, you asked me to"
} elseif (-not (Have 'docker')) {
    Warn "skipping the build, Docker is not installed"
} else {
    # Docker Desktop takes a while to be ready after login. Check before building.
    Note "checking Docker is actually running"
    $running = $false
    try { docker info 2>&1 | Out-Null; $running = ($LASTEXITCODE -eq 0) } catch { }
    if (-not $running) {
        Warn "Docker is installed but not running. Launch Docker Desktop, wait for the whale icon to stop animating, then run: .\stitch.ps1 build"
    } elseif (Ask "Build the stitcher image now? First build takes several minutes.") {
        # Build through stitch.ps1, the one place that knows the image tag and stamps the build time.
        Note "running: .\stitch.ps1 build"
        & (Join-Path $root 'stitch.ps1') build
        if ($LASTEXITCODE -eq 0) { Good "image built" }
        else { Fail "the build failed. The error above says why. A package name differing between Ubuntu and Debian is the most likely cause." }
    }
}

# ---------------------------------------------------------------------------
Write-Host "`n=====================================================" -ForegroundColor White
Write-Host " Summary" -ForegroundColor White
Write-Host "=====================================================" -ForegroundColor White

if ($script:NeedsReboot) {
    Write-Host "`n  REBOOT NEEDED. After it comes back up:" -ForegroundColor Yellow
    Write-Host "    1. Launch Docker Desktop once by hand and let it finish setting up."
    Write-Host "    2. Run this script again. It will skip everything already done."
}

if ($script:Problems.Count) {
    Write-Host "`n  Things worth a look:" -ForegroundColor Yellow
    $script:Problems | ForEach-Object { Write-Host "    - $_" }
} else {
    Write-Host "`n  Nothing went wrong." -ForegroundColor Green
}

Write-Host @"

  When everything above is done:

    1. Put your full-resolution .pts exports in .\scans (not thinned).

    2. Check it works:
         .\stitch.ps1 qa-quick

    3. Run it:
         .\stitch.ps1 bench           placement bench at http://localhost:8080
         .\stitch.ps1 stitch          command-line run; results appear in .\out

"@ -ForegroundColor White
