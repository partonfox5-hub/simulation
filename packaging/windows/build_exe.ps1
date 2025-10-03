param(
    [string]$Python = "py"
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Message,
        [scriptblock]$Action
    )
    Write-Host "[+] $Message"
    & $Action
}

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python interpreter '$Python' not found. Pass -Python <path> to override."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$venvPath = Join-Path $repoRoot ".venv-build"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$distDir = Join-Path $repoRoot "dist"
$buildDir = Join-Path $repoRoot "build"
$packageName = "AuroraEvolutions"
$packageFolder = Join-Path $distDir "${packageName}-win64"
$zipPath = Join-Path $distDir "${packageName}-win64.zip"

Invoke-Step "Creating build virtual environment" {
    if (-not (Test-Path $venvPython)) {
        & $Python -m venv $venvPath
    }
}

Invoke-Step "Upgrading pip" {
    & $venvPython -m pip install --upgrade pip
}

Invoke-Step "Installing dependencies" {
    & $venvPython -m pip install -r (Join-Path $repoRoot "requirements.txt")
    & $venvPython -m pip install pyinstaller
}

Invoke-Step "Cleaning previous build output" {
    Remove-Item $distDir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
}

Invoke-Step "Running PyInstaller" {
    & $venvPython -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name $packageName `
        (Join-Path $repoRoot "src\main.py")
}

$exePath = Join-Path $distDir "${packageName}.exe"
if (-not (Test-Path $exePath)) {
    throw "Expected executable not found at $exePath"
}

Invoke-Step "Preparing portable zip" {
    New-Item -ItemType Directory -Force -Path $packageFolder | Out-Null
    Copy-Item $exePath -Destination (Join-Path $packageFolder "${packageName}.exe") -Force
    Copy-Item (Join-Path $repoRoot "readme.md") -Destination (Join-Path $packageFolder "AuroraEvolutions-README.txt") -Force
    if (Test-Path (Join-Path $repoRoot "assets")) {
        Copy-Item (Join-Path $repoRoot "assets") -Destination $packageFolder -Recurse -Force
    }
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path (Join-Path $packageFolder '*') -DestinationPath $zipPath -Force
}

Write-Host "Build complete. Upload 'dist/${packageName}-win64.zip' to a GitHub release for a double-clickable executable."
