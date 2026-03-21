# =============================================================================
#  Bloth v1.0 — One-Command Installer  (Windows PowerShell)
# =============================================================================
#
#  Usage (open PowerShell as normal user):
#    .\install.ps1                  # auto-detects GPU and installs
#    .\install.ps1 -Full            # include training tools
#    .\install.ps1 -Dev             # include pytest, black
#    .\install.ps1 -CpuOnly         # CPU-only, no CUDA
#    .\install.ps1 -Help            # show this message
#
#  If you see "execution policy" error, run first:
#    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# =============================================================================

[CmdletBinding()]
param(
    [switch]$Full,
    [switch]$Dev,
    [switch]$CpuOnly,
    [switch]$Help
)

# ── Colour helpers ────────────────────────────────────────────────────────────
function Write-Info    { param($msg) Write-Host "[Bloth] $msg"      -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[Bloth] ✅ $msg"   -ForegroundColor Green }
function Write-Warn    { param($msg) Write-Host "[Bloth] ⚠  $msg"   -ForegroundColor Yellow }
function Write-Err     { param($msg) Write-Host "[Bloth] ❌ $msg"   -ForegroundColor Red; exit 1 }
function Write-Sep     { Write-Host ("═" * 60) -ForegroundColor White }

if ($Help) {
    Get-Content $MyInvocation.MyCommand.Path | Select-String "^#  " | ForEach-Object {
        $_.Line -replace "^#  ",""
    }
    exit 0
}

Write-Sep
Write-Host "  ⚡ Bloth v1.0 — Windows Installer" -ForegroundColor White -BackgroundColor DarkBlue
Write-Sep

# ── Check Python ──────────────────────────────────────────────────────────────
$Python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3") { $Python = $cmd; break }
    } catch {}
}
if (-not $Python) {
    Write-Err "Python 3.8+ not found. Download from https://python.org"
}

$PyVer = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Info "Python: $PyVer"

$VerOk = & $Python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"
if ($LASTEXITCODE -ne 0) { Write-Err "Python 3.8+ required (found $PyVer)" }

$Pip = "$Python -m pip"

# ── Detect CUDA and GPU ───────────────────────────────────────────────────────
$CudaAvailable = $false
$GpuName       = "None"
$CudaVer       = ""

if (-not $CpuOnly) {
    try {
        $nvOutput  = & nvidia-smi 2>&1 | Select-String "CUDA Version"
        $gpuOutput = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
        if ($nvOutput) {
            $CudaVer       = ($nvOutput -split "\s+")[-1]
            $GpuName       = $gpuOutput | Select-Object -First 1
            $CudaAvailable = $true
            Write-Info "GPU: $GpuName"
            Write-Info "CUDA: $CudaVer"
        }
    } catch {
        Write-Warn "nvidia-smi not found — switching to CPU-only mode"
    }
}

# ── Select PyTorch install channel ────────────────────────────────────────────
$TorchChannel = "cpu"
if ($CudaAvailable) {
    $CudaMajor = [int]($CudaVer.Split(".")[0])
    if     ($CudaMajor -ge 12) { $TorchChannel = "cu121" }
    elseif ($CudaMajor -eq 11) { $TorchChannel = "cu118" }
    else                        { Write-Warn "CUDA $CudaVer may not be supported. Using CPU."; $TorchChannel = "cpu" }
}
Write-Info "PyTorch channel: $TorchChannel"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
Write-Info "Upgrading pip..."
& $Python -m pip install --upgrade pip --quiet

# ── Check if torch already installed ─────────────────────────────────────────
$TorchInstalled = $false
try {
    & $Python -c "import torch; assert torch.cuda.is_available()" 2>&1 | Out-Null
    $TorchInstalled = ($LASTEXITCODE -eq 0)
} catch {}

if (-not $TorchInstalled) {
    Write-Info "Installing PyTorch..."
    if ($TorchChannel -eq "cpu") {
        & $Python -m pip install torch --quiet
    } else {
        & $Python -m pip install torch `
            --index-url "https://download.pytorch.org/whl/$TorchChannel" `
            --quiet
    }
} else {
    $TorchVer = & $Python -c "import torch; print(torch.__version__)"
    Write-Info "PyTorch already installed: $TorchVer"
}

# ── Install Triton ────────────────────────────────────────────────────────────
Write-Info "Installing Triton..."
# On Windows, use triton-windows (official Windows build)
try {
    & $Python -m pip install triton --quiet
} catch {
    Write-Warn "triton package failed, trying triton-windows..."
    & $Python -m pip install triton-windows --quiet
}

# ── Install Bloth ─────────────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Write-Info "Installing Bloth from $RepoRoot..."
& $Python -m pip install -e $RepoRoot --quiet

# ── Pinned training libraries (fixes trl/transformers version mismatch) ───────
Write-Info "Installing transformers + trl (pinned versions)..."
& $Python -m pip install `
    "transformers>=4.46.3" `
    "trl>=0.11.4" `
    "peft>=0.7.0" `
    --quiet

# ── Optional: full stack ──────────────────────────────────────────────────────
if ($Full) {
    Write-Info "Installing full stack (bitsandbytes, datasets, accelerate)..."
    & $Python -m pip install bitsandbytes datasets accelerate --quiet
}

# ── Optional: dev tools ───────────────────────────────────────────────────────
if ($Dev) {
    Write-Info "Installing dev tools (pytest, black, isort)..."
    & $Python -m pip install pytest black isort --quiet
}

# ── Sanity check ─────────────────────────────────────────────────────────────
Write-Sep
Write-Info "Running sanity check..."

$CheckScript = @"
import sys
try:
    import bloth
    print(f'  OK import bloth v{bloth.__version__}')
    from bloth import FastModel
    print('  OK FastModel ready')
    from bloth.kernels import bloth_rms_norm
    print('  OK bloth_rms_norm exported')
    import torch
    if torch.cuda.is_available():
        info = bloth.get_device_info()
        print(f'  OK GPU: {info[\"name\"]}  ({info[\"vram_gb\"]} GB)')
    else:
        print('  WARN No CUDA GPU (CPU mode)')
    print()
    print('  Bloth is ready!')
except Exception as e:
    print(f'  FAIL {e}', file=sys.stderr)
    sys.exit(1)
"@

$tmpFile = [System.IO.Path]::GetTempFileName() + ".py"
$CheckScript | Out-File -FilePath $tmpFile -Encoding utf8
& $Python $tmpFile
Remove-Item $tmpFile -ErrorAction SilentlyContinue

Write-Sep
Write-Success "Installation complete!"
Write-Host ""
Write-Host "  Quick start:" -ForegroundColor White
Write-Host "    python tests/test_kernels.py    # run full test suite" -ForegroundColor Green
Write-Host "    python examples/train_llm.py    # fine-tune LLaMA-3" -ForegroundColor Green
Write-Host ""
