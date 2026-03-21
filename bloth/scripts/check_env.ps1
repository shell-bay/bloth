# =============================================================================
#  Bloth v1.0 — Environment Check  (Windows PowerShell)
# =============================================================================
#  Run this before installing to diagnose issues.
#
#  Usage:  .\scripts\check_env.ps1
# =============================================================================

$FailCount = 0
function Ok   { Write-Host "  ✅  $args" -ForegroundColor Green }
function Fail { Write-Host "  ❌  $args" -ForegroundColor Red; $script:FailCount++ }
function Warn { Write-Host "  ⚠   $args" -ForegroundColor Yellow }
function Sep  { Write-Host ("─" * 50) -ForegroundColor Gray }

Write-Host ""
Write-Host "  Bloth — Environment Check (Windows)" -ForegroundColor White
Sep

# ── Python ────────────────────────────────────────────────────────────────────
Write-Host "`nPython:" -ForegroundColor Cyan
$Python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $v = & $cmd --version 2>&1
        if ($v -match "Python (\d+)\.(\d+)") {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            if ($maj -ge 3 -and $min -ge 8) { $Python = $cmd; Ok "Python $maj.$min ($cmd)"; break }
            else { Fail "Python $maj.$min too old — need 3.8+" }
        }
    } catch {}
}
if (-not $Python) { Fail "Python not found — https://python.org" }

# ── CUDA / GPU ───────────────────────────────────────────────────────────────
Write-Host "`nGPU / CUDA:" -ForegroundColor Cyan
try {
    $smi = & nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $gpu  = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>&1) | Select-Object -First 1
        $vram = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>&1) | Select-Object -First 1
        $cuda = ($smi | Select-String "CUDA Version").ToString().Split()[-1]
        Ok "GPU: $gpu  ($vram)"
        Ok "CUDA: $cuda"
        $cudaMaj = [int]$cuda.Split(".")[0]
        if ($cudaMaj -ge 11) { Ok "CUDA version compatible" }
        else { Fail "CUDA $cuda too old (need 11.8+)" }
    }
} catch {
    Warn "nvidia-smi not found — GPU features unavailable (CPU-only mode)"
}

# ── Python packages ───────────────────────────────────────────────────────────
Write-Host "`nPython packages:" -ForegroundColor Cyan
foreach ($pkg in @("torch", "triton")) {
    $ver = & $Python -c "import $pkg; print($pkg.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) { Ok "${pkg}: $ver" }
    else                      { Fail "$pkg not installed (pip install $pkg)" }
}

$cudaOk = & $Python -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaOk -eq "True") { Ok "torch CUDA: available" }
else                     { Warn "torch CUDA: not available (CPU mode)" }

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Sep
if ($FailCount -eq 0) {
    Write-Host "  All checks passed — ready to install! ✅" -ForegroundColor Green
    Write-Host "  Run: .\install.ps1" -ForegroundColor Green
} else {
    Write-Host "  $FailCount issue(s) found — fix before installing." -ForegroundColor Red
}
Write-Host ""
