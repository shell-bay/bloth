# =============================================================================
#  Bloth v1.0 — Test Runner  (Windows PowerShell)
# =============================================================================
#
#  Usage:
#    .\scripts\run_tests.ps1              # run all tests
#    .\scripts\run_tests.ps1 -Quick       # syntax + import only
#    .\scripts\run_tests.ps1 -Benchmark   # run benchmarks too
#    .\scripts\run_tests.ps1 -Verbose     # verbose output
# =============================================================================

param(
    [switch]$Quick,
    [switch]$Benchmark,
    [switch]$Verbose
)

function Write-Sep { Write-Host ("─" * 60) -ForegroundColor Gray }
function Write-Info { Write-Host "[Bloth] $args" -ForegroundColor Cyan }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$Python = "python"

Write-Sep
Write-Host "  ⚡ Bloth — Test Suite (Windows)" -ForegroundColor White
Write-Sep

if ($Quick) {
    Write-Info "Quick check: syntax + imports..."
    $QuickScript = @"
import ast, glob, sys
files = glob.glob("**/*.py", recursive=True)
errors = [f for f in files if not (lambda f: (ast.parse(open(f).read()), True)[1])(f)]
print(f"  OK {len(files)} files checked")
import bloth
print(f"  OK bloth v{bloth.__version__}")
from bloth.kernels import bloth_rms_norm
print("  OK bloth_rms_norm exported")
print("  Quick check passed!")
"@
    $tmp = [System.IO.Path]::GetTempFileName() + ".py"
    $QuickScript | Out-File $tmp -Encoding utf8
    & $Python $tmp
    Remove-Item $tmp
    exit 0
}

if ($Verbose) {
    & $Python -m pytest tests/ -v --tb=short
} else {
    & $Python tests/test_kernels.py
}

if ($Benchmark) {
    Write-Sep
    Write-Info "Running benchmarks..."
    & $Python examples/benchmark_kernels.py
}

Write-Sep
Write-Host "  All tests passed! ✅" -ForegroundColor Green
