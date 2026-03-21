#!/usr/bin/env bash
# =============================================================================
#  Bloth v1.0 — Test Runner  (Linux / macOS / WSL / Colab)
# =============================================================================
#
#  Usage:
#    bash scripts/run_tests.sh              # run all tests
#    bash scripts/run_tests.sh --quick      # import + syntax only
#    bash scripts/run_tests.sh --benchmark  # run benchmarks too
#    bash scripts/run_tests.sh --verbose    # verbose pytest output
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; NC='\033[0m'

info() { echo -e "\033[0;34m[Bloth]\033[0m $*"; }
sep()  { echo -e "${BOLD}$(printf '─%.0s' {1..60})${NC}"; }

QUICK=0; BENCH=0; VERBOSE=0
for arg in "$@"; do
  case $arg in
    --quick)     QUICK=1 ;;
    --benchmark) BENCH=1 ;;
    --verbose)   VERBOSE=1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

PYTHON=$(command -v python3 || command -v python)

sep
echo -e "${BOLD}  ⚡ Bloth — Test Suite${NC}"
sep

# ── Quick mode: syntax + import only ─────────────────────────────────────────
if [[ $QUICK -eq 1 ]]; then
  info "Quick check: syntax + imports"
  $PYTHON - << 'PYEOF'
import ast, glob, sys, os
root = os.getcwd()
files = glob.glob("**/*.py", recursive=True)
errors = []
for f in files:
    try:
        ast.parse(open(f).read())
    except SyntaxError as e:
        errors.append(f"{f}: {e}")
if errors:
    for e in errors: print(f"  ❌ {e}")
    sys.exit(1)
print(f"  ✅ {len(files)} files syntax-valid")

import bloth
print(f"  ✅ import bloth v{bloth.__version__}")
from bloth.kernels import bloth_rms_norm
print(f"  ✅ bloth_rms_norm exported")
print("\n  Quick check passed ✅")
PYEOF
  exit 0
fi

# ── Full test suite ───────────────────────────────────────────────────────────
info "Running full test suite..."
if [[ $VERBOSE -eq 1 ]]; then
  $PYTHON -m pytest tests/ -v --tb=short
else
  $PYTHON tests/test_kernels.py
fi

# ── Optional benchmarks ───────────────────────────────────────────────────────
if [[ $BENCH -eq 1 ]]; then
  sep
  info "Running benchmarks..."
  $PYTHON examples/benchmark_kernels.py
fi

sep
echo -e "${GREEN}  All tests passed! 🚀${NC}"
