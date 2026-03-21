#!/usr/bin/env bash
# =============================================================================
#  Bloth v1.0 — Environment Diagnostics
# =============================================================================
#  Run this BEFORE installing if you're getting errors.
#  It checks everything Bloth needs and prints exactly what's missing.
#
#  Usage:
#    bash scripts/check_env.sh
# =============================================================================

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "  ${GREEN}✅${NC}  $*"; }
fail() { echo -e "  ${RED}❌${NC}  $*"; FAIL_COUNT=$((FAIL_COUNT+1)); }
warn() { echo -e "  ${YELLOW}⚠${NC}   $*"; }
FAIL_COUNT=0

echo ""
echo -e "${BOLD}  Bloth — Environment Check${NC}"
echo -e "$(printf '─%.0s' {1..50})"

# ── Python ────────────────────────────────────────────────────────────────────
echo -e "\n${BLUE}Python:${NC}"
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
if [[ -z "$PYTHON" ]]; then
  fail "Python not found — install from https://python.org"
else
  PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
  PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
  PY_MINOR=$(echo $PY_VER | cut -d. -f2)
  if [[ $PY_MAJOR -ge 3 && $PY_MINOR -ge 8 ]]; then
    ok "Python $PY_VER  ($PYTHON)"
  else
    fail "Python $PY_VER too old — need 3.8+"
  fi
fi

# ── pip ───────────────────────────────────────────────────────────────────────
PIP_VER=$($PYTHON -m pip --version 2>&1 | awk '{print $2}')
ok "pip $PIP_VER"

# ── CUDA and GPU ──────────────────────────────────────────────────────────────
echo -e "\n${BLUE}GPU / CUDA:${NC}"
if command -v nvidia-smi &>/dev/null; then
  GPU=$( nvidia-smi --query-gpu=name        --format=csv,noheader 2>/dev/null | head -1)
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
  CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' 2>/dev/null)
  ok "nvidia-smi found"
  ok "GPU: $GPU  ($VRAM)"
  ok "CUDA Driver: $CUDA"

  CUDA_MAJOR="${CUDA%%.*}"
  if [[ $CUDA_MAJOR -ge 11 ]]; then
    ok "CUDA version compatible (need 11.8+)"
  else
    fail "CUDA $CUDA too old — need 11.8+"
  fi
else
  warn "nvidia-smi not found — GPU features unavailable"
  warn "Bloth will run in CPU-only mode (no kernel acceleration)"
fi

# ── Key Python packages ───────────────────────────────────────────────────────
echo -e "\n${BLUE}Python packages:${NC}"
check_pkg() {
  local pkg=$1; local min=$2
  $PYTHON -c "
import importlib, sys
m = importlib.import_module('$pkg')
v = getattr(m, '__version__', 'unknown')
print(v)
" 2>/dev/null && ok "$pkg: $($PYTHON -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo 'installed')" \
              || fail "$pkg: NOT installed  (pip install $pkg)"
}

check_pkg torch    "2.0.0"
check_pkg triton   "2.1.0"

# Check CUDA torch
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'no cuda'" 2>/dev/null \
  && ok "torch.cuda.is_available() = True" \
  || warn "torch.cuda.is_available() = False (CPU only)"

# ── Disk space ────────────────────────────────────────────────────────────────
echo -e "\n${BLUE}Storage:${NC}"
FREE_GB=$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}' 2>/dev/null || echo "?")
if [[ "$FREE_GB" != "?" && $FREE_GB -lt 10 ]]; then
  warn "Only ${FREE_GB}GB free — models need 10-80GB depending on size"
else
  ok "Disk: ${FREE_GB}GB free"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "$(printf '─%.0s' {1..50})"
if [[ $FAIL_COUNT -eq 0 ]]; then
  echo -e "  ${GREEN}${BOLD}All checks passed — ready to install Bloth! ✅${NC}"
  echo ""
  echo -e "  Run: ${GREEN}bash install.sh${NC}"
else
  echo -e "  ${RED}${BOLD}$FAIL_COUNT issue(s) found — fix them before installing.${NC}"
fi
echo ""
