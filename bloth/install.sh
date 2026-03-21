#!/usr/bin/env bash
# =============================================================================
#  Bloth v1.0 — One-Command Installer  (Linux / macOS / WSL / Google Colab)
# =============================================================================
#
#  Usage:
#    bash install.sh                   # auto-detects your GPU and environment
#    bash install.sh --colab           # Google Colab optimised install
#    bash install.sh --cpu             # CPU-only (no CUDA)
#    bash install.sh --full            # include training tools (trl, datasets)
#    bash install.sh --dev             # include dev tools (pytest, black)
#    bash install.sh --help            # show this message
#
#  What it does:
#    1. Detects CUDA version and GPU architecture
#    2. Installs compatible PyTorch + Triton
#    3. Installs Bloth and all dependencies
#    4. Runs a quick sanity check
#    5. Prints a summary of what's enabled on your GPU
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[Bloth]${NC} $*"; }
success() { echo -e "${GREEN}[Bloth] ✅${NC} $*"; }
warn()    { echo -e "${YELLOW}[Bloth] ⚠${NC}  $*"; }
error()   { echo -e "${RED}[Bloth] ❌${NC} $*"; exit 1; }
sep()     { echo -e "${BOLD}$(printf '═%.0s' {1..60})${NC}"; }

# ── Parse arguments ───────────────────────────────────────────────────────────
MODE_COLAB=0
MODE_CPU=0
MODE_FULL=0
MODE_DEV=0

for arg in "$@"; do
  case $arg in
    --colab) MODE_COLAB=1 ;;
    --cpu)   MODE_CPU=1   ;;
    --full)  MODE_FULL=1  ;;
    --dev)   MODE_DEV=1   ;;
    --help)
      head -20 "$0" | grep '^#  ' | sed 's/^#  //'
      exit 0
      ;;
    *) warn "Unknown argument: $arg (ignored)" ;;
  esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
sep
echo -e "${BOLD}  ⚡ Bloth v1.0 — Installer${NC}"
sep

# ── Check Python ──────────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python || true)
[[ -z "$PYTHON" ]] && error "Python 3.8+ not found. Install from https://python.org"

PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python: $PY_VER  ($PYTHON)"
$PYTHON -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" \
  || error "Python 3.8+ required (found $PY_VER)"

PIP="$PYTHON -m pip"

# ── Detect environment ────────────────────────────────────────────────────────
IN_COLAB=0
$PYTHON -c "import google.colab" 2>/dev/null && IN_COLAB=1 || true
[[ $MODE_COLAB -eq 1 ]] && IN_COLAB=1

if [[ $IN_COLAB -eq 1 ]]; then
  info "Environment: Google Colab"
else
  info "Environment: Local / Server"
fi

# ── Detect CUDA ───────────────────────────────────────────────────────────────
CUDA_AVAILABLE=0
CUDA_VER=""
SM=""

if [[ $MODE_CPU -eq 0 ]]; then
  if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' || echo "")
    CUDA_AVAILABLE=1
    info "GPU: $GPU_NAME"
    info "CUDA: $CUDA_VER"

    # Get SM version
    SM=$($PYTHON -c "
import subprocess, re
out = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(out.replace('.','') if out else '75')
" 2>/dev/null || echo "75")
    info "SM version: SM$SM"
  else
    warn "nvidia-smi not found — installing CPU-only mode"
    CUDA_AVAILABLE=0
  fi
fi

# ── Select PyTorch CUDA channel ───────────────────────────────────────────────
if [[ $CUDA_AVAILABLE -eq 1 ]]; then
  CUDA_MAJOR="${CUDA_VER%%.*}"
  if   [[ $CUDA_MAJOR -ge 12 ]]; then TORCH_CHANNEL="cu121"
  elif [[ $CUDA_MAJOR -eq 11 ]]; then TORCH_CHANNEL="cu118"
  else                                  TORCH_CHANNEL="cpu"; CUDA_AVAILABLE=0
  fi
else
  TORCH_CHANNEL="cpu"
fi

# ── Upgrade pip ───────────────────────────────────────────────────────────────
info "Upgrading pip..."
$PIP install --upgrade pip --quiet

# ── Install PyTorch (skip if already has correct CUDA version) ────────────────
TORCH_INSTALLED=0
$PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
  && TORCH_INSTALLED=1 || true

if [[ $TORCH_INSTALLED -eq 0 ]]; then
  info "Installing PyTorch (channel: $TORCH_CHANNEL)..."
  if [[ $TORCH_CHANNEL == "cpu" ]]; then
    $PIP install torch --quiet
  else
    $PIP install torch \
      --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}" \
      --quiet
  fi
else
  TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)")
  info "PyTorch already installed: $TORCH_VER"
fi

# ── Install Triton ────────────────────────────────────────────────────────────
info "Installing Triton..."
$PIP install triton --quiet

# ── Install Bloth ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

info "Installing Bloth from $REPO_ROOT..."
$PIP install -e "$REPO_ROOT" --quiet

# ── Install pinned training libraries (fixes trl/transformers mismatch) ───────
info "Installing transformers + trl (pinned compatible versions)..."
$PIP install \
  "transformers>=4.46.3" \
  "trl>=0.11.4" \
  "peft>=0.7.0" \
  --quiet

# ── Optional: full training stack ────────────────────────────────────────────
if [[ $MODE_FULL -eq 1 || $IN_COLAB -eq 1 ]]; then
  info "Installing full training stack (bitsandbytes, datasets, accelerate)..."
  $PIP install bitsandbytes datasets accelerate --quiet
fi

# ── Optional: dev tools ───────────────────────────────────────────────────────
if [[ $MODE_DEV -eq 1 ]]; then
  info "Installing dev tools (pytest, black, isort)..."
  $PIP install pytest black isort --quiet
fi

# ── Quick sanity check ────────────────────────────────────────────────────────
sep
info "Running sanity check..."

$PYTHON - << 'PYCHECK'
import sys
try:
    import bloth
    print(f"  ✅ import bloth  v{bloth.__version__}")
    from bloth import FastModel
    print(f"  ✅ FastModel ready")
    from bloth.kernels import bloth_rms_norm
    print(f"  ✅ bloth_rms_norm kernel exported")
    import torch
    if torch.cuda.is_available():
        info = bloth.get_device_info()
        print(f"  ✅ GPU: {info['name']}  ({info['vram_gb']} GB)")
        features = []
        if info.get('bf16_support'):  features.append('BF16')
        if info.get('fp8_support'):   features.append('FP8')
        if info.get('tma_support'):   features.append('TMA+WGMMA')
        if features:
            print(f"  ✅ Hardware features: {', '.join(features)}")
    else:
        print(f"  ⚠  No CUDA GPU (CPU-only mode)")
    print()
    print("  Bloth is ready! 🚀")
except Exception as e:
    print(f"  ❌ {e}", file=sys.stderr)
    sys.exit(1)
PYCHECK

sep
success "Installation complete!"
echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo -e "  ${GREEN}python tests/test_kernels.py${NC}   — run full test suite"
echo -e "  ${GREEN}python examples/train_llm.py${NC}   — fine-tune LLaMA-3"
echo ""
