#!/usr/bin/env bash
# =============================================================================
#  Bloth v1.0 — Google Colab Quick Setup
# =============================================================================
#  This is the script behind the Colab one-liner:
#
#    !bash <(curl -s https://raw.githubusercontent.com/shell-bay/bloth/main/scripts/colab_setup.sh)
#
#  Or manually, paste these 3 cells into Colab:
#
#  Cell 1 — Install:
#    !git clone https://github.com/shell-bay/bloth.git && cd bloth && bash install.sh --colab
#
#  Cell 2 — Test:
#    !python tests/test_kernels.py
#
#  Cell 3 — Train:
#    from bloth import FastModel
#    model, tokenizer = FastModel.from_pretrained("unsloth/Llama-3.2-1B", load_in_4bit=True)
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}⚡ Bloth v1.0 — Colab Setup${NC}"
echo ""

# Check if already cloned
if [ ! -d "bloth" ]; then
  echo -e "${BLUE}Cloning repository...${NC}"
  git clone https://github.com/shell-bay/bloth.git
fi
cd bloth

echo -e "${BLUE}Installing Bloth...${NC}"
pip install -e . -q
pip install "transformers>=4.46.3" "trl>=0.11.4" peft bitsandbytes datasets accelerate -q

echo ""
echo -e "${BLUE}Verifying installation...${NC}"
python - << 'PYEOF'
import bloth, torch
print(f"  ✅ Bloth v{bloth.__version__}")
print(f"  ✅ PyTorch {torch.__version__}")
if torch.cuda.is_available():
    info = bloth.get_device_info()
    print(f"  ✅ GPU: {info['name']}  ({info['vram_gb']} GB VRAM)")
    print(f"  ✅ Architecture: {info['arch']}")
else:
    print("  ⚠  No GPU (CPU mode)")
PYEOF

echo ""
echo -e "${GREEN}${BOLD}✅ Bloth is ready! Next steps:${NC}"
echo -e "  Run tests:    ${GREEN}python tests/test_kernels.py${NC}"
echo -e "  Train model:  ${GREEN}python examples/train_llm.py${NC}"
echo ""
