"""GPU device detection — works Maxwell (SM52) through Blackwell (SM100)."""
import torch

ARCH_MAP = {
    52: "Maxwell (GTX 9xx)",
    60: "Pascal (GTX 10xx)",
    61: "Pascal (GTX 10xx)",
    70: "Volta (V100)",
    72: "Volta (Xavier)",
    75: "Turing (RTX 20xx)",
    80: "Ampere (A100 / RTX 30xx)",
    86: "Ampere (RTX 30xx)",
    87: "Ampere (Jetson)",
    89: "Ada Lovelace (RTX 40xx)",
    90: "Hopper (H100)",
    100: "Blackwell (B200)",
}

def get_device_info(device_id: int = 0) -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    p   = torch.cuda.get_device_properties(device_id)
    sm  = p.major * 10 + p.minor
    return {
        "available":    True,
        "name":         p.name,
        "sm":           sm,
        "arch":         ARCH_MAP.get(sm, f"Unknown SM{sm}"),
        "vram_gb":      round(p.total_memory / 1e9, 1),
        "sm_count":     p.multi_processor_count,
        "fp8_support":  sm >= 89,
        "tma_support":  sm >= 90,
        "wgmma_support": sm >= 90,
        "flashattn3_support": sm >= 90,
    }

def print_device_info(device_id: int = 0):
    info = get_device_info(device_id)
    if not info["available"]:
        print("No CUDA GPU found.")
        return
    print(f"\n{'='*50}")
    print(f"  Bloth Device Report")
    print(f"{'='*50}")
    print(f"  GPU:          {info['name']}")
    print(f"  Architecture: {info['arch']}")
    print(f"  VRAM:         {info['vram_gb']} GB")
    print(f"  SM Count:     {info['sm_count']}")
    print(f"  FP8:          {'✅' if info['fp8_support'] else '❌'}")
    print(f"  TMA:          {'✅' if info['tma_support'] else '❌'}")
    print(f"  WGMMA:        {'✅' if info['wgmma_support'] else '❌'}")
    print(f"{'='*50}\n")
