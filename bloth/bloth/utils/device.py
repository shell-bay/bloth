"""
Bloth Device Detection
Works for every NVIDIA GPU from Maxwell (GTX 9xx, SM52) to Blackwell (B200, SM100).
"""
import torch

_ARCH_MAP = {
    52:  "Maxwell (GTX 9xx)",
    53:  "Maxwell (Tegra)",
    60:  "Pascal (GTX 10xx)",
    61:  "Pascal (GTX 10xx)",
    62:  "Pascal (Tegra)",
    70:  "Volta (V100)",
    72:  "Volta (Xavier)",
    75:  "Turing (RTX 20xx / T4)",
    80:  "Ampere (A100 / RTX 30xx)",
    86:  "Ampere (RTX 30xx)",
    87:  "Ampere (Jetson Orin)",
    89:  "Ada Lovelace (RTX 40xx / L40)",
    90:  "Hopper (H100 / H200)",
    100: "Blackwell (B200 / GB200)",
}


def get_device_info(device_id: int = 0) -> dict:
    """Return a dict of GPU capabilities for the given device index."""
    if not torch.cuda.is_available():
        return {"available": False}

    p  = torch.cuda.get_device_properties(device_id)
    sm = p.major * 10 + p.minor

    return {
        "available":           True,
        "name":                p.name,
        "sm":                  sm,
        "arch":                _ARCH_MAP.get(sm, f"Unknown (SM{sm})"),
        "vram_gb":             round(p.total_memory / 1e9, 1),
        "sm_count":            p.multi_processor_count,
        "fp8_support":         sm >= 89,
        "tma_support":         sm >= 90,
        "wgmma_support":       sm >= 90,
        "flashattn3_support":  sm >= 90,
        "bf16_support":        sm >= 80,
    }


def print_device_info(device_id: int = 0):
    info = get_device_info(device_id)
    if not info["available"]:
        print("  No CUDA GPU detected.")
        return
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Bloth — GPU Report")
    print(sep)
    print(f"  GPU          : {info['name']}")
    print(f"  Architecture : {info['arch']}")
    print(f"  VRAM         : {info['vram_gb']} GB")
    print(f"  SM Count     : {info['sm_count']}")
    print(f"  BF16         : {'✅' if info['bf16_support']   else '❌ (use fp16)'}")
    print(f"  FP8          : {'✅' if info['fp8_support']    else '❌ (Hopper+ only)'}")
    print(f"  TMA/WGMMA    : {'✅' if info['tma_support']    else '❌ (Hopper+ only)'}")
    print(f"  FlashAttn-3  : {'✅' if info['flashattn3_support'] else '❌'}")
    print(f"{sep}\n")
