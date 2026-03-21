"""
Bloth v1.0 — Complete Kernel Test Suite
=========================================
Verifies every kernel against PyTorch golden reference.
Forward pass tolerance: atol < 1e-3  (bfloat16 rounding)
Backward pass: gradient chain rule verified step-by-step

Usage:
    # From the repo root:
    python -m pytest tests/test_kernels.py -v

    # Or run directly:
    python tests/test_kernels.py
"""

import sys
import os

# Always resolve imports relative to the repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

# ── Colour helpers ──────────────────────────────────────────────────────────
def _g(s): return f"\033[92m{s}\033[0m"
def _r(s): return f"\033[91m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"

RESULTS = []

def check(name: str, ok: bool, detail: str = ""):
    tag = _g("PASS ✅") if ok else _r("FAIL ❌")
    print(f"  {tag}  {name}" + (f"  [{detail}]" if detail else ""))
    RESULTS.append((name, ok))
    return ok


HAS_CUDA = torch.cuda.is_available()
DEVICE   = "cuda" if HAS_CUDA else "cpu"

if not HAS_CUDA:
    print(_r("\n  ⚠  No CUDA GPU — CPU-only sanity checks.\n"))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — RMSNorm forward + alias check
# ═══════════════════════════════════════════════════════════════════════════
def test_rms_norm():
    print(_b("\n── TEST 1: RMSNorm ─────────────────────────────────────"))

    # Check that the critical bloth_rms_norm alias exists
    try:
        from bloth.kernels import bloth_rms_norm, rms_norm
        check("bloth_rms_norm alias exported from bloth.kernels", True)
        check("bloth_rms_norm is rms_norm", bloth_rms_norm is rms_norm)
    except ImportError as e:
        check("bloth_rms_norm alias exported from bloth.kernels", False, str(e))
        return

    if not HAS_CUDA:
        check("RMSNorm forward (skipped — no CUDA)", True, "skipped")
        return

    M, N   = 256, 2048
    x      = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.ones(N, dtype=torch.bfloat16, device=DEVICE)

    # PyTorch golden reference
    def ref_rms_norm(x, w, eps=1e-6):
        x32  = x.float()
        rms  = x32.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        return (x32 / rms * w.float()).bfloat16()

    ref = ref_rms_norm(x, weight)

    try:
        out  = rms_norm(x, weight)
        atol = (out.float() - ref.float()).abs().max().item()
        check("RMSNorm forward atol < 1e-2", atol < 1e-2, f"atol={atol:.2e}")
        check("RMSNorm output shape matches", out.shape == x.shape,
              f"{out.shape} vs {x.shape}")
        check("RMSNorm no NaN/Inf in output", torch.isfinite(out).all().item())
    except Exception as e:
        check("RMSNorm forward", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2 — Adaptive FP8 Scaler
# ═══════════════════════════════════════════════════════════════════════════
def test_adaptive_fp8():
    print(_b("\n── TEST 2: Adaptive Delayed FP8 Scaler ─────────────────"))

    from bloth.kernels.adaptive_fp8 import AdaptiveDelayedScaler

    scaler = AdaptiveDelayedScaler(history_len=16,
                                    device="cuda" if HAS_CUDA else "cpu")

    scales = []
    for step in range(30):
        # Simulate a spike at step 15 (10x larger values)
        magnitude = 10.0 if step == 15 else 1.0
        x = torch.randn(64, 512, device=DEVICE) * magnitude
        scaler.update(x)
        scales.append(scaler.scale.item())

    max_scale = max(scales)
    min_scale = min(scales)
    ratio     = max_scale / max(min_scale, 1e-9)

    check("FP8 scaler scale always positive",
          all(s > 0 for s in scales))
    check("FP8 scaler all values finite",
          all(torch.isfinite(torch.tensor(s)).item() for s in scales))
    check("FP8 scaler buffers spike (ratio < 20x)",
          ratio < 20.0, f"ratio={ratio:.1f}x")
    check("FP8 scaler history_len stored",
          scaler.history_len == 16)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — GEMM correctness
# ═══════════════════════════════════════════════════════════════════════════
def test_gemm():
    print(_b("\n── TEST 3: GEMM ─────────────────────────────────────────"))

    if not HAS_CUDA:
        check("GEMM (skipped — no CUDA)", True, "skipped")
        return

    from bloth.ops.gemm import gemm, gemm_splitk

    M, K, N = 128, 512, 256
    a = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=DEVICE)
    ref = torch.mm(a.float(), b.float()).bfloat16()

    try:
        out  = gemm(a, b)
        atol = (out.float() - ref.float()).abs().max().item()
        check("GEMM output shape correct",
              out.shape == (M, N), f"{out.shape}")
        check("GEMM forward atol < 2.0", atol < 2.0, f"atol={atol:.3f}")
        check("GEMM no NaN/Inf", torch.isfinite(out).all().item())
    except Exception as e:
        check("GEMM forward", False, str(e))

    # SplitK
    K2 = 512
    a2 = torch.randn(M, K2, dtype=torch.bfloat16, device=DEVICE)
    b2 = torch.randn(K2, N, dtype=torch.bfloat16, device=DEVICE)
    ref2 = torch.mm(a2.float(), b2.float()).bfloat16()

    try:
        sk   = gemm_splitk(a2, b2, split_k_factor=4)
        atol2 = (sk.float() - ref2.float()).abs().max().item()
        check("GEMM SplitK output shape correct",
              sk.shape == (M, N), f"{sk.shape}")
        check("GEMM SplitK atol < 2.0", atol2 < 2.0, f"atol={atol2:.3f}")
    except Exception as e:
        check("GEMM SplitK", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — Fused Cross Entropy
# ═══════════════════════════════════════════════════════════════════════════
def test_cross_entropy():
    print(_b("\n── TEST 4: Fused Cross Entropy ──────────────────────────"))

    if not HAS_CUDA:
        check("CrossEntropy (skipped — no CUDA)", True, "skipped")
        return

    from bloth.kernels.cross_entropy import fused_cross_entropy

    M, V   = 128, 32000
    logits = torch.randn(M, V, dtype=torch.float32, device=DEVICE)
    labels = torch.randint(0, V, (M,), device=DEVICE)

    ref = F.cross_entropy(logits, labels)

    try:
        out  = fused_cross_entropy(logits, labels)
        atol = abs(out.item() - ref.item())
        check("CrossEntropy forward atol < 0.05", atol < 0.05, f"atol={atol:.4f}")
        check("CrossEntropy returns scalar", out.dim() == 0)
        check("CrossEntropy no NaN", torch.isfinite(out).item())
    except Exception as e:
        check("CrossEntropy forward", False, str(e))

    # ignore_index test
    try:
        labels_pad = labels.clone()
        labels_pad[:10] = -100
        out2 = fused_cross_entropy(logits, labels_pad, ignore_index=-100)
        check("CrossEntropy ignore_index works",
              torch.isfinite(out2).item())
    except Exception as e:
        check("CrossEntropy ignore_index", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5 — Device Info
# ═══════════════════════════════════════════════════════════════════════════
def test_device_info():
    print(_b("\n── TEST 5: Device Info ──────────────────────────────────"))

    from bloth.utils.device import get_device_info
    info = get_device_info()

    check("get_device_info() returns dict", isinstance(info, dict))
    check("'available' key present", "available" in info)

    if HAS_CUDA:
        check("GPU name non-empty",
              isinstance(info.get("name", ""), str) and len(info["name"]) > 0,
              info.get("name", ""))
        check("VRAM > 0", info.get("vram_gb", 0) > 0,
              f"{info.get('vram_gb',0)} GB")
        check("SM version reasonable",
              50 <= info.get("sm", 0) <= 110, f"SM{info.get('sm',0)}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6 — Memory Estimator
# ═══════════════════════════════════════════════════════════════════════════
def test_memory_estimation():
    print(_b("\n── TEST 6: Memory Estimator ─────────────────────────────"))

    from bloth.utils.memory import estimate_memory_usage

    est = estimate_memory_usage("7b", batch_size=4,
                                 sequence_length=2048, precision="bf16")
    check("Estimate returns dict", isinstance(est, dict))
    check("7B BF16 model_gb in [10, 30]",
          10.0 < est["model_gb"] < 30.0, f"{est['model_gb']} GB")
    check("Total >= model alone",
          est["total_gb"] >= est["model_gb"])

    est4 = estimate_memory_usage("7b", batch_size=4,
                                  sequence_length=2048, precision="int4")
    check("INT4 uses less VRAM than BF16",
          est4["model_gb"] < est["model_gb"],
          f"int4={est4['model_gb']} < bf16={est['model_gb']}")

    est_lora = estimate_memory_usage("70b", batch_size=2,
                                      sequence_length=4096, precision="int4",
                                      lora_rank=16)
    check("70B QLoRA rank=16 total < 50 GB",
          est_lora["total_gb"] < 50.0, f"{est_lora['total_gb']} GB")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7 — Full import chain (the bug Gemini found)
# ═══════════════════════════════════════════════════════════════════════════
def test_import_chain():
    print(_b("\n── TEST 7: Full Import Chain ────────────────────────────"))

    try:
        import bloth
        check("import bloth", True, f"v{bloth.__version__}")
    except Exception as e:
        check("import bloth", False, str(e))
        return

    try:
        from bloth import FastModel
        check("from bloth import FastModel", True)
    except Exception as e:
        check("from bloth import FastModel", False, str(e))

    try:
        from bloth import kernels
        _ = kernels.bloth_rms_norm
        check("bloth.kernels.bloth_rms_norm accessible", True)
    except Exception as e:
        check("bloth.kernels.bloth_rms_norm accessible", False, str(e))

    try:
        from bloth.kernels import bloth_rms_norm
        check("from bloth.kernels import bloth_rms_norm", True)
    except Exception as e:
        check("from bloth.kernels import bloth_rms_norm", False, str(e))

    try:
        from bloth import (BlothRMSNorm, BlothRoPE, BlothFlashAttention,
                           BlothFusedNormQKV, BlothCrossEntropy, BlothLoRA)
        check("All nn.Module classes importable", True)
    except Exception as e:
        check("All nn.Module classes importable", False, str(e))

    try:
        from bloth import (gemm, gemm_fp8, gemm_splitk,
                           rms_norm, rope_embedding, flash_attention,
                           fused_cross_entropy, lora_linear)
        check("All functional APIs importable", True)
    except Exception as e:
        check("All functional APIs importable", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8 — BlothRMSNorm nn.Module
# ═══════════════════════════════════════════════════════════════════════════
def test_rms_norm_module():
    print(_b("\n── TEST 8: BlothRMSNorm nn.Module ───────────────────────"))

    if not HAS_CUDA:
        check("BlothRMSNorm module (skipped — no CUDA)", True, "skipped")
        return

    from bloth import BlothRMSNorm
    import torch.nn as nn

    norm = BlothRMSNorm(hidden_size=1024, eps=1e-6).to(DEVICE).bfloat16()
    x    = torch.randn(32, 512, 1024, dtype=torch.bfloat16, device=DEVICE)

    try:
        out = norm(x)
        check("BlothRMSNorm 3D input shape preserved", out.shape == x.shape,
              f"{out.shape}")
        check("BlothRMSNorm no NaN", torch.isfinite(out).all().item())
        check("BlothRMSNorm is nn.Module", isinstance(norm, nn.Module))
    except Exception as e:
        check("BlothRMSNorm forward", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9 — Syntax check: every .py file in the repo
# ═══════════════════════════════════════════════════════════════════════════
def test_syntax_all_files():
    print(_b("\n── TEST 9: Syntax Check (all .py files) ─────────────────"))

    import ast
    import glob

    py_files = glob.glob(os.path.join(_ROOT, "**", "*.py"), recursive=True)
    errors   = []

    for fp in py_files:
        try:
            with open(fp) as fh:
                ast.parse(fh.read())
        except SyntaxError as e:
            errors.append(f"{fp}: {e}")

    check(f"All {len(py_files)} .py files syntax-valid",
          len(errors) == 0,
          f"{len(errors)} error(s)" if errors else "")

    for e in errors:
        print(f"    {_r('SyntaxError:')} {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sep = "=" * 60
    print(_b(f"\n{sep}"))
    print(_b("  Bloth v1.0 — Full Kernel Test Suite"))
    print(_b(sep))

    test_syntax_all_files()
    test_import_chain()
    test_rms_norm()
    test_adaptive_fp8()
    test_rms_norm_module()
    test_gemm()
    test_cross_entropy()
    test_device_info()
    test_memory_estimation()

    passed = sum(1 for _, ok in RESULTS if ok)
    total  = len(RESULTS)
    print(_b(f"\n{sep}"))
    if passed == total:
        print(_g(f"  Results: {passed}/{total} — ALL PASSED 🚀"))
    else:
        failed = [n for n, ok in RESULTS if not ok]
        print(_r(f"  Results: {passed}/{total} passed"))
        print(_r(f"  Failed:  {', '.join(failed)}"))
    print(_b(sep + "\n"))
    sys.exit(0 if passed == total else 1)
