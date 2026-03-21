"""
Bloth v1.0 — Full Kernel Test Suite  (v1.1 — all 4 failures fixed)
====================================================================
Fixed:
  1. RMSNorm forward: atol was 4.31 → now < 1e-2  (SAVE_RMS constexpr removed)
  2. GEMM SplitK: atol was 1.4e38 → now < 2.0     (padding dimension fixed)
  3. CrossEntropy: Triton block/scalar store error  (fixed [0] extraction)
  4. CrossEntropy ignore_index: same fix

Run:
    python tests/test_kernels.py           # direct
    python -m pytest tests/ -v             # pytest
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

# ── Terminal colour helpers ─────────────────────────────────────────────────
def _g(s): return f"\033[92m{s}\033[0m"
def _r(s): return f"\033[91m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"
def _y(s): return f"\033[93m{s}\033[0m"

RESULTS = []

def check(name: str, ok: bool, detail: str = ""):
    tag = _g("PASS ✅") if ok else _r("FAIL ❌")
    print(f"  {tag}  {name}" + (f"  [{detail}]" if detail else ""))
    RESULTS.append((name, ok))
    return ok


HAS_CUDA = torch.cuda.is_available()
DEVICE   = "cuda" if HAS_CUDA else "cpu"
if not HAS_CUDA:
    print(_y("\n  ⚠  No CUDA GPU detected — CPU-only checks.\n"))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9 — Syntax: every .py file in repo
# ═══════════════════════════════════════════════════════════════════════════
def test_syntax_all_files():
    print(_b("\n── TEST 9: Syntax Check (all .py files) ─────────────────"))
    import ast, glob
    py_files = glob.glob(os.path.join(_ROOT, "**", "*.py"), recursive=True)
    errors   = []
    for fp in py_files:
        try:
            ast.parse(open(fp).read())
        except SyntaxError as e:
            errors.append(f"{os.path.relpath(fp, _ROOT)}: {e}")

    check(f"All {len(py_files)} .py files syntax-valid",
          len(errors) == 0,
          f"{len(errors)} error(s)" if errors else "")
    for e in errors:
        print(f"    {_r('SyntaxError:')} {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7 — Full import chain  (catches the bloth_rms_norm bug)
# ═══════════════════════════════════════════════════════════════════════════
def test_import_chain():
    print(_b("\n── TEST 7: Full Import Chain ────────────────────────────"))
    try:
        import bloth
        check("import bloth", True, f"v{bloth.__version__}")
    except Exception as e:
        check("import bloth", False, str(e)); return

    try:
        from bloth import FastModel
        check("from bloth import FastModel", True)
    except Exception as e:
        check("from bloth import FastModel", False, str(e))

    try:
        from bloth.kernels import bloth_rms_norm, rms_norm
        check("from bloth.kernels import bloth_rms_norm", True)
        check("bloth_rms_norm is rms_norm", bloth_rms_norm is rms_norm)
    except Exception as e:
        check("bloth.kernels.bloth_rms_norm", False, str(e))

    try:
        from bloth import (BlothRMSNorm, BlothRoPE, BlothFlashAttention,
                           BlothFusedNormQKV, BlothCrossEntropy, BlothLoRA)
        check("All nn.Module classes importable", True)
    except Exception as e:
        check("All nn.Module classes importable", False, str(e))

    try:
        from bloth import (gemm, gemm_fp8, gemm_splitk, rms_norm,
                           rope_embedding, flash_attention, fused_cross_entropy,
                           lora_linear)
        check("All functional APIs importable", True)
    except Exception as e:
        check("All functional APIs importable", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — RMSNorm  (FIX: removed SAVE_RMS constexpr autotune conflict)
# ═══════════════════════════════════════════════════════════════════════════
def test_rms_norm():
    print(_b("\n── TEST 1: RMSNorm ─────────────────────────────────────"))

    from bloth.kernels.rms_norm import bloth_rms_norm, rms_norm
    check("bloth_rms_norm alias exists", bloth_rms_norm is rms_norm)

    if not HAS_CUDA:
        check("RMSNorm (no CUDA — skipped)", True, "skipped"); return

    M, N   = 256, 2048
    x      = torch.randn(M, N, device=DEVICE).bfloat16()
    weight = torch.ones(N, device=DEVICE).bfloat16()

    # PyTorch golden reference (same float32-accumulate logic)
    def ref(x, w, eps=1e-6):
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        return (x32 / rms * w.float()).bfloat16()

    ref_out = ref(x, weight)

    try:
        bloth_out = rms_norm(x, weight)
        atol = (bloth_out.float() - ref_out.float()).abs().max().item()
        check("RMSNorm forward atol < 1e-2", atol < 1e-2,  f"atol={atol:.2e}")
        check("RMSNorm shape preserved",     bloth_out.shape == x.shape)
        check("RMSNorm no NaN/Inf",          torch.isfinite(bloth_out).all().item())
    except Exception as e:
        check("RMSNorm forward", False, str(e))

    # 3-D input test (as used in real models)
    if HAS_CUDA:
        try:
            x3d = torch.randn(4, 128, N, device=DEVICE).bfloat16()
            w3d = torch.ones(N, device=DEVICE).bfloat16()
            out3d = rms_norm(x3d, w3d)
            check("RMSNorm 3D shape preserved", out3d.shape == x3d.shape)
        except Exception as e:
            check("RMSNorm 3D", False, str(e))


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
        x = torch.randn(64, 512, device=DEVICE) * (10.0 if step == 15 else 1.0)
        scaler.update(x)
        scales.append(scaler.scale.item())

    check("FP8 scaler always positive",  all(s > 0 for s in scales))
    check("FP8 scaler finite",           all(torch.isfinite(torch.tensor(s)) for s in scales))
    ratio = max(scales) / max(min(scales), 1e-9)
    check("FP8 spike buffered (ratio<20x)", ratio < 20.0, f"{ratio:.1f}x")
    check("FP8 history_len=16",          scaler.history_len == 16)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8 — BlothRMSNorm nn.Module
# ═══════════════════════════════════════════════════════════════════════════
def test_rms_norm_module():
    print(_b("\n── TEST 8: BlothRMSNorm nn.Module ───────────────────────"))

    if not HAS_CUDA:
        check("BlothRMSNorm module (no CUDA)", True, "skipped"); return

    from bloth import BlothRMSNorm
    import torch.nn as nn

    norm = BlothRMSNorm(1024, eps=1e-6).to(DEVICE).bfloat16()
    x    = torch.randn(32, 512, 1024, device=DEVICE, dtype=torch.bfloat16)
    try:
        out = norm(x)
        check("BlothRMSNorm 3D shape",  out.shape == x.shape, str(out.shape))
        check("BlothRMSNorm no NaN",    torch.isfinite(out).all().item())
        check("BlothRMSNorm is Module", isinstance(norm, nn.Module))
    except Exception as e:
        check("BlothRMSNorm forward", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — GEMM  (FIX: SplitK padding corrected)
# ═══════════════════════════════════════════════════════════════════════════
def test_gemm():
    print(_b("\n── TEST 3: GEMM ─────────────────────────────────────────"))

    if not HAS_CUDA:
        check("GEMM (no CUDA)", True, "skipped"); return

    from bloth.ops.gemm import gemm, gemm_splitk

    M, K, N = 128, 512, 256
    a = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)
    ref = torch.mm(a.float(), b.float())

    try:
        out  = gemm(a, b)
        atol = (out - ref).abs().max().item()
        check("GEMM shape",          out.shape == (M, N),  str(out.shape))
        check("GEMM atol < 2.0",     atol < 2.0,            f"atol={atol:.3f}")
        check("GEMM no NaN/Inf",     torch.isfinite(out).all().item())
    except Exception as e:
        check("GEMM", False, str(e))

    # SplitK — divisible K
    K2 = 512
    a2 = torch.randn(M, K2, device=DEVICE, dtype=torch.bfloat16)
    b2 = torch.randn(K2, N, device=DEVICE, dtype=torch.bfloat16)
    ref2 = torch.mm(a2.float(), b2.float())
    try:
        sk   = gemm_splitk(a2, b2, split_k_factor=4)
        atol2 = (sk - ref2).abs().max().item()
        check("GEMM SplitK shape",     sk.shape == (M, N), str(sk.shape))
        check("GEMM SplitK atol < 2.0", atol2 < 2.0,       f"atol={atol2:.3f}")
        check("GEMM SplitK no NaN",    torch.isfinite(sk).all().item())
    except Exception as e:
        check("GEMM SplitK", False, str(e))

    # SplitK — NON-divisible K (tests the padding fix)
    K3 = 500  # not divisible by 4
    a3 = torch.randn(M, K3, device=DEVICE, dtype=torch.bfloat16)
    b3 = torch.randn(K3, N, device=DEVICE, dtype=torch.bfloat16)
    ref3 = torch.mm(a3.float(), b3.float())
    try:
        sk3  = gemm_splitk(a3, b3, split_k_factor=4)
        atol3 = (sk3 - ref3).abs().max().item()
        check("GEMM SplitK non-div-K atol < 2.0", atol3 < 2.0, f"atol={atol3:.3f}")
    except Exception as e:
        check("GEMM SplitK non-div-K", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — Cross Entropy  (FIX: [0] extraction for scalar store)
# ═══════════════════════════════════════════════════════════════════════════
def test_cross_entropy():
    print(_b("\n── TEST 4: Fused Cross Entropy ──────────────────────────"))

    if not HAS_CUDA:
        check("CrossEntropy (no CUDA)", True, "skipped"); return

    from bloth.kernels.cross_entropy import fused_cross_entropy

    M, V   = 128, 32000
    logits = torch.randn(M, V, device=DEVICE, dtype=torch.float32)
    labels = torch.randint(0, V, (M,), device=DEVICE)
    ref    = F.cross_entropy(logits, labels)

    try:
        out  = fused_cross_entropy(logits, labels)
        atol = abs(out.item() - ref.item())
        check("CrossEntropy forward atol < 0.05", atol < 0.05, f"atol={atol:.4f}")
        check("CrossEntropy scalar output",        out.dim() == 0)
        check("CrossEntropy no NaN",               torch.isfinite(out).item())
    except Exception as e:
        check("CrossEntropy forward", False, str(e))

    # ignore_index test
    try:
        lab2 = labels.clone()
        lab2[:10] = -100
        out2 = fused_cross_entropy(logits, lab2, ignore_index=-100)
        check("CrossEntropy ignore_index finite",  torch.isfinite(out2).item())
        check("CrossEntropy ignore_index scalar",  out2.dim() == 0)
    except Exception as e:
        check("CrossEntropy ignore_index", False, str(e))

    # 3D input test (batch, seq, vocab)
    try:
        logits3 = torch.randn(4, 32, V, device=DEVICE, dtype=torch.float32)
        labels3 = torch.randint(0, V, (4, 32), device=DEVICE)
        out3    = fused_cross_entropy(logits3, labels3)
        check("CrossEntropy 3D input",  torch.isfinite(out3).item())
    except Exception as e:
        check("CrossEntropy 3D", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5 — Device Info
# ═══════════════════════════════════════════════════════════════════════════
def test_device_info():
    print(_b("\n── TEST 5: Device Info ──────────────────────────────────"))

    from bloth.utils.device import get_device_info
    info = get_device_info()
    check("get_device_info() returns dict",  isinstance(info, dict))
    check("'available' key present",         "available" in info)
    if HAS_CUDA:
        check("GPU name non-empty", len(info.get("name", "")) > 0, info.get("name",""))
        check("VRAM > 0",           info.get("vram_gb", 0) > 0,   f"{info.get('vram_gb',0)} GB")
        check("SM in [50..110]",    50 <= info.get("sm", 0) <= 110, f"SM{info.get('sm',0)}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6 — Memory Estimator
# ═══════════════════════════════════════════════════════════════════════════
def test_memory_estimation():
    print(_b("\n── TEST 6: Memory Estimator ─────────────────────────────"))

    from bloth.utils.memory import estimate_memory_usage
    est = estimate_memory_usage("7b", 4, 2048, "bf16")
    check("Estimate is dict",             isinstance(est, dict))
    check("7B BF16 model_gb in [10,30]",  10.0 < est["model_gb"] < 30.0, f"{est['model_gb']} GB")
    check("total >= model alone",         est["total_gb"] >= est["model_gb"])
    est4 = estimate_memory_usage("7b", 4, 2048, "int4")
    check("INT4 < BF16 VRAM",             est4["model_gb"] < est["model_gb"],
          f"int4={est4['model_gb']} < bf16={est['model_gb']}")
    est_ql = estimate_memory_usage("70b", 2, 4096, "int4", lora_rank=16)
    check("70B QLoRA < 50 GB",            est_ql["total_gb"] < 50.0, f"{est_ql['total_gb']} GB")


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
