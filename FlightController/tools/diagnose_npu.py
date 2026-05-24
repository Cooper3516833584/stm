"""Diagnose NPU availability on STM32MP257.

Checks installed ONNX Runtime providers, NPU driver nodes, and recommends
the correct execution provider for the board.

Usage:
    PYTHONPATH=. python FlightController/tools/diagnose_npu.py
"""

import os
import sys


def _check_sysfs_nodes() -> dict[str, bool]:
    nodes = {
        "/sys/class/drm/card0": False,
        "/sys/class/drm/card1": False,
        "/dev/galcore": False,
        "/dev/dri/renderD128": False,
        "/sys/kernel/debug/gc": False,
        "/sys/devices/platform/npu": False,
        "/sys/devices/platform/vsi_npu": False,
        "/sys/class/misc/galcore": False,
    }
    for path in nodes:
        nodes[path] = os.path.exists(path)
    return nodes


def _check_stai_mpu() -> bool:
    try:
        import stai_mpu  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    print("=" * 60)
    print("  NPU / AI Accelerator Diagnostic")
    print("=" * 60)

    # 1. Check sysfs nodes
    print("\n[1] Kernel device nodes:")
    nodes = _check_sysfs_nodes()
    found_any = False
    for path, exists in sorted(nodes.items()):
        if exists:
            print(f"     [EXISTS]  {path}")
            found_any = True
    if not found_any:
        print("     (none of the expected NPU nodes found)")

    # 2. Check stai_mpu
    print("\n[2] stai_mpu (ST AI package):")
    if _check_stai_mpu():
        print("     [FOUND]  stai_mpu is installed")
    else:
        print("     [MISSING] stai_mpu not installed")

    # 3. Check ONNX Runtime providers
    print("\n[3] ONNX Runtime available providers:")
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        for p in available:
            print(f"     {p}")
        print()
        print("   Recommended for this board (in priority order):")
        npu_like = [p for p in available if any(
            kw in p.lower() for kw in ["vsi", "npu", "xnnpack", "acl", "arm", "nnapi"]
        )]
        if npu_like:
            for p in npu_like:
                print(f"     -> {p}")
        else:
            print("     (no NPU/accelerator provider found — CPU only)")
        print()
        print(f"   Current default provider in road_perception.py: CPUExecutionProvider")
        if "XnnpackExecutionProvider" in available:
            print("   Note: XnnpackExecutionProvider uses CPU SIMD (not NPU)")
            print("         but is typically 2-4x faster than CPUExecutionProvider on ARM")
    except ImportError:
        print("     onnxruntime not installed")
        return 1

    # 4. Check if model is ONNX opset compatible with NPU
    model_path = "FlightController/Solutions/model/road_yolo11n_seg.onnx"
    print(f"\n[4] Model: {model_path}")
    if os.path.isfile(model_path):
        try:
            import onnx
            m = onnx.load(model_path)
            print(f"     IR version : {m.ir_version}")
            print(f"     opset      : {m.opset_import[0].domain} v{m.opset_import[0].version}")
            print(f"     producer   : {m.producer_name}")
            print(f"     size       : {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
            # Check if model needs simplification for NPU
            for node in m.graph.node:
                if node.op_type in ("Resize", "NonMaxSuppression"):
                    print(f"     WARNING: op '{node.op_type}' may not be supported by NPU")
        except ImportError:
            print("     (onnx package not installed, cannot inspect model)")
    else:
        print("     [MISSING] Model file not found")

    # 5. Summary
    print("\n[5] Recommendation:")
    has_stai = _check_stai_mpu()
    try:
        providers = __import__("onnxruntime").get_available_providers()
    except ImportError:
        providers = []

    npu_provider = None
    for p in providers:
        if "vsi" in p.lower() or "npu" in p.lower():
            npu_provider = p
            break

    if has_stai:
        print("     Use stai_mpu API (ST official NPU path)")
        print("     Model needs to be compiled with stai_mpu tools first")
    elif npu_provider:
        print(f"     Use ONNX Runtime with '{npu_provider}'")
    elif "XnnpackExecutionProvider" in providers:
        print("     Use XnnpackExecutionProvider (ARM SIMD, ~2-4x CPU speedup)")
    else:
        print("     CPUExecutionProvider only — visual pipeline ~0.6 FPS")
        print("     Consider:")
        print("       1. Install onnxruntime with NPU support")
        print("       2. Or use stai_mpu + model compilation tools")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
