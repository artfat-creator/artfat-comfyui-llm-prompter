"""Auto-install a CUDA (GPU) build of llama-cpp-python matching THIS ComfyUI's torch
CUDA version, for the Artfat LLM Prompter node.

Why this exists:
  * `pip install llama-cpp-python` pulls the CPU-only PyPI wheel -> the LLM silently
    runs on CPU (slow). That is the #1 "why is it slow" support issue.
  * A CUDA wheel only works if its CUDA MAJOR matches torch's. `ggml-cuda.dll` links
    against `cudart64_<major>.dll`, which is shipped by torch (torch/lib). So a cu128
    wheel (needs cudart64_12) on a cu130 torch (ships cudart64_13) fails to load and
    silently falls back to CPU. We therefore read torch.version.cuda and install the
    JamePeng wheel whose CUDA major matches.

Runs automatically via ComfyUI-Manager right after the node is cloned. Also safe to run
by hand (with ComfyUI FULLY CLOSED, else the llama DLLs are locked):
    <python.exe> install.py

Skips (and leaves the load-time warning in llama_core to advise manually) when:
  non-Windows, no NVIDIA GPU, or torch is a CPU-only build / not installed.
"""

import subprocess
import sys

# --- JamePeng release this installer targets. Bump all three when upgrading. ------------
# Newest wheels: https://github.com/JamePeng/llama-cpp-python/releases
_LLAMA_VER = "0.3.48"
_REL_DATE = "20260821"
# CUDA major -> cu-tags JamePeng ships for it (ascending). Within a major we pick the
# highest tag whose minor is <= torch's minor; cudart64_<major> is forward-compatible
# across minors, so a same-major wheel loads against torch's runtime DLLs.
_CU_TAGS = {
    12: [124, 126, 128],
    13: [130, 131],
}
# ---------------------------------------------------------------------------------------


def _torch_cuda():
    """(major, minor) of torch's CUDA build, or None if torch is CPU-only / absent."""
    try:
        import torch
    except Exception:
        return None
    cuda = getattr(torch.version, "cuda", None)  # "13.0", or None on a CPU build
    if not cuda:
        return None
    try:
        major, minor = (int(x) for x in cuda.split(".")[:2])
        return major, minor
    except Exception:
        return None


def _pick_cu_tag(major, minor):
    """Best cu-tag: same major, highest tag whose minor <= torch minor (else lowest)."""
    tags = _CU_TAGS.get(major)
    if not tags:
        return None
    fitting = [t for t in tags if (t % 10) <= minor]  # 128 -> minor 8
    return fitting[-1] if fitting else tags[0]


def _has_nvidia():
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def main():
    if sys.platform != "win32":
        print("[llm-prompter] Non-Windows platform - GPU auto-install not wired up yet; "
              "see the load-time warning for the manual command.")
        return
    if not _has_nvidia():
        print("[llm-prompter] No NVIDIA GPU detected (nvidia-smi missing) - "
              "leaving llama-cpp-python as-is.")
        return

    tc = _torch_cuda()
    if tc is None:
        print("[llm-prompter] torch has no CUDA (CPU-only build / not installed) - "
              "skipping GPU install.")
        return
    major, minor = tc

    cu = _pick_cu_tag(major, minor)
    if cu is None:
        print(f"[llm-prompter] No known JamePeng wheel for CUDA {major}.{minor}. "
              "Install manually: https://github.com/JamePeng/llama-cpp-python/releases")
        return

    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    url = (f"https://github.com/JamePeng/llama-cpp-python/releases/download/"
           f"v{_LLAMA_VER}-cu{cu}-win-{_REL_DATE}/"
           f"llama_cpp_python-{_LLAMA_VER}+cu{cu}-{py}-{py}-win_amd64.whl")

    print(f"[llm-prompter] torch CUDA {major}.{minor} -> installing cu{cu} build for {py} ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", url],
            check=True,
        )
        print("[llm-prompter] Done. Restart ComfyUI; the LLM will run on the GPU.")
    except subprocess.CalledProcessError as e:
        print(f"[llm-prompter] CUDA install failed ({e}). Install manually - see "
              "https://github.com/JamePeng/llama-cpp-python/releases")


if __name__ == "__main__":
    main()
