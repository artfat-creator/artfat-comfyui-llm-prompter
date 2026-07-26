"""Resident llama.cpp engine for the Artfat LLM Prompter node.

A single process-wide model is kept in VRAM and reused across runs. It is only
reloaded when a load-relevant setting changes, so repeat generations are fast.
Adapted from lihaoyun6/ComfyUI-llama-cpp (MIT), extended with n_cpu_moe and
KV-cache quantization for low-VRAM / MoE setups.
"""

import gc
import os
import sys

import folder_paths
import comfy.model_management as mm

from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler,
    Llava16ChatHandler,
    MoondreamChatHandler,
    NanoLlavaChatHandler,
    Llama3VisionAlphaChatHandler,
    MiniCPMv26ChatHandler,
)

from .support.gguf_layers import get_layer_count

# GPU status, checked once at import. NOTE: llama_cpp.llama_supports_gpu_offload() is
# UNRELIABLE here — it returns False even on a working CUDA build (checked before backend
# init), so we DON'T use it. Instead we check the actual ggml-cuda.dll: whether it exists
# (CPU-only build if not), and whether its required CUDA major (from its cudart/cublas
# import) matches torch's CUDA major — a mismatch (e.g. cu128 wheel on cu130 torch) means
# ggml-cuda.dll can't load its runtime DLLs and silently falls back to CPU.
_JAMEPENG_VER, _JAMEPENG_DATE = "0.3.44", "20260721"
_CU_TAGS = {12: [124, 126, 128], 13: [130, 131]}


def _ggml_cuda_major(dll_path):
    """CUDA major ggml-cuda.dll needs (from its cudart64_X/cublas64_X import), or None."""
    try:
        import re
        import struct
        data = open(dll_path, "rb").read()
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        coff = e_lfanew + 4
        numsec = struct.unpack_from("<H", data, coff + 2)[0]
        optsz = struct.unpack_from("<H", data, coff + 16)[0]
        opt = coff + 20
        is64 = struct.unpack_from("<H", data, opt)[0] == 0x20B
        imp_rva = struct.unpack_from("<I", data, opt + (112 if is64 else 96) + 8)[0]
        secs = []
        sh = opt + optsz
        for i in range(numsec):
            o = sh + i * 40
            secs.append((struct.unpack_from("<I", data, o + 12)[0],
                         struct.unpack_from("<I", data, o + 16)[0],
                         struct.unpack_from("<I", data, o + 20)[0]))

        def rva2off(rva):
            for va, rawsz, rawptr in secs:
                if va <= rva < va + rawsz:
                    return rawptr + (rva - va)
            return None

        off = rva2off(imp_rva)
        while off:
            name_rva = struct.unpack_from("<I", data, off + 12)[0]
            if name_rva == 0:
                break
            no = rva2off(name_rva)
            name = data[no:data.index(b"\0", no)].decode(errors="ignore")
            m = re.match(r"(?:cublas|cudart)64_(\d+)\.dll", name, re.I)
            if m:
                return int(m.group(1))
            off += 20
    except Exception:
        return None
    return None


def _gpu_status():
    """('ok'|'cpu'|'mismatch'|'unknown', (need_major, torch_major))."""
    try:
        import llama_cpp
        cuda_dll = os.path.join(os.path.dirname(llama_cpp.__file__), "lib", "ggml-cuda.dll")
        if not os.path.exists(cuda_dll):
            return "cpu", None
        need = _ggml_cuda_major(cuda_dll)
        try:
            import torch
            tc = getattr(torch.version, "cuda", None)
            tmaj = int(tc.split(".")[0]) if tc else None
        except Exception:
            tmaj = None
        if need is None or tmaj is None:
            return "unknown", (need, tmaj)
        return ("ok" if need == tmaj else "mismatch"), (need, tmaj)
    except Exception:
        return "unknown", None


_GPU_STATUS, _GPU_DETAIL = _gpu_status()


def _cuda_wheel_hint():
    """Reinstall command for THIS Python + THIS torch CUDA version (mirrors install.py)."""
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cu = 128
    try:
        import torch
        tc = getattr(torch.version, "cuda", None)
        if tc:
            maj, mi = (int(x) for x in tc.split(".")[:2])
            tags = _CU_TAGS.get(maj, [])
            fitting = [t for t in tags if (t % 10) <= mi]
            cu = fitting[-1] if fitting else (tags[0] if tags else 128)
    except Exception:
        pass
    url = (f"https://github.com/JamePeng/llama-cpp-python/releases/download/"
           f"v{_JAMEPENG_VER}-cu{cu}-win-{_JAMEPENG_DATE}/"
           f"llama_cpp_python-{_JAMEPENG_VER}+cu{cu}-{tag}-{tag}-win_amd64.whl")
    return sys.executable, url


# Base handlers always present in the fork.
CHAT_HANDLERS = [
    "None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA",
    "llama3-Vision-Alpha", "MiniCPM-v2.6",
]

# ggml KV-cache data types (stable enum values from llama_cpp._ggml).
GGML_KV_TYPES = {"f16": 1, "q8_0": 8, "q4_0": 2}

_HANDLERS = {
    "LLaVA-1.5": Llava15ChatHandler,
    "LLaVA-1.6": Llava16ChatHandler,
    "Moondream2": MoondreamChatHandler,
    "nanoLLaVA": NanoLlavaChatHandler,
    "llama3-Vision-Alpha": Llama3VisionAlphaChatHandler,
    "MiniCPM-v2.6": MiniCPMv26ChatHandler,
    "None": None,
}


def _try_add(names, import_name):
    """Register optional, version-dependent chat handlers if the fork ships them."""
    try:
        module = __import__("llama_cpp.llama_chat_format", fromlist=[import_name])
        handler = getattr(module, import_name)
    except Exception:
        return
    for name in names:
        CHAT_HANDLERS.append(name)
        _HANDLERS[name] = handler


_try_add(["Gemma3"], "Gemma3ChatHandler")
_try_add(["Gemma4"], "Gemma4ChatHandler")
_try_add(["Qwen2.5-VL", "MinerU2.5-Pro"], "Qwen25VLChatHandler")
_try_add(["Qwen3-VL", "Qwen3-VL-Thinking"], "Qwen3VLChatHandler")
_try_add(["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking"], "Qwen35ChatHandler")
_try_add(["GLM-4.6V", "GLM-4.6V-Thinking"], "GLM46VChatHandler")
_try_add(["GLM-4.1V-Thinking"], "GLM41VChatHandler")
_try_add(["LFM2-VL"], "LFM2VLChatHandler")
_try_add(["LFM2.5-VL"], "LFM25VLChatHandler")
_try_add(["MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"], "MiniCPMv45ChatHandler")
_try_add(["MiniCPM-v4.6", "MiniCPM-v4.6-Thinking"], "MiniCPMv46ChatHandler")


def _resolve_gguf_path(filename):
    """Full path for a GGUF picked in the dropdown.

    The dropdown merges two ComfyUI categories — "LLM" (models/LLM) and "clip"
    (models/text_encoders) — and either may span several roots via extra_model_paths.yaml, so the
    old hardcoded models_dir/LLM/<name> was only right by luck. Ask folder_paths where the file
    actually is; fall back to the historical path if it can't resolve one.
    """
    for category in ("LLM", "clip"):
        try:
            full = folder_paths.get_full_path(category, filename)
        except Exception:
            full = None
        if full and os.path.exists(full):
            return full
    return os.path.join(folder_paths.models_dir, "LLM", filename)


class LLMEngine:
    """Process-wide singleton holding one resident llama.cpp model."""

    llm = None
    chat_handler = None
    current_config = None

    @classmethod
    def unload(cls):
        try:
            if cls.llm is not None:
                cls.llm.close()
        except Exception:
            pass
        try:
            if cls.chat_handler is not None:
                cls.chat_handler._exit_stack.close()
        except Exception:
            pass
        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None
        gc.collect()
        mm.soft_empty_cache()

    @classmethod
    def ensure_loaded(cls, config):
        """Load the model only if no model is resident or the config changed."""
        if cls.llm is not None and cls.current_config == config:
            return
        cls._load(config)

    @classmethod
    def reset_context(cls, chat_handler_name):
        """Clear KV/context between one-shot runs so history never leaks."""
        try:
            cls.llm.n_tokens = 0
            cls.llm._ctx.memory_clear(True)
            if getattr(cls.llm, "is_hybrid", False) and cls.llm._hybrid_cache_mgr is not None:
                cls.llm._hybrid_cache_mgr.clear()
        except Exception:
            pass

    @classmethod
    def _load(cls, config):
        cls.unload()
        # Evict ComfyUI's diffusion models from VRAM BEFORE llama.cpp allocates its GPU layers.
        # llama.cpp allocates outside ComfyUI's torch pool, so if the diffusion model is still
        # resident the LLM's layers spill into shared system RAM (Windows WDDM) and load + inference
        # crawl. Freeing here gives llama.cpp clean VRAM; the sampler reloads the diffusion model on
        # its next run (and the unload_all_models hook already frees the LLM in the other direction).
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        cls.current_config = dict(config)

        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler_name = config["chat_handler"]
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        n_cpu_moe = config["n_cpu_moe"]
        image_min_tokens = config["image_min_tokens"]
        image_max_tokens = config["image_max_tokens"]
        type_k = GGML_KV_TYPES.get(config.get("type_k", "f16"))
        type_v = GGML_KV_TYPES.get(config.get("type_v", "f16"))

        handler_cls = _HANDLERS.get(chat_handler_name)
        if chat_handler_name not in _HANDLERS:
            raise ValueError(f'Unknown chat_handler: "{chat_handler_name}"')

        model_path = _resolve_gguf_path(model)
        n_gpu_layers = -1

        if vram_limit != -1:
            layers = get_layer_count(model_path) or 32
            gguf_gb = os.path.getsize(model_path) * 1.55 / (1024 ** 3)
            layer_gb = gguf_gb / max(1, layers)

        if mmproj and mmproj != "None":
            mmproj_path = _resolve_gguf_path(mmproj)
            if chat_handler_name == "None":
                raise ValueError('"chat_handler" cannot be None when an mmproj is set.')

            if vram_limit != -1:
                mmproj_gb = os.path.getsize(mmproj_path) * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_gb) / layer_gb))

            print(f"[llm-prompter] Loading clip: {mmproj}")
            think = "Thinking" in chat_handler_name
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}
            if chat_handler_name in ("Qwen3-VL", "Qwen3-VL-Thinking"):
                kwargs["force_reasoning"] = think
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            elif chat_handler_name in ("MiniCPM-v4.5", "GLM-4.6V", "Qwen3.5"):
                kwargs["enable_thinking"] = think
            try:
                cls.chat_handler = handler_cls(**kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"{e}\nUpdate llama-cpp-python from "
                    "https://github.com/JamePeng/llama-cpp-python/releases"
                )
        else:
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / layer_gb))
            cls.chat_handler = handler_cls(verbose=False) if handler_cls is not None else None

        if n_gpu_layers != 0 and _GPU_STATUS in ("cpu", "mismatch"):
            exe, url = _cuda_wheel_hint()
            print("=" * 72)
            if _GPU_STATUS == "mismatch":
                need, tmaj = _GPU_DETAIL
                print(f"[llm-prompter] WARNING: llama-cpp-python is built for CUDA {need}.x, but "
                      f"your torch uses CUDA {tmaj}.x -> GPU offload FAILS (silent CPU fallback).")
            else:
                print("[llm-prompter] WARNING: llama-cpp-python is a CPU-ONLY build (no CUDA).")
            print("[llm-prompter] The LLM will run on CPU (slow). To fix, FULLY CLOSE ComfyUI, then run:")
            print(f'[llm-prompter]   "{exe}" -m pip install --force-reinstall --no-deps {url}')
            print("[llm-prompter] (or re-run the node's install.py). Wheels: "
                  "https://github.com/JamePeng/llama-cpp-python/releases")
            print("=" * 72)
        print(f"[llm-prompter] Loading model: {model}  (n_gpu_layers={n_gpu_layers}, n_cpu_moe={n_cpu_moe})")
        cls.llm = Llama(
            model_path,
            chat_handler=cls.chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_cpu_moe=n_cpu_moe,
            n_ctx=n_ctx,
            type_k=type_k,
            type_v=type_v,
            verbose=False,
        )


# Free the resident LLM when ComfyUI unloads all models (frees VRAM for diffusion).
if not hasattr(mm, "_llm_prompter_hooked"):
    mm._llm_prompter_hooked = True
    _orig_unload_all = mm.unload_all_models

    def _patched_unload_all(*args, **kwargs):
        LLMEngine.unload()
        return _orig_unload_all(*args, **kwargs)

    mm.unload_all_models = _patched_unload_all
    print("[llm-prompter] VRAM cleanup hook applied.")
