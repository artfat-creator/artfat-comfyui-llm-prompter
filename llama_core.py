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

from .support.gguf_layers import get_layer_count, get_nextn_count

# GPU status, checked once at import. NOTE: llama_cpp.llama_supports_gpu_offload() is
# UNRELIABLE here — it returns False even on a working GPU build (checked before backend
# init), so we DON'T use it. Instead we look for a real ggml backend library:
#   - HIP/ROCm: ggml-hip(.dll/.so)  — AMD; no torch CUDA major check (ROCm torch often
#     has torch.version.cuda is None)
#   - Vulkan / Metal: ggml-vulkan / ggml-metal
#   - CUDA: ggml-cuda + cudart/cublas major must match torch's CUDA major — a mismatch
#     (e.g. cu128 wheel on cu130 torch) means ggml-cuda can't load and silently falls
#     back to CPU.
_JAMEPENG_VER, _JAMEPENG_DATE = "0.3.44", "20260721"
_CU_TAGS = {12: [124, 126, 128], 13: [130, 131]}
# Windows AMD: build JamePeng llama-cpp-python with HIP via the bundled script
# (abetlen hip-radeon wheels lack Qwen3.5 / other handlers this node expects).
_HIP_BUILD_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scripts", "build-jamepeng-hip.ps1"
)


def _ggml_lib_dir():
    import llama_cpp
    return os.path.join(os.path.dirname(llama_cpp.__file__), "lib")


def _find_ggml_backend(lib_dir, stem):
    """Path to ggml-<stem> shared lib (.dll / .so), or None."""
    for name in (f"{stem}.dll", f"{stem}.so", f"lib{stem}.so"):
        path = os.path.join(lib_dir, name)
        if os.path.exists(path):
            return path
    return None


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
    """(status, backend, detail).

    status: 'ok' | 'cpu' | 'mismatch' | 'unknown'
    backend: 'hip' | 'vulkan' | 'metal' | 'cuda' | None
    detail: for CUDA mismatch/unknown -> (need_major, torch_major); else None
    """
    try:
        lib_dir = _ggml_lib_dir()
        # Non-CUDA GPU backends: library present == GPU-capable build. Do this before
        # CUDA so a HIP wheel is never mislabeled "CPU-ONLY (no CUDA)".
        for backend, stem in (
            ("hip", "ggml-hip"),
            ("vulkan", "ggml-vulkan"),
            ("metal", "ggml-metal"),
        ):
            if _find_ggml_backend(lib_dir, stem):
                return "ok", backend, None

        cuda_lib = _find_ggml_backend(lib_dir, "ggml-cuda")
        if not cuda_lib:
            return "cpu", None, None
        need = _ggml_cuda_major(cuda_lib) if cuda_lib.endswith(".dll") else None
        try:
            import torch
            tc = getattr(torch.version, "cuda", None)
            tmaj = int(tc.split(".")[0]) if tc else None
        except Exception:
            tmaj = None
        if need is None or tmaj is None:
            # Linux .so or unreadable PE imports: assume OK if the backend lib exists.
            if need is None and tmaj is None and not cuda_lib.endswith(".dll"):
                return "ok", "cuda", None
            return "unknown", "cuda", (need, tmaj)
        if need == tmaj:
            return "ok", "cuda", (need, tmaj)
        return "mismatch", "cuda", (need, tmaj)
    except Exception:
        return "unknown", None, None


_GPU_STATUS, _GPU_BACKEND, _GPU_DETAIL = _gpu_status()
if _GPU_STATUS == "ok" and _GPU_BACKEND:
    print(f"[llm-prompter] GPU backend: {_GPU_BACKEND}")


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


def _hip_build_hint():
    """How Windows AMD users should (re)build a HIP llama-cpp-python for this node."""
    return (
        f'powershell -ExecutionPolicy Bypass -File "{_HIP_BUILD_SCRIPT}" '
        "(ComfyUI fully closed; needs existing ROCm in python_env)"
    )


def _llama_update_hint():
    """Where to get a matching llama-cpp-python build for this GPU backend."""
    if sys.platform == "win32" and _GPU_BACKEND == "hip":
        return _hip_build_hint()
    return "https://github.com/JamePeng/llama-cpp-python/releases"


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


def _try_add(names, *import_names):
    """Register optional, version-dependent chat handlers if the fork ships them.

    Accepts several class names for one handler: upstream sometimes renames a class
    between releases (llama-cpp-python 0.3.48 renamed MiniCPMv46ChatHandler ->
    MiniCPMV46ChatHandler), and a single hardcoded name makes the handler vanish from
    the dropdown silently. First name that imports wins.
    """
    handler = None
    for import_name in import_names:
        try:
            module = __import__("llama_cpp.llama_chat_format", fromlist=[import_name])
            handler = getattr(module, import_name)
            break
        except Exception:
            continue
    if handler is None:
        return
    for name in names:
        CHAT_HANDLERS.append(name)
        _HANDLERS[name] = handler


_try_add(["Gemma3"], "Gemma3ChatHandler")
_try_add(["Gemma4"], "Gemma4ChatHandler")
_try_add(["Qwen2.5-VL", "MinerU2.5-Pro"], "Qwen25VLChatHandler")
_try_add(["Qwen3-VL (no thinking)", "Qwen3-VL (thinking)"], "Qwen3VLChatHandler")
# One entry per family, not per release. Qwen3.6 and Qwen3.8 GGUFs both report `qwen35`
# as their architecture, so they share this handler; spelling that out in the label saves
# people hunting for a "Qwen3.8" entry that will never exist.
_try_add(["Qwen3.5 / 3.6 / 3.8 (no thinking)", "Qwen3.5 / 3.6 / 3.8 (thinking)"], "Qwen35ChatHandler")
_try_add(["GLM-4.6V (no thinking)", "GLM-4.6V (thinking)"], "GLM46VChatHandler")
_try_add(["GLM-4.1V (thinking)"], "GLM41VChatHandler")
_try_add(["LFM2-VL"], "LFM2VLChatHandler")
_try_add(["LFM2.5-VL"], "LFM25VLChatHandler")
_try_add(["MiniCPM-v4.5 (no thinking)", "MiniCPM-v4.5 (thinking)"], "MiniCPMv45ChatHandler")
_try_add(["MiniCPM-v4.6 (no thinking)", "MiniCPM-v4.6 (thinking)"],
         "MiniCPMV46ChatHandler", "MiniCPMv46ChatHandler")
# Generic multimodal path when the installed llama-cpp-python ships it.
_try_add(["MTMD"], "MTMDChatHandler")

# Labels shipped before v0.4.0. ComfyUI stores widget values by their text, so a saved
# workflow still carries the old label and would otherwise land on "Value not in list".
# These stay resolvable forever; they are deliberately absent from CHAT_HANDLERS so the
# dropdown only offers the current names.
_LEGACY_LABELS = {
    "Qwen3-VL": "Qwen3-VL (no thinking)",
    "Qwen3-VL-Thinking": "Qwen3-VL (thinking)",
    "Qwen3.5": "Qwen3.5 / 3.6 / 3.8 (no thinking)",
    "Qwen3.5-Thinking": "Qwen3.5 / 3.6 / 3.8 (thinking)",
    "Qwen3.6": "Qwen3.5 / 3.6 / 3.8 (thinking)",
    "Qwen3.6-Thinking": "Qwen3.5 / 3.6 / 3.8 (thinking)",
    "GLM-4.6V": "GLM-4.6V (no thinking)",
    "GLM-4.6V-Thinking": "GLM-4.6V (thinking)",
    "GLM-4.1V-Thinking": "GLM-4.1V (thinking)",
    "MiniCPM-v4.5": "MiniCPM-v4.5 (no thinking)",
    "MiniCPM-v4.5-Thinking": "MiniCPM-v4.5 (thinking)",
    "MiniCPM-v4.6": "MiniCPM-v4.6 (no thinking)",
    "MiniCPM-v4.6-Thinking": "MiniCPM-v4.6 (thinking)",
}


def normalize_handler(name):
    """Current label for a handler name, or None if it is not one we know.

    Accepts both the current labels and the pre-0.4.0 ones, so a workflow saved before
    the rename keeps working without the user touching the node.
    """
    if name in _HANDLERS:
        return name
    mapped = _LEGACY_LABELS.get(name)
    return mapped if mapped in _HANDLERS else None


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


def _mtp_kwargs(model_path, enabled, draft_max, vision=False):
    """Extra Llama() kwargs for MTP speculative decoding, or {} when it cannot be used.

    MTP (Multi-Token Prediction) uses NextN heads baked into an "-mtp" GGUF as a built-in
    draft model, so generation needs no second model. Measured on Qwen3.8-27B Q4_K_M /
    RTX 3090: 17.6 -> 33.5 tok/s (+90%) for +430 MiB VRAM. Gains scale with how predictable
    the text is, so other workloads will differ.

    The toggle means "use it if this model has it", never "enable or die": every way MTP can
    fail is checked up front and downgraded to a normal load with a printed reason, because
    each of them otherwise surfaces as a stack trace on an ordinary generation:

      * no NextN tensors -> llama.cpp raises at context creation ("context type MTP requested
        but model doesn't contain MTP layers").
      * llama-cpp-python older than 0.3.48 -> the speculative API does not exist yet.
      * an mmproj is loaded -> images enter the sequence as NEGATIVE placeholder token ids,
        the MTP drafter re-evals that prefix, and Llama._validate_eval_tokens rejects them
        ("invalid negative token id at index N: -10214670"). Verified on 0.3.48: text-only
        with MTP works, the same model with mmproj fails every run. Text and vision are
        separate loads anyway, so this only costs the speedup on image runs.
    """
    if not enabled:
        return {}
    if vision:
        print("[llm-prompter] mtp_speculative is ON but an mmproj is loaded -> MTP disabled "
              "for this run (MTP cannot draft across image tokens). Text-only runs still get it.")
        return {}
    heads = get_nextn_count(model_path)
    if heads <= 0:
        print("[llm-prompter] mtp_speculative is ON but this GGUF has no MTP/NextN tensors "
              "-> loading normally. Use the '-mtp' build of the model to get the speedup.")
        return {}
    try:
        from llama_cpp.llama_speculative import SpecConfig, SpeculativeType
    except Exception:
        print("[llm-prompter] mtp_speculative needs llama-cpp-python >= 0.3.48 "
              "-> loading normally. Run install.py to upgrade.")
        return {}
    print(f"[llm-prompter] MTP enabled: {heads} NextN head(s), draft_n_max={draft_max}")
    return {
        "load_mtp": True,
        "speculative": SpecConfig(spec_type=SpeculativeType.DRAFT_MTP, draft_n_max=draft_max),
    }


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
        mtp_speculative = bool(config.get("mtp_speculative", False))
        mtp_draft_max = int(config.get("mtp_draft_max", 2))

        handler_cls = _HANDLERS.get(chat_handler_name)
        if chat_handler_name not in _HANDLERS:
            raise ValueError(f'Unknown chat_handler: "{chat_handler_name}"')

        model_path = _resolve_gguf_path(model)
        n_gpu_layers = -1
        mtp_kwargs = _mtp_kwargs(model_path, mtp_speculative, mtp_draft_max,
                                 vision=bool(mmproj and mmproj != "None"))

        if vram_limit != -1:
            layers = get_layer_count(model_path) or 32
            gguf_gb = os.path.getsize(model_path) * 1.55 / (1024 ** 3)
            layer_gb = gguf_gb / max(1, layers)

        if mmproj and mmproj != "None":
            mmproj_path = _resolve_gguf_path(mmproj)
            if chat_handler_name == "None":
                vision_handlers = [h for h in CHAT_HANDLERS if h != "None"]
                raise ValueError(
                    '"chat_handler" cannot be None when an mmproj is set. '
                    "Pick the handler that matches your VLM family "
                    f"(available: {', '.join(vision_handlers)})."
                )

            if vram_limit != -1:
                mmproj_gb = os.path.getsize(mmproj_path) * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_gb) / layer_gb))

            print(f"[llm-prompter] Loading clip: {mmproj}")
            # Labels carry the mode as a suffix, e.g. "Qwen3.5 / 3.6 / 3.8 (thinking)".
            think = "(thinking)" in chat_handler_name
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}
            if chat_handler_name.startswith("Qwen3-VL"):
                kwargs["force_reasoning"] = think
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            elif chat_handler_name.startswith(("MiniCPM-v4.5", "MiniCPM-v4.6", "GLM-4.6V", "Qwen3.5")):
                kwargs["enable_thinking"] = think
            try:
                cls.chat_handler = handler_cls(**kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"{e}\nUpdate llama-cpp-python: {_llama_update_hint()}"
                )
        else:
            # No mmproj set. Every chat_handler in _HANDLERS is a VISION handler that
            # REQUIRES an mmproj (llama_cpp raises "mmproj_path is required" otherwise).
            # So a VL handler picked without an mmproj can only mean text-only intent —
            # fall back to the model's own chat template instead of crashing.
            if handler_cls is not None:
                print(f"[llm-prompter] chat_handler '{chat_handler_name}' needs an mmproj (vision), "
                      f"but none is set -> loading text-only (no vision). Set chat_handler=None to silence this.")
                handler_cls = None
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / layer_gb))
            cls.chat_handler = handler_cls(verbose=False) if handler_cls is not None else None

        # Advisory when there is no usable GPU backend (or CUDA major mismatch).
        # HIP already counts as ok above — do not push CUDA-only / abetlen HIP wheels.
        if n_gpu_layers != 0 and _GPU_STATUS in ("cpu", "mismatch"):
            exe, url = _cuda_wheel_hint()
            print("=" * 72)
            if _GPU_STATUS == "mismatch":
                need, tmaj = _GPU_DETAIL
                print(f"[llm-prompter] WARNING: llama-cpp-python is built for CUDA {need}.x, but "
                      f"your torch uses CUDA {tmaj}.x -> GPU offload FAILS (silent CPU fallback).")
                print("[llm-prompter] The LLM will run on CPU (slow). To fix, FULLY CLOSE ComfyUI, then run:")
                print(f'[llm-prompter]   "{exe}" -m pip install --force-reinstall --no-deps {url}')
                print("[llm-prompter] (or re-run the node's install.py). Wheels: "
                      "https://github.com/JamePeng/llama-cpp-python/releases")
            else:
                print("[llm-prompter] WARNING: llama-cpp-python has no GPU backend "
                      "(no ggml-cuda / ggml-hip).")
                print("[llm-prompter] The LLM will run on CPU (slow). FULLY CLOSE ComfyUI, then:")
                print(f'[llm-prompter]   NVIDIA: "{exe}" -m pip install --force-reinstall --no-deps {url}')
                if sys.platform == "win32":
                    print(f"[llm-prompter]   AMD HIP (Windows): {_hip_build_hint()}")
            print("=" * 72)
        backend = _GPU_BACKEND or "cpu"
        print(f"[llm-prompter] Loading model: {model}  "
              f"(n_gpu_layers={n_gpu_layers}, n_cpu_moe={n_cpu_moe}, backend={backend})")
        cls.llm = Llama(
            model_path,
            chat_handler=cls.chat_handler,
            n_gpu_layers=n_gpu_layers,
            n_cpu_moe=n_cpu_moe,
            n_ctx=n_ctx,
            type_k=type_k,
            type_v=type_v,
            verbose=False,
            **mtp_kwargs,
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
