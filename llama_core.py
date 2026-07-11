"""Resident llama.cpp engine for the Artfat LLM Prompter node.

A single process-wide model is kept in VRAM and reused across runs. It is only
reloaded when a load-relevant setting changes, so repeat generations are fast.
Adapted from lihaoyun6/ComfyUI-llama-cpp (MIT), extended with n_cpu_moe and
KV-cache quantization for low-VRAM / MoE setups.
"""

import gc
import os

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

        model_path = os.path.join(folder_paths.models_dir, "LLM", model)
        n_gpu_layers = -1

        if vram_limit != -1:
            layers = get_layer_count(model_path) or 32
            gguf_gb = os.path.getsize(model_path) * 1.55 / (1024 ** 3)
            layer_gb = gguf_gb / max(1, layers)

        if mmproj and mmproj != "None":
            mmproj_path = os.path.join(folder_paths.models_dir, "LLM", mmproj)
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
