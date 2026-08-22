"""Artfat LLM Prompter — one all-in-one LLM/VLM prompt node for ComfyUI.

Flow:  (image_1 / image_2 / text) -> LLM -> prompt -> CLIP -> CONDITIONING

Highlights:
  * resident llama.cpp model (no reload between runs) with optional force_offload
  * dual reference images, composite or batch (dataset captioning) modes
  * .txt system presets + task/instruction presets (auto-filled, editable live)
  * built-in CLIP Text Encode: with llm_enabled off it just encodes raw instruction
  * cache-correct: no random IS_CHANGED, so a fixed seed reuses the cached result
"""

import base64
import gc
import io
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image

import comfy.model_management as mm
import folder_paths

from .llama_core import LLMEngine, CHAT_HANDLERS
from .presets import (
    INSTRUCTION_PRESETS,
    INSTRUCTION_NAMES,
    list_system_presets,
    load_system_preset,
)
from .support.cqdm import cqdm

# Register models/LLM. `folder_names_and_paths` is a GLOBAL namespace shared by every installed
# pack, so another node may have claimed the "LLM" key before us — ComfyUI-Florence2 does exactly
# that at import time, and it loads first (custom_nodes are imported alphabetically). Its
# add_model_folder_path() creates the key with an EMPTY extension set, and an empty set means
# filter_files_extensions() lets EVERY file through — which is why .safetensors/.bin showed up in
# the model dropdown. So: merge instead of skipping — add our path if missing and union in our
# extensions. Purely additive; nothing another pack registered is removed.
_LLM_DIR = os.path.join(folder_paths.models_dir, "LLM")
_LLM_EXTS = {".gguf", ".bin", ".safetensors"}
if "LLM" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["LLM"] = ([_LLM_DIR], set(_LLM_EXTS))
else:
    _paths, _exts = folder_paths.folder_names_and_paths["LLM"]
    if _LLM_DIR not in _paths:
        _paths.append(_LLM_DIR)
    # An empty set is "allow everything"; only narrow it once we know who else contributed.
    if _exts:
        _exts.update(_LLM_EXTS)


# --- remember last-used SETTINGS so a freshly-dragged node inherits them ------------------
# Only technical settings (model/handler/sampler/…), NEVER prompt content (instruction,
# final_prompt, negative, seed, prefix/suffix). Written on every run, read in INPUT_TYPES.
_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "last_settings.json")
_REMEMBER = (
    "model", "mmproj", "chat_handler", "n_ctx", "vram_limit", "n_cpu_moe", "llm_enabled",
    "system_preset", "instruction_preset", "mode", "force_offload",
    "max_tokens", "temperature", "top_k", "top_p", "min_p", "typical_p", "repeat_penalty",
    "frequency_penalty", "mirostat_mode", "mirostat_tau", "mirostat_eta", "type_k", "type_v",
    "max_size", "image_min_tokens", "image_max_tokens",
    "mtp_speculative", "mtp_draft_max",
)


def _load_last_settings():
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_last_settings(values):
    try:
        data = {k: values[k] for k in _REMEMBER if k in values}
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _apply_saved_defaults(required, saved):
    """Patch each remembered field's default from the saved settings. Combo fields are only
    overridden if the saved value is still a valid choice (e.g. the model still exists)."""
    for name, val in saved.items():
        if name not in required:
            continue
        spec = required[name]
        first = spec[0]
        opts = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(first, list):
            if val not in first:
                continue
        opts["default"] = val
        required[name] = (first, opts)
    return required


class AnyType(str):
    def __ne__(self, other):
        return False


any_type = AnyType("*")

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)
_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.I)


def _clean(text, keep_think):
    if not keep_think:
        # Remove paired <think>...</think> blocks.
        text = _THINK_RE.sub("", text)
        # Some reasoning models emit thoughts with only a closing tag (no opening);
        # keep everything after the LAST closing tag.
        last = None
        for last in _CLOSE_RE.finditer(text):
            pass
        if last:
            text = text[last.end():]
    text = text.strip()
    text = text.removeprefix("```json").removeprefix("```")
    text = text.removesuffix("```")
    return text.strip()


def _tensor_to_b64(frame, max_size=None):
    arr = np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    if max_size:
        w, h = pil.size
        scale = min(max_size / max(w, h), 1.0)
        if scale < 1.0:
            pil = pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_item(b64):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _encode(clip, text):
    if clip is None:
        return None
    tokens = clip.tokenize(text or "")
    return clip.encode_from_tokens_scheduled(tokens)


def _num(v, default, lo=None, hi=None, cast=float):
    """Coerce a widget value to a number, falling back to default and clamping."""
    try:
        x = cast(v)
    except (TypeError, ValueError):
        return default
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x


class ArtfatLLMPrompter:
    @classmethod
    def INPUT_TYPES(cls):
        # GGUF only: the engine is llama-cpp-python, so a .safetensors/.bin pick could only ever
        # fail with a generic "Failed to load model from file". Also scan "clip"
        # (models/text_encoders) — GGUF vision/text encoders (Qwen2.5-VL / Qwen3-VL) conventionally
        # live there because CLIP loaders need them too.
        llms = folder_paths.get_filename_list("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
        try:
            clips = folder_paths.get_filename_list("clip")
        except Exception:
            clips = []
        ggufs = sorted({f for f in list(llms) + list(clips) if f.lower().endswith(".gguf")})
        models = [f for f in ggufs if "mmproj" not in f.lower()] or ["<put GGUF in models/LLM or models/text_encoders>"]
        mmprojs = ["None"] + [f for f in ggufs if "mmproj" in f.lower()]
        sys_presets = list_system_presets()
        types = {
            "required": {
                "model": (models,),
                "mmproj": (mmprojs, {"default": "None"}),
                "chat_handler": (CHAT_HANDLERS, {"default": "None"}),
                "n_ctx": ("INT", {"default": 8192, "min": 1024, "max": 327680, "step": 128}),
                "vram_limit": ("INT", {"default": -1, "min": -1, "max": 1024, "step": 1,
                                       "tooltip": "VRAM budget in GB for the LLM (-1 = put all layers on GPU)."}),
                "n_cpu_moe": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1,
                                      "tooltip": "Keep the MoE experts of the first N layers on CPU (frees VRAM on MoE models)."}),
                "llm_enabled": ("BOOLEAN", {"default": True,
                                            "tooltip": "OFF = skip the LLM and encode the instruction text as a plain CLIP Text Encode."}),
                "system_preset": (sys_presets, {"default": "Custom"}),
                "mode": (["composite", "batch"], {"default": "composite",
                         "tooltip": "composite: all images -> one prompt.  batch: each image -> its own caption (dataset)."}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "step": 1,
                                 "tooltip": "Fixed seed reuses the cached prompt. -1 = random each call. Use control_after_generate=randomize for a fresh prompt every queue."}),
                "force_offload": ("BOOLEAN", {"default": False,
                                              "tooltip": "Unload the LLM from VRAM after running (frees VRAM for diffusion; next LLM call reloads)."}),
                "prefix": ("STRING", {"default": "", "multiline": False,
                                      "placeholder": "added BEFORE the prompt — e.g. LoRA trigger word",
                                      "tooltip": "Text auto-prepended to the final prompt before CLIP encode. Use it for a LoRA trigger word so you never type it into the prompt by hand."}),
                "suffix": ("STRING", {"default": "", "multiline": False,
                                      "placeholder": "added AFTER the prompt — e.g. quality tags",
                                      "tooltip": "Text auto-appended to the final prompt before CLIP encode. Use it for trailing style/quality tags (e.g. 'amateur photo, film grain')."}),
                "instruction_preset": (INSTRUCTION_NAMES, {"default": "Describe -> prompt"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True,
                                             "placeholder": "system prompt (auto-filled from preset, editable)"}),
                "instruction": ("STRING", {"default": "", "multiline": True,
                                           "placeholder": "task / instruction (auto-filled from preset, editable)"}),
                "user_preset": ("STRING", {"default": "", "multiline": True,
                                           "placeholder": "extra ad-hoc instructions, appended (not saved to a file)"}),
                "negative": ("STRING", {"default": "", "multiline": True,
                                        "placeholder": "negative prompt (encoded to the negative output)"}),
                "final_prompt": ("STRING", {"default": "", "multiline": True,
                                            "placeholder": "LLM writes the generated prompt here (editable, copyable). With LLM off, type your prompt here — it is encoded to CLIP (falls back to 'instruction' if left empty).",
                                            "tooltip": "OUTPUT when LLM is on: the generated prompt appears here, editable and copyable. INPUT when LLM is off: this text is encoded directly to CLIP."}),
                # --- advanced (collapsed by web/llm_prompter.js) ---
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "type_k": (["f16", "q8_0", "q4_0"], {"default": "f16"}),
                "type_v": (["f16", "q8_0", "q4_0"], {"default": "f16"}),
                "max_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64,
                                     "tooltip": "Downscale reference images to this max side before encoding (batch/multi-image)."}),
                "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                # Hidden flag driven by web/llm_prompter.js: True only when the seed's
                # control_after_generate == "fixed". Placed at the END of INPUT_TYPES so adding it
                # does NOT shift any existing saved node's positional widget values (no re-add needed).
                "freeze": ("BOOLEAN", {"default": False}),
                # --- batch prompts (added at the END so no positional widget-value drift) ---
                "batch_mode": ("BOOLEAN", {"default": False,
                                           "tooltip": "ON: ignore the LLM and treat 'batch_prompts' as a list (one prompt per line). Wire the 'positive_list' output to KSampler.positive -> one image per line from a single Queue."}),
                "batch_prompts": ("STRING", {"default": "", "multiline": True,
                                             "placeholder": "batch mode: one prompt per line. blank lines skipped, lines starting with # are comments. prefix/suffix still apply to each."}),
                # --- MTP speculative decoding (added at the END so no positional widget-value drift) ---
                "mtp_speculative": ("BOOLEAN", {"default": False,
                                                "tooltip": "Use the model's built-in MTP/NextN heads to draft tokens. Only works on '-mtp' GGUF builds; on any other model it is ignored and the model loads normally. Costs a few hundred MB of VRAM. Measured +90% tok/s on Qwen3.8-27B."}),
                "mtp_draft_max": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                                          "tooltip": "How many tokens MTP drafts ahead per step. 2 is the value upstream recommends for 27B. Higher drafts more but wastes more when a guess is rejected."}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "clip": ("CLIP",),
                "queue": (any_type, {"tooltip": "Optional chain input to force execution order between prompter nodes."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }
        # A freshly-dragged node inherits the last-used settings (models, handler, sampler…).
        _apply_saved_defaults(types["required"], _load_last_settings())
        return types

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "IMAGE", "IMAGE", any_type, "CLIP", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative", "prompt", "prompt_list", "image_1", "image_2", "queue", "clip", "positive_list")
    # positive_list (index 8) is a LIST: batch_mode -> one CONDITIONING per prompt line; else a 1-element list.
    OUTPUT_IS_LIST = (False, False, False, True, False, False, False, False, True)
    FUNCTION = "run"
    CATEGORY = "artfat/llm"
    # NOT an OUTPUT_NODE: that would force re-execution every queue and defeat caching.
    # The final-prompt UI still updates via the {"ui": {"text": ...}} return whenever the node runs.

    # --- helpers -------------------------------------------------------------

    def _collect_frames(self, image_1, image_2):
        frames = []
        for img in (image_1, image_2):
            if img is not None:
                for i in range(img.shape[0]):
                    frames.append(img[i])
        return frames

    def _resolve_text(self, system_preset, system_prompt, instruction_preset, instruction, user_preset):
        sys_text = system_prompt.strip()
        if not sys_text and system_preset not in ("Custom", "None", ""):
            sys_text = load_system_preset(system_preset) or ""
        instr = instruction.strip()
        if not instr and instruction_preset != "Custom":
            instr = INSTRUCTION_PRESETS.get(instruction_preset, "")
        parts = [p for p in (instr, user_preset.strip()) if p]
        user_text = "\n\n".join(parts)
        return sys_text, user_text

    def _sampler(self, max_tokens, temperature, top_k, top_p, min_p, typical_p,
                 repeat_penalty, frequency_penalty, mirostat_mode, mirostat_tau, mirostat_eta):
        return dict(
            max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, typical_p=typical_p, repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty, mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau, mirostat_eta=mirostat_eta,
        )

    def _gen(self, messages, run_seed, sampler):
        out = LLMEngine.llm.create_chat_completion(messages=messages, seed=run_seed, **sampler)
        return out["choices"][0]["message"]["content"].removeprefix(": ").lstrip()

    # --- main ----------------------------------------------------------------

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Accept anything; run() coerces each value to a safe default. This keeps a
        # node saved under an older widget layout running instead of hard-erroring.
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Deterministic hash of every input so ComfyUI caches the node when nothing changed.
        # With a fixed seed and unchanged inputs the hash is identical -> the node is skipped and
        # the previous prompt/conditioning flows straight to the sampler (no LLM re-run).
        import hashlib
        h = hashlib.sha256()
        # BATCH: hash only the fields that shape positive_list (skips the LLM path entirely, so
        # image tensors / LLM fields don't affect the result). Re-runs when the list/wrappers change.
        if bool(kwargs.get("batch_mode", False)):
            h.update((f"batch|bp={kwargs.get('batch_prompts', '')!r}|neg={kwargs.get('negative', '')!r}"
                      f"|pre={kwargs.get('prefix', '')!r}|suf={kwargs.get('suffix', '')!r}"
                      f"|clip={kwargs.get('clip') is not None}").encode("utf-8", "ignore"))
            return h.hexdigest()
        # LLM OFF: the node encodes ONLY final_prompt, so hash just the fields that affect that
        # output. Skipping the reference-image tensors (a multi-MB .cpu()+sha256 per queue) and the
        # LLM-only fields makes IS_CHANGED instant -> OFF mode goes straight to the sampler.
        if not bool(kwargs.get("llm_enabled", True)):
            h.update((f"off|fp={kwargs.get('final_prompt', '')!r}|neg={kwargs.get('negative', '')!r}"
                      f"|pre={kwargs.get('prefix', '')!r}|suf={kwargs.get('suffix', '')!r}"
                      f"|clip={kwargs.get('clip') is not None}").encode("utf-8", "ignore"))
            return h.hexdigest()
        # LLM ON: hash every input. A changed seed (control_after_generate=randomize/increment)
        # changes the hash -> the node re-executes and run() regenerates. An unchanged seed (fixed)
        # with unchanged inputs keeps the hash stable -> ComfyUI caches and the prompt flows to the
        # sampler with no model work. final_prompt stays in the hash so a manual edit re-encodes;
        # run()'s frozen branch prevents any regenerate-on-its-own-output loop.
        for name in sorted(kwargs):
            v = kwargs[name]
            if v is not None and hasattr(v, "cpu") and hasattr(v, "numpy"):
                try:
                    h.update(name.encode())
                    h.update(v.cpu().numpy().tobytes())
                    continue
                except Exception:
                    pass
            h.update(f"{name}={v!r}".encode("utf-8", "ignore"))
        return h.hexdigest()

    def run(self, model, mmproj, chat_handler, n_ctx, vram_limit, n_cpu_moe, llm_enabled,
            system_preset, system_prompt, instruction_preset, instruction, user_preset,
            negative, final_prompt, prefix, suffix, mode, max_tokens, temperature, seed, force_offload,
            top_k, top_p, min_p, typical_p, repeat_penalty, frequency_penalty,
            mirostat_mode, mirostat_tau, mirostat_eta, type_k, type_v, max_size,
            image_min_tokens, image_max_tokens, freeze=False, batch_mode=False, batch_prompts="",
            mtp_speculative=False, mtp_draft_max=2,
            image_1=None, image_2=None, clip=None, queue=None, unique_id=None):

        # --- sanitize every widget value (tolerate stale / shifted saved values) ---
        n_ctx = int(_num(n_ctx, 8192, 1024, cast=int))
        vram_limit = int(_num(vram_limit, -1, -1, cast=int))
        n_cpu_moe = int(_num(n_cpu_moe, 0, 0, cast=int))
        mtp_speculative = bool(mtp_speculative)
        mtp_draft_max = int(_num(mtp_draft_max, 2, 1, 8, cast=int))
        max_tokens = int(_num(max_tokens, 512, 16, 8192, cast=int))
        temperature = _num(temperature, 0.6, 0.0, 2.0)
        seed = int(_num(seed, 0, -1, cast=int))
        top_k = int(_num(top_k, 40, 0, cast=int))
        top_p = _num(top_p, 0.9, 0.0, 1.0)
        min_p = _num(min_p, 0.05, 0.0, 1.0)
        typical_p = _num(typical_p, 1.0, 0.0, 1.0)
        repeat_penalty = _num(repeat_penalty, 1.05, 0.0, 10.0)
        frequency_penalty = _num(frequency_penalty, 0.0, 0.0, 2.0)
        mirostat_mode = int(_num(mirostat_mode, 0, 0, 2, cast=int))
        mirostat_tau = _num(mirostat_tau, 5.0, 0.0, 10.0)
        mirostat_eta = _num(mirostat_eta, 0.1, 0.0, 1.0)
        max_size = int(_num(max_size, 512, 128, cast=int))
        image_min_tokens = int(_num(image_min_tokens, 0, 0, cast=int))
        image_max_tokens = int(_num(image_max_tokens, 0, 0, cast=int))
        if type_k not in ("f16", "q8_0", "q4_0"):
            type_k = "f16"
        if type_v not in ("f16", "q8_0", "q4_0"):
            type_v = "f16"
        if mode not in ("composite", "batch"):
            mode = "composite"
        if chat_handler not in CHAT_HANDLERS:
            chat_handler = "None"
        if instruction_preset not in INSTRUCTION_PRESETS:
            instruction_preset = "Custom"
        system_preset = str(system_preset or "Custom")
        system_prompt = str(system_prompt or "")
        instruction = str(instruction or "")
        user_preset = str(user_preset or "")
        negative = str(negative or "")
        final_prompt = str(final_prompt or "")
        freeze = bool(freeze)
        prefix = str(prefix or "")
        suffix = str(suffix or "")
        batch_mode = bool(batch_mode)
        batch_prompts = str(batch_prompts or "")

        # Remember the technical settings so the next freshly-dragged node inherits them.
        _save_last_settings(locals())

        # --- BATCH MODE: one prompt per line -> one CONDITIONING each (skip the LLM) ---
        # Wire the `positive_list` output to KSampler.positive: ComfyUI runs the graph once per
        # list item, so a single Queue produces one image per line. prefix/suffix still wrap each.
        if batch_mode and batch_prompts.strip():
            # Split into prompts. A line STARTING with "N." ("1.", "4. text", "4.Prompt") is a numbered-list
            # marker (the negative lookahead (?!\d) keeps decimals like "2.5"/"f2.8"/"5:30" intact).
            # If markers are present -> split ON them: everything from one "N." to the next is ONE prompt
            # (so a multi-line prompt stays whole), and any preamble BEFORE the first "N." (a description
            # header) is dropped. If no markers -> fall back to one prompt per line. "#" lines are comments.
            _mark = re.compile(r"^\s*\d+\.(?!\d)\s*")
            raw = batch_prompts.splitlines()
            marker_idx = [i for i, l in enumerate(raw) if _mark.match(l)]
            lines = []
            if marker_idx:
                for j, start in enumerate(marker_idx):
                    end = marker_idx[j + 1] if j + 1 < len(marker_idx) else len(raw)
                    block = list(raw[start:end])
                    block[0] = _mark.sub("", block[0], count=1)  # strip the leading "N." from block start
                    txt = " ".join(l.strip() for l in block
                                   if l.strip() and not l.strip().startswith("#")).strip()
                    if txt:
                        lines.append(txt)
            else:
                for ln in raw:
                    ln = ln.strip()
                    if ln and not ln.startswith("#"):
                        lines.append(ln)
            wrapped = [f"{prefix}{ln}{suffix}" if (prefix or suffix) else ln for ln in lines]
            if not wrapped:
                wrapped = [""]
            print(f"[llm-prompter] batch_mode: {len(wrapped)} prompt(s) -> positive_list (LLM skipped)")
            pos_list = [_encode(clip, t) for t in wrapped]
            neg_cond = _encode(clip, negative)
            main = wrapped[0]
            result = (pos_list[0], neg_cond, main, wrapped,
                      image_1, image_2, queue, clip, pos_list)
            return {"ui": {"text": ["\n\n".join(wrapped)]}, "result": result}

        sys_text, user_text = self._resolve_text(
            system_preset, system_prompt, instruction_preset, instruction, user_preset)
        frames = self._collect_frames(image_1, image_2)
        keep_think = False  # reasoning is always stripped from the prompt output

        def finalize(p):
            p = _clean(p, keep_think) if llm_enabled else p.strip()
            return f"{prefix}{p}{suffix}" if (prefix or suffix) else p

        # `freeze` is set by web/llm_prompter.js to True only when the seed's control_after_generate
        # is "fixed" (a stable per-user choice, unlike the seed which changes on randomize). So:
        # fixed + a prompt already in the field => reuse it, skip the LLM entirely. Empty field or
        # control != fixed => fall through and (re)generate.
        frozen = bool(llm_enabled and freeze and final_prompt.strip())

        prompts = []

        if not llm_enabled:
            # LLM off: encode ONLY the final_prompt text. The LLM-only fields (instruction,
            # user_preset, system_prompt, presets) are inert here and must NOT leak into the
            # prompt. prefix/suffix still apply via finalize(). Empty final_prompt -> empty prompt.
            prompts = [finalize(final_prompt.strip())]
        elif frozen:
            # FROZEN — fixed seed (unchanged since the last generation) + a prompt already in the
            # field. Reuse it verbatim and skip the ENTIRE LLM path: no LLMEngine.ensure_loaded(), so
            # the model is NOT loaded/reloaded even if force_offload unloaded it; no generation. The
            # existing prompt goes straight to CLIP -> sampler. It already has prefix/suffix baked in,
            # so it is encoded AS-IS. To regenerate: switch control_after_generate off "fixed"
            # (randomize/increment), or clear the final_prompt field.
            print("[llm-prompter] control_after_generate=fixed: reusing final_prompt, LLM not called, model untouched.")
            prompts = [final_prompt.strip()]
        else:
            config = {
                "model": model, "mmproj": mmproj, "chat_handler": chat_handler,
                "n_ctx": n_ctx, "vram_limit": vram_limit, "n_cpu_moe": n_cpu_moe,
                "type_k": type_k, "type_v": type_v,
                "image_min_tokens": image_min_tokens, "image_max_tokens": image_max_tokens,
                "mtp_speculative": mtp_speculative, "mtp_draft_max": mtp_draft_max,
            }
            LLMEngine.ensure_loaded(config)

            # llama-cpp-python >= 0.3.44 renamed the handler's vision-projector path from
            # `clip_model_path` to `mmproj_path` (old name is only a deprecated kwarg alias,
            # no longer an attribute). Accept either so both old and new builds work.
            _handler = LLMEngine.chat_handler
            _mmproj = getattr(_handler, "mmproj_path", None) or getattr(_handler, "clip_model_path", None)
            if frames and _mmproj is None:
                raise ValueError("Images are connected but the loaded model has no mmproj (vision) module.")

            run_seed = seed if seed >= 0 else random.randint(0, 2 ** 31 - 1)
            sampler = self._sampler(max_tokens, temperature, top_k, top_p, min_p, typical_p,
                                    repeat_penalty, frequency_penalty, mirostat_mode,
                                    mirostat_tau, mirostat_eta)

            base_msgs = []
            if sys_text:
                base_msgs.append({"role": "system", "content": sys_text})

            if not frames:
                msgs = base_msgs + [{"role": "user", "content": user_text}]
                prompts = [finalize(self._gen(msgs, run_seed, sampler))]
            elif mode == "batch":
                print(f"[llm-prompter] Batch captioning {len(frames)} image(s)")
                for frame in cqdm(frames):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()
                    content = [{"type": "text", "text": user_text},
                               _image_item(_tensor_to_b64(frame, max_size))]
                    msgs = base_msgs + [{"role": "user", "content": content}]
                    prompts.append(finalize(self._gen(msgs, run_seed, sampler)))
            else:  # composite
                content = [{"type": "text", "text": user_text}]
                for frame in frames:
                    content.append(_image_item(_tensor_to_b64(frame, max_size)))
                msgs = base_msgs + [{"role": "user", "content": content}]
                prompts = [finalize(self._gen(msgs, run_seed, sampler))]

            LLMEngine.reset_context(chat_handler)
            if force_offload:
                LLMEngine.unload()

        main_prompt = "\n\n".join(prompts) if mode == "batch" and len(prompts) > 1 else (prompts[0] if prompts else "")
        positive = _encode(clip, main_prompt)
        negative_cond = _encode(clip, negative)

        # Only collect after a real LLM run freed context/VRAM. In OFF / frozen mode nothing was
        # allocated, so skip the (heap-walking, ~0.5-1s) GC to keep those paths instant.
        if llm_enabled and not frozen:
            gc.collect()
        result = (positive, negative_cond, main_prompt, prompts,
                  image_1, image_2, queue, clip, [positive])
        return {"ui": {"text": [main_prompt]}, "result": result}


NODE_CLASS_MAPPINGS = {"ArtfatLLMPrompter": ArtfatLLMPrompter}
NODE_DISPLAY_NAME_MAPPINGS = {"ArtfatLLMPrompter": "Artfat LLM Prompter"}


# --- server routes: serve preset text to the web UI for live auto-fill --------
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/artfat_llm/system_preset")
    async def _sys_preset(request):
        name = request.query.get("name", "")
        return web.json_response({"text": load_system_preset(name) or ""})

    @PromptServer.instance.routes.get("/artfat_llm/instruction_preset")
    async def _instr_preset(request):
        name = request.query.get("name", "")
        return web.json_response({"text": INSTRUCTION_PRESETS.get(name, "")})
except Exception as e:
    print(f"[llm-prompter] Preset routes not registered: {e}")
