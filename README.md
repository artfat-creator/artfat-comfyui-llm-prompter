# ComfyUI — Artfat LLM Prompter

One all-in-one LLM/VLM node for ComfyUI. It turns reference images (or plain text)
into a generation prompt using a **resident** `llama.cpp` model, then encodes that
prompt straight into `CONDITIONING` — no separate loader, sampler-params or
CLIP Text Encode nodes needed.

```
(image_1 / image_2 / text) --> LLM --> prompt --> CLIP --> CONDITIONING
```

## Why one node

- **Resident model** — the GGUF is loaded once and kept in VRAM. Repeat runs are fast;
  it only reloads when a load-relevant setting changes.
- **Cache-correct** — no random `IS_CHANGED`. A fixed seed reuses the cached prompt, so
  KSampler with the same seed does **not** regenerate. Change something → only that recomputes.
- **Built-in CLIP Text Encode** — turn `llm_enabled` off and the node just encodes the
  instruction text as a plain positive/negative CONDITIONING.
- **Low-VRAM friendly** — `vram_limit`, `n_cpu_moe` (MoE expert offload), KV-cache
  quantization and an optional `force_offload` to free VRAM for diffusion.

## Features

- Dual reference images (`image_1`, `image_2`) with **composite** (one prompt) or
  **batch** (one caption per image — dataset captioning) modes.
- `.txt` **system presets** read from `models/LLM/prompts/` + **instruction presets**
  (Describe / Tags / Cinematic / Replace subject / Appearance only / …). Both auto-fill an
  editable box so you see and can tweak the text live.
- A free `user_preset` field for ad-hoc instructions (not written to any file).
- `prefix` / `suffix` — e.g. auto-prepend a LoRA trigger word before CLIP encode.
- Positive **and** negative CONDITIONING outputs.
- Reasoning-model support (`<think>` blocks stripped by default).
- Progress bar for batch runs; image pass-through outputs; optional `queue` chain input.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/artfat-creator/comfyui-llm-prompter.git
python -m pip install -r comfyui-llm-prompter/requirements.txt
```

`requirements.txt` pins the [JamePeng llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)
build (ships the VLM chat handlers). Pick the wheel that matches your Python/OS if pip
does not auto-select it.

## Models

Put GGUF files in `ComfyUI/models/LLM/`. For image input (VLM) also download the matching
`mmproj` weights into the same folder and pick the correct `chat_handler`
(e.g. `Qwen3.5`, `Qwen3-VL`, `MiniCPM-v4.5`).

System-prompt presets are plain `.txt` files in `ComfyUI/models/LLM/prompts/`.

## Inputs

| Input | What it does |
|---|---|
| `model` / `mmproj` | GGUF LLM + optional vision projector |
| `chat_handler` | Chat template — must match the model |
| `n_ctx` | Context length (default 8192) |
| `vram_limit` | VRAM budget in GB for the LLM (-1 = all layers on GPU) |
| `n_cpu_moe` | Keep MoE experts of the first N layers on CPU (frees VRAM) |
| `llm_enabled` | Off = skip LLM, encode the instruction as plain CLIP Text Encode |
| `system_preset` / `system_prompt` | Style/engine (.txt preset auto-fills the editable box) |
| `instruction_preset` / `instruction` | Task / output format (preset auto-fills the box) |
| `user_preset` | Extra ad-hoc instructions, appended |
| `negative` | Negative prompt → negative output |
| `prefix` / `suffix` | Wrap the final prompt (trigger words, quality tags) |
| `mode` | `composite` (images → one prompt) or `batch` (each image → its caption) |
| `max_tokens` / `temperature` / `seed` | Generation length / creativity / seed |
| `force_offload` | Unload the LLM after running (frees VRAM; next call reloads) |
| `reasoning` | `off`/`auto` strip `<think>` blocks, `on` keep them |
| advanced | `top_k`, `top_p`, `min_p`, `typical_p`, `repeat_penalty`, `frequency_penalty`, `mirostat_*`, `type_k`/`type_v` (KV quant), `max_size`, `image_min/max_tokens` |
| `image_1` / `image_2` / `clip` / `queue` | optional |

## Outputs

`positive` (CONDITIONING) · `negative` (CONDITIONING) · `prompt` (STRING) ·
`prompt_list` (STRING list, for batch) · `image_1` / `image_2` (pass-through) · `queue`

## Seed & caching

- **Fixed seed** → same input signature → ComfyUI returns the cached prompt, the LLM does
  not even run, and a downstream KSampler with the same seed serves its cached image.
- **`control_after_generate = randomize`** on the seed → a fresh prompt each queue.
- Seed only varies output when `temperature > 0`.

## VRAM / offload

Keep `force_offload` **off** to keep the LLM resident (fast repeat prompting). Turn it **on**
only on low-VRAM systems to free VRAM for diffusion — the trade-off is a model reload on the
next LLM call. The node also frees the LLM when ComfyUI unloads all models.

## Credits

- [lihaoyun6/ComfyUI-llama-cpp](https://github.com/lihaoyun6/ComfyUI-llama-cpp) — engine design this builds on
- [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) — llama.cpp bindings with VLM handlers
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

MIT © artfat-creator
