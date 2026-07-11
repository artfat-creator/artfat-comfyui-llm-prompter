# ComfyUI — Artfat LLM Prompter

One all-in-one LLM/VLM node for ComfyUI. It turns reference images (or plain text)
into a generation prompt using a **resident** `llama.cpp` model, then encodes that
prompt straight into `CONDITIONING` — no separate loader, sampler-params or
CLIP Text Encode nodes needed.

```
(image_1 / image_2 / text) --> LLM --> prompt --> CLIP --> CONDITIONING
```

![Two reference images composited into a new photo by the node](img/example.png)

*Two reference images → one node → prompt → image: the subject from Image 1 dropped into
the café setting of Image 2.*

## Why one node

- **Resident model** — the GGUF is loaded once and kept in VRAM. Repeat runs are fast;
  it only reloads when a load-relevant setting changes.
- **Editable `final_prompt`** — the generated prompt lands in an editable, copyable box under
  `negative`. Tweak it by hand, or (with the LLM off) type your own prompt there — that text is
  what gets encoded to CLIP.
- **Freeze with a `fixed` seed** — set `control_after_generate = fixed` and, once a prompt is in
  `final_prompt`, the node reuses it verbatim straight into the sampler: **the LLM is not called
  and the model is not even loaded**. Switch to `randomize` (or clear the box) to generate a fresh
  one. Iterate on the image at zero LLM cost.
- **Built-in CLIP Text Encode** — turn `llm_enabled` off and the node just encodes your
  `final_prompt` text as plain positive/negative CONDITIONING.
- **Low-VRAM friendly** — `vram_limit`, `n_cpu_moe` (MoE expert offload), KV-cache
  quantization, an optional `force_offload`, and it **frees the diffusion model from VRAM before
  loading the LLM** so llama.cpp doesn't spill to shared system RAM (much faster on 12 GB cards).

## Features

- Dual reference images (`image_1`, `image_2`) with **composite** (one prompt) or
  **batch** (one caption per image — dataset captioning) modes.
- `.txt` **system presets** read from `models/LLM/prompts/` + **instruction presets**
  (Describe / Tags / Cinematic / Replace subject / Appearance only / …). Both auto-fill an
  editable box so you see and can tweak the text live.
- A free `user_preset` field for ad-hoc instructions (not written to any file).
- Editable **`final_prompt`** box — the LLM writes its result there (copy or hand-edit it); with the
  LLM off it becomes your manual prompt box. A `fixed` seed **freezes** it (reuse, no LLM run, no
  model load); `randomize` regenerates.
- `prefix` / `suffix` — e.g. auto-prepend a LoRA trigger word before CLIP encode (applied in both
  LLM-on and LLM-off modes).
- Positive **and** negative CONDITIONING outputs.
- Reasoning-model support (`<think>` blocks stripped by default).
- Progress bar for batch runs; image pass-through outputs; optional `queue` chain input.

## Also a plain CLIP Text Encode (no learning curve)

New to this and just want a prompt box? Turn **`⚡ LLM enabled` off**. The node skips the model and
encodes **only your `final_prompt`** to CLIP — type your positive prompt straight into that box, put
your negative in `negative`, and with `clip` connected the `positive` / `negative` outputs are
ready-to-use CONDITIONING, exactly like the CLIP Text Encode you already know.

With the LLM off, the LLM-only fields (`system_prompt`, `instruction`, `user_preset`) grey out and
are ignored — they never leak into your prompt. `prefix` / `suffix` still apply if set.

![Turn the LLM off, then type your prompt straight into the final_prompt box — a plain CLIP Text Encode](img/clip_mode.png)

*Toggle `⚡ LLM enabled` off, then write your prompt into `final_prompt` — the node encodes it to
`positive` / `negative` just like a CLIP Text Encode.*

Prefer feeding the prompt from an external node? Right-click the node →
**Convert final_prompt to input** (and/or `negative`) and wire any text / note node into it.
Both ways work — type in the box, or drive it from outside, whatever you're used to.

## Example: compose across two images

Set `mode = composite`, connect both `image_1` and `image_2`, and refer to the two
references as **Image 1** and **Image 2** in the instruction. For example — put the subject
from one reference into the scene of the other:

> Take the same girl as in reference Image 1 and place her, in the exact same setting as
> reference Image 2, sitting in a summer street café taking a phone selfie with an outstretched
> arm and smiling. Write a detailed generation prompt, like the one you'd write for Image 2, but
> with the girl from Image 1.

The node feeds both images to the model (`image_1` → "Image 1", `image_2` → "Image 2"), so the
generated prompt keeps Image 2's setting with Image 1's subject. Short, concrete instructions that
name the images explicitly work best.

> **Two-image compositing depends on the model.** It needs a VLM that accepts *multiple* images —
> the **Qwen-VL / Qwen3.5** family and **MiniCPM-V** handle it well. Single-image models (LLaVA,
> Moondream) effectively see only one. If the second image seems ignored, switch to a multi-image
> model. Single-image captioning works on any VLM.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/artfat-creator/artfat-comfyui-llm-prompter.git
python -m pip install -r artfat-comfyui-llm-prompter/requirements.txt
```

`requirements.txt` pins the [JamePeng llama-cpp-python](https://github.com/JamePeng/llama-cpp-python)
build (ships the VLM chat handlers). Pick the wheel that matches your Python/OS if pip
does not auto-select it.

## Models

Put GGUF files in `ComfyUI/models/LLM/`. For image input (VLM) also download the matching
`mmproj-*.gguf` from the same repo into that folder, and pick the `chat_handler` that matches the
model family.

> **For image input you need a vision model (VLM) — not a plain text LLM.** Look for **“VL”,
> “Vision”, or “multimodal”** in the model name and an **`mmproj-*.gguf`** file in the repo — that
> projector is what lets the model see the image. Without it the model is text-only (still fine for
> rewriting/expanding a text prompt, just no image understanding).

### Recommended models

| Model (GGUF) | `chat_handler` | Notes |
|---|---|---|
| [DavidAU Qwen3.5-9B — Claude-4.6 HERETIC Thinking](https://huggingface.co/DavidAU/Qwen3.5-9B-Claude-4.6-OS-Auto-Variable-HERETIC-UNCENSORED-THINKING-MAX-NEOCODE-Imatrix-GGUF) | `Qwen3.5-Thinking` | vision (mmproj), uncensored, reasons then answers — the node strips the reasoning so only the prompt reaches CLIP |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) · [4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) | `Qwen3-VL` | official vision-language, lighter / faster |
| [unsloth Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | `Qwen3.5` | general Qwen3.5 build |
| [Gemma 3 4B](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF) | `Gemma3` | **lightest vision model** — grab its `mmproj`; great for low VRAM. Turn `force_offload = on` so it unloads after each prompt (Gemma 3 1B / 270M are text-only, no images) |

Any VLM whose family is in the `chat_handler` dropdown works — MiniCPM-V, GLM-4.x-V, Gemma 3,
LLaVA, and more. For a `-Thinking` handler the model reasons before answering and the node keeps
only the final prompt.

System-prompt presets are plain `.txt` files in `ComfyUI/models/LLM/prompts/`.

## Inputs

The essentials stay visible; the sampler / KV-cache / image-token knobs tuck behind the
collapsible **▸ advanced** section — compact by default, everything on demand.

![Compact by default, advanced collapsed](img/node_collapsed.png)

![Advanced expanded — full sampler and image controls](img/node.png)

| Input | What it does |
|---|---|
| `model` / `mmproj` | GGUF LLM + optional vision projector |
| `chat_handler` | Chat template — must match the model |
| `n_ctx` | Context length (default 8192) |
| `vram_limit` | VRAM budget in GB for the LLM (-1 = all layers on GPU) |
| `n_cpu_moe` | Keep MoE experts of the first N layers on CPU (frees VRAM) |
| `llm_enabled` | Off = skip the LLM and encode **only** `final_prompt` (the LLM-only fields below grey out and are ignored) |
| `system_preset` / `system_prompt` | Style/engine (.txt preset auto-fills the editable box) — LLM only |
| `instruction_preset` / `instruction` | Task / output format (preset auto-fills the box) — LLM only |
| `user_preset` | Extra ad-hoc instructions, appended — LLM only |
| `negative` | Negative prompt → negative output |
| `final_prompt` | The generated prompt (editable, copyable). **LLM on:** the model writes its result here. **LLM off:** type your own prompt here — it's what gets encoded to CLIP |
| `prefix` / `suffix` | Wrap the final prompt (trigger words, quality tags) — applied in both LLM on and off |
| `mode` | `composite` (images → one prompt) or `batch` (each image → its caption) |
| `seed` | Prompt seed. **`control_after_generate = fixed`** freezes `final_prompt` (reuse it, no LLM run, no model load); **`randomize`** generates a fresh prompt each queue |
| `force_offload` | Unload the LLM after running (frees VRAM; next call reloads) |
| advanced | `max_tokens`, `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`, `repeat_penalty`, `frequency_penalty`, `mirostat_*`, `type_k`/`type_v` (KV quant), `max_size`, `image_min/max_tokens` |
| `image_1` / `image_2` / `clip` / `queue` | optional |

## Outputs

`positive` (CONDITIONING) · `negative` (CONDITIONING) · `prompt` (STRING) ·
`prompt_list` (STRING list, for batch) · `image_1` / `image_2` (pass-through) · `queue` ·
`clip` (pass-through, so a bypassed node still routes CLIP downstream)

## Seed = freeze / regenerate

The seed's `control_after_generate` is the switch between **reusing** and **regenerating** the prompt:

- **`fixed`** + a prompt already in `final_prompt` → the node **reuses it verbatim** straight to
  CLIP → the sampler. The LLM is **not called** and the model is **not loaded**, so you can re-run
  the sampler (or tweak other nodes) at zero LLM cost. Hand-edit `final_prompt` and your edit is
  what's used.
- **`randomize`** / `increment` / `decrement`, **or an empty** `final_prompt` → the LLM runs and
  writes a fresh prompt into the box.
- Seed only varies the *generated* text when `temperature > 0`.

> How it knows: `control_after_generate` is a front-end setting the Python side can't read directly,
> so the node's web script mirrors "is it `fixed`?" into a hidden `freeze` flag at queue time — no
> seed guessing, no one-run lag.

## VRAM / offload

Keep `force_offload` **off** to keep the LLM resident (fast repeat prompting). Turn it **on**
only on low-VRAM systems to free VRAM for diffusion — the trade-off is a model reload on the
next LLM call. The node also frees the LLM when ComfyUI unloads all models, **and evicts the
diffusion model from VRAM before loading the LLM** — otherwise llama.cpp's GPU layers spill into
shared system RAM and load/inference crawl (the sampler reloads the diffusion model on its next run).

### By VRAM (LLM running next to a diffusion model)

| GPU VRAM | Suggested setup |
|---|---|
| **8 GB** | A lighter model ([Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)) or a Q4 9B with `force_offload = true` so the LLM unloads before sampling. Set `type_k` / `type_v = q8_0` and a few `n_cpu_moe` layers. |
| **12 GB** (shared) | 9B at Q4_K_M. Either `force_offload = true` for heavy diffusion models, or keep it resident and raise `n_cpu_moe` to push MoE experts to CPU. Use `vram_limit` to cap the LLM's share. |
| **16 GB** | 9B at Q5 / Q6 kept resident (`force_offload = false`) → instant repeat prompting. |
| **24 GB+** | 9B or larger at Q6 / Q8 fully on GPU, resident, comfortably beside diffusion. |

Running the LLM on its own (no diffusion in the same graph)? Everything fits far more easily —
keep it resident and enjoy near-instant prompts.

## Power tips

- **Any field can become an input.** Right-click the node → *Convert … to input* (or just drag a
  wire onto the widget) for `final_prompt`, `system_prompt`, `instruction`, `user_preset`, `negative`,
  `prefix`, `suffix`, or even numeric fields like `seed` — then drive them from external text / primitive nodes.
- **Let the model think, keep the output clean.** Pick a `-Thinking` chat handler (e.g.
  `Qwen3.5-Thinking`) for better prompts; the node always strips the reasoning so only the final
  prompt reaches CLIP.
- **Batch a dataset.** Set `mode = batch`, feed a batch of images into `image_1`, and read
  `prompt_list` — one caption per image, ready for LoRA training.
- **Share VRAM with diffusion.** On tight VRAM turn `force_offload` on (frees the LLM after each run)
  and/or raise `n_cpu_moe` on MoE models. Keep `force_offload` off for instant repeat prompting.

## Credits

- [lihaoyun6/ComfyUI-llama-cpp](https://github.com/lihaoyun6/ComfyUI-llama-cpp) — engine design this builds on
- [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) — llama.cpp bindings with VLM handlers
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

MIT © artfat-creator
