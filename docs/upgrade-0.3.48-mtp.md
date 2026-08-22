# Upgrade notes: llama-cpp-python 0.3.44 -> 0.3.48 (MTP)

Working notes for a future release. Nothing here is shipped yet — the node still
pins 0.3.44 and does not expose any MTP option.

Verified on 2026-08-22, one machine only: Windows 11, RTX 3090 24 GB, portable
ComfyUI, Python 3.13.12, torch 2.13.0+cu130, wheel
`llama_cpp_python-0.3.48+cu130-cp313-cp313-win_amd64.whl` (JamePeng release
`v0.3.48-cu130-win-20260821`).

## Why 0.3.48 matters

0.3.44 has no MTP driver at the Python level. `llama.dll` exports the NextN
functions (`llama_set_embeddings_nextn`, `llama_set_nextn_layer_offset`,
`llama_get_embeddings_nextn`) and `llama_cpp.py` has the ctypes wrappers, but no
`.py` in the package ever calls them. Speculative decoding in 0.3.44 goes only
through `draft_model`, which is n-gram / prompt-lookup — model-free, unrelated to
MTP. So an `-mtp.gguf` file loads and runs, but the MTP tensors are dead weight.

0.3.48 adds "Stateful MTP Speculative Decoding":

- `SpecConfig`, `SpeculativeType`, `LlamaSpecEngine`, `LlamaMTPDecoding`
- `create_spec_engine`, `create_native_spec_engine`
- `Llama.__init__` gains `load_mtp`, `ctx_type`, `speculative` (also `load_mode`,
  `no_alloc` from 0.3.45) — 70 parameters total, up from 65
- upstream changelog reports validation on Qwen3.5 / Qwen3.6 / Qwen3.8 with MTP,
  suggesting `draft_n_max=2` as a starting point for Qwen3.8 27B

Breaking change in 0.3.48: `LlamaPromptLookupDecoding` was removed; NGram mode now
goes through `SpecConfig`. This node never used it, so nothing breaks here.

## What was measured

- CUDA: `found 1 CUDA devices (Total VRAM: 24575 MiB)`, `offloaded 33/33 layers to
  GPU`, node GPU detector reports `ok (13, 13)`
- vision intact: `Qwen35ChatHandler` exposes `mmproj_path`, so the
  `mmproj_path or clip_model_path` check in `nodes.py` still passes. (This is the
  exact thing that broke on the 0.3.40 -> 0.3.44 jump.)
- hybrid DeltaNet path produces coherent output, not garbage: Qwen3.5-4B Q8_0
  loads as `arch=qwen35` with `ssm.*` metadata and `full_attention_interval=4`;
  the KV cache is allocated only on layers 3/7/11/.../31, the other layers are
  Gated DeltaNet with no KV. Same family as Qwen3.8-27B.
- `graph_mtp` coverage grew. 0.3.44 had cohere2moe, hy_v3, qwen35, qwen35moe,
  step35. 0.3.48 adds bailingmoe3, deepseek2, deepseek32, deepseek4, glm_dsa,
  mimo2, nemotron_h_moe, qwen3next.
- live in ComfyUI (2026-08-22), confirmed by the author on the machine described
  above: the node registers, loads and returns a generated prompt on 0.3.48, both
  text-only and with two images connected (composite mode). The vision path is the
  one that broke on the 0.3.40 -> 0.3.44 jump, so this is the check that matters.
- the list of qwen architecture strings did NOT change. Qwen3.8 registers as
  `qwen35` upstream, so "arch not in the list" is not the failure mode to expect.

## MTP measured (2026-08-22)

Model: `RVN-Q4_K_M-multilingual-mtp.gguf`, 15.83 GiB, from
`Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF`.

GGUF metadata confirms the detection path works: `general.architecture = qwen35`,
`qwen35.nextn_predict_layers = 1`, four NextN tensors in `blk.64` (65 blocks total,
64 trunk + 1 NextN head). The key name guessed earlier was correct, so a pre-load
check on `<arch>.nextn_predict_layers` is viable.

Enabled with `Llama(..., load_mtp=True, speculative=SpecConfig(spec_type=DRAFT_MTP,
draft_n_max=2))`. Runtime confirms: `MTP speculative decoding enabled`,
`type=draft-mtp, model='target-internal-heads'`, `mtp_heads=1`, `n_layer_nextn=1`.

A/B on one machine, same seed, same prompt, warmup before timing, ~250 tokens each:

| | tokens | sec | tok/s |
|---|---|---|---|
| plain | 249 | 14.13 | 17.62 |
| MTP | 250 | 7.46 | 33.51 |

**+90%.** VRAM cost: CUDA0 buffer 15088 -> 15518 MiB, i.e. **+430 MiB** — matching the
0.42 GiB the model card quotes.

The DeltaNet CUDA bug from ggml-org/llama.cpp discussion #27164 did NOT reproduce on
the 27B hybrid: output is coherent, not garbage.

## Live confirmation (2026-08-22)

Confirmed by the author in a real ComfyUI graph after the vision gate landed: the node
loads, generates, and the MTP toggle can be left on permanently. With an mmproj wired
the run proceeds normally and prints why MTP stood down; text-only runs take the
speedup. No crash in either mode.

## What was NOT measured

- one prompt, one workload type. MTP gains scale with how predictable the text is,
  so other workloads will differ. Do not quote 90% as a general figure.
- no sweep across `draft_n_max` values; 2 was used throughout.
- one machine, one CUDA version. cu124/126/128/131 untested.

## Regression found

`MiniCPM-v4.6` and `MiniCPM-v4.6-Thinking` disappear from `CHAT_HANDLERS` on
0.3.48. The class was renamed `MiniCPMv46ChatHandler` -> `MiniCPMV46ChatHandler`
(capital `V`), and `_try_add` in `llama_core.py` swallows the failure silently. A
saved workflow with that handler selected will raise "Value not in list".

Handlers present in 0.3.48 that this node does not offer yet:
`Step3VLChatHandler`, `Qwen3ASRChatHandler`, `PaddleOCRChatHandler`,
`GraniteDoclingChatHandler`, `GenericMTMDChatHandler`, `ObsidianChatHandler`.

## Testing trap

Running `python -c "from llama_cpp import Llama"` without importing torch first
reports `backend registry count: 1` and assigns every layer to CPU. That is not a
CPU build — `cudart64_13.dll` ships inside `torch/lib`, and without importing
torch it is not on the DLL search path, so `ggml-cuda.dll` fails to load. ComfyUI
imports torch first, so it is unaffected. Import torch before llama_cpp in any
standalone check.

## TODO before this can ship

1. `install.py` pins `_LLAMA_VER = "0.3.44"`. Until it is bumped, running
   install.py (or a Manager reinstall) rolls the wheel back to 0.3.44.
2. Wire `load_mtp` and a spec config through `LLMEngine._load` into `Llama(...)`,
   and add the widgets at the END of `INPUT_TYPES` — ComfyUI stores widget values
   positionally, so inserting anywhere else shifts saved nodes.
3. Fix the `MiniCPMV46ChatHandler` name.
4. Benchmark MTP on a real Qwen3.8-27B MTP GGUF before claiming any speedup, and
   keep the plain file as the documented default.
