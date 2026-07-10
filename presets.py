"""Preset text for the Artfat LLM Prompter node.

Two layers:
  * system presets  -> HOW to write (engine / style). Read from models/LLM/prompts/*.txt
  * instruction presets -> WHAT to do (task / output format). Defined below.
Both auto-fill an editable widget on the node (web/llm_prompter.js), so the
chosen text is visible and can be edited live before running.
"""

import os

import folder_paths

PROMPTS_DIR = os.path.join(folder_paths.models_dir, "LLM", "prompts")

# name -> instruction text ("Custom" leaves the box empty for a free / user preset)
INSTRUCTION_PRESETS = {
    "Custom": "",
    "Describe -> prompt": (
        "Look at the image and write a single detailed text-to-image prompt that "
        "faithfully describes it: subject, setting, composition, lighting and mood. "
        "Output only the prompt as one flowing paragraph, no preamble, no analysis."
    ),
    "Extreme detailed": (
        "Write an extremely detailed text-to-image prompt from the image. Elaborate on "
        "the subject's appearance, clothing textures, specific background elements, the "
        "quality and colour of light, shadows and overall atmosphere. One rich paragraph. "
        "Output only the prompt."
    ),
    "Tags": (
        "Generate a clean list of comma-separated tags for a text-to-image model based only "
        "on the visible content of the image: subject, clothing, environment, colours, lighting, "
        "composition. Max 50 unique tags, no abstract or marketing terms. Output only the tags."
    ),
    "Cinematic": (
        "Act as a master prompt engineer. Write a highly detailed, evocative cinematic prompt for "
        "an image-generation model: subject, pose, environment, lighting, mood and photographic "
        "style. Weave everything into one natural-language paragraph. Output only the prompt."
    ),
    "Refine & expand": (
        "Refine and enhance the following prompt for text-to-image generation. Keep its meaning "
        "and key words, make it more expressive and visually rich. Output only the improved prompt "
        "text itself, with no reasoning, thinking or commentary."
    ),
    "Replace subject (Image1 scene + Image2 person)": (
        "Generate a detailed prompt describing the reference Image 1. Replace the person from "
        "Image 1 with the person from Image 2, adapting the pose naturally to the scene. Output "
        "only the final prompt as one paragraph."
    ),
    "Replace + keep outfit from Image1": (
        "Generate a detailed prompt describing the reference Image 1. Replace the person from "
        "Image 1 with the person from Image 2, but she should be dressed exactly the same as in "
        "Image 1. Output only the final prompt as one paragraph."
    ),
    "Appearance only (face + body)": (
        "Describe only the appearance of the person: face first, then body. No clothing, "
        "accessories, environment, pose or action unless explicitly told. If two images are given, "
        "blend the features into one coherent person. Output only the description."
    ),
    "Scene + lighting mix": (
        "Combine the architecture and composition of Image 1 with the time-of-day lighting and "
        "colour mood of Image 2 into a single text-to-image prompt. Output only the prompt."
    ),
}

INSTRUCTION_NAMES = list(INSTRUCTION_PRESETS.keys())


def list_system_presets():
    """Dropdown values: 'Custom' plus every .txt in models/LLM/prompts/."""
    names = ["Custom"]
    try:
        if os.path.isdir(PROMPTS_DIR):
            for fn in sorted(os.listdir(PROMPTS_DIR)):
                if fn.lower().endswith(".txt"):
                    names.append(fn)
    except Exception as e:
        print(f"[llm-prompter] Could not list system presets: {e}")
    return names


def load_system_preset(name):
    """Return the text of a .txt system preset, or None for 'Custom'/missing."""
    if not name or name in ("Custom", "None"):
        return None
    path = os.path.join(PROMPTS_DIR, name)
    try:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                return fh.read()
    except Exception as e:
        print(f"[llm-prompter] Could not read preset {name}: {e}")
    return None
