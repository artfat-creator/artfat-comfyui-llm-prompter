import { app } from "../../scripts/app.js";

const ADVANCED = [
    "max_tokens", "temperature",
    "top_k", "top_p", "min_p", "typical_p", "repeat_penalty", "frequency_penalty",
    "mirostat_mode", "mirostat_tau", "mirostat_eta", "type_k", "type_v",
    "max_size", "image_min_tokens", "image_max_tokens",
];

const HIDDEN = "artfat_hidden";

function widget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function hide(w) {
    if (!w || w.type === HIDDEN) return;
    w._origType = w.type;
    w._origCompute = w.computeSize;
    w.type = HIDDEN;
    w.computeSize = () => [0, -4];
}

function show(w) {
    if (!w || w.type !== HIDDEN) return;
    w.type = w._origType;
    w.computeSize = w._origCompute;
}

async function presetText(route, name) {
    try {
        const r = await fetch(`${route}?name=${encodeURIComponent(name)}`);
        const j = await r.json();
        return j.text || "";
    } catch (e) {
        return "";
    }
}

function bindAutofill(node, presetName, targetName, route) {
    const p = widget(node, presetName);
    const t = widget(node, targetName);
    if (!p || !t) return;
    const orig = p.callback;
    p.callback = async (v) => {
        if (orig) orig.call(p, v);
        if (v && v !== "Custom" && v !== "None") {
            t.value = await presetText(route, v);
            node.setDirtyCanvas(true, true);
        }
    };
}

app.registerExtension({
    name: "artfat.llm.prompter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ArtfatLLMPrompter") return;

        // NOTE: final_prompt is now a real Python INPUT_TYPES field (see nodes.py). It lives in
        // the frozen Python widget order, so it can never shift the positional widgets_values
        // list — the drift bug is impossible by construction. This file no longer creates,
        // moves, or serialize-flags any widget; it only tweaks labels/heights and writes the
        // LLM result back into the existing final_prompt field on execution.

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated && onCreated.apply(this, arguments);
            const node = this;
            node._advOpen = false;

            const en = widget(node, "llm_enabled");
            if (en) en.label = "⚡ LLM enabled";

            // Hidden `freeze` flag -> backend. It mirrors the seed's control_after_generate == "fixed".
            // control_after_generate.value is the user's real, STABLE choice (the seed itself changes
            // on randomize, so it is not a reliable signal). serializeValue recomputes at Queue time,
            // so the backend always gets the current fixed/not-fixed state with no one-run lag.
            const freezeW = widget(node, "freeze");
            if (freezeW) {
                hide(freezeW);
                freezeW.serializeValue = () => {
                    const c = widget(node, "control_after_generate");
                    return !!(c && String(c.value).toLowerCase() === "fixed");
                };
            }

            const instr = widget(node, "instruction");
            // LLM-only text fields: inert (greyed + non-editable) while the LLM is off, because in
            // that mode ONLY final_prompt is encoded (see nodes.py). Prevents typing into fields that
            // do nothing and makes it visually obvious they are disabled.
            const LLM_ONLY = ["system_prompt", "instruction", "user_preset"];
            const refreshEnabled = () => {
                const off = en && en.value === false;
                if (instr && instr.inputEl) {
                    instr.inputEl.placeholder = off
                        ? "task for the LLM (ignored while LLM is off — type your prompt in 'final prompt' below)"
                        : "task / instruction for the LLM (auto-filled from preset, editable)";
                }
                for (const nm of LLM_ONLY) {
                    const w = widget(node, nm);
                    if (w && w.inputEl) {
                        w.inputEl.disabled = off;
                        w.inputEl.style.opacity = off ? "0.4" : "";
                    }
                }
                node.setDirtyCanvas(true, true);
            };
            if (en) {
                const oc = en.callback;
                en.callback = (v) => { if (oc) oc.call(en, v); refreshEnabled(); };
            }
            // inputEl exists only after the DOM widgets mount — refresh now and shortly after.
            refreshEnabled();
            setTimeout(refreshEnabled, 30);

            const prefix = widget(node, "prefix");
            if (prefix) prefix.label = "prefix: trigger word";
            const suffix = widget(node, "suffix");
            if (suffix) suffix.label = "suffix: quality tags";

            const fp = widget(node, "final_prompt");
            if (fp) fp.label = "final prompt (LLM output / manual input)";

            // --- Elastic multiline heights (freely resizable BOTH ways, and no "jump") -----------
            // All five text areas SHARE the node's free vertical space by weight, and each one's
            // computeSize is derived from the LIVE node.size so that node.computeSize()[1] ALWAYS
            // equals node.size[1]. Two consequences:
            //   * No jump — ComfyUI resets node.size -> node.computeSize() on window focus / redraw;
            //     since they are equal, that reset is a no-op no matter what event triggers it.
            //   * Fully elastic — drag the node taller and the fields grow; drag it shorter and they
            //     shrink down to MIN, so you can size the node to whatever you want (both directions).
            // The recursion guard lets the nested node.computeSize() measure the NON-text widgets
            // while the text areas report their MIN, so the split is exact (zero px drift).
            // Knobs: WEIGHT = how the free space is split (final_prompt gets the most); MIN = the
            // smallest each field may shrink to (also sets the node's minimum height).
            const ML = ["system_prompt", "instruction", "user_preset", "negative", "final_prompt"];
            const WEIGHT = { system_prompt: 1, instruction: 1, user_preset: 1, negative: 1, final_prompt: 4 };
            const MIN = { system_prompt: 34, instruction: 34, user_preset: 34, negative: 34, final_prompt: 60 };
            const SUMMIN = ML.reduce((a, n) => a + MIN[n], 0);
            const SUMW = ML.reduce((a, n) => a + WEIGHT[n], 0);
            let inCompute = false;
            const mlSize = (w, width) => {
                if (inCompute) return [width, MIN[w.name]];             // nested measure: report MIN
                inCompute = true;
                let total;
                try { total = node.computeSize()[1]; } finally { inCompute = false; }
                const nonML = total - SUMMIN;                          // everything except the text areas
                const free = Math.max(0, (node.size ? node.size[1] : total) - nonML - SUMMIN);
                return [width, MIN[w.name] + free * (WEIGHT[w.name] / SUMW)];
            };
            ML.forEach((nm) => { const w = widget(node, nm); if (w) w.computeSize = (width) => mlSize(w, width); });

            // Spawn a freshly-dragged node at a comfortable height (final_prompt big by default). A node
            // loaded from a saved workflow keeps its saved size — it is restored AFTER onNodeCreated,
            // so this only affects new nodes; the user can still drag it to any size afterwards.
            setTimeout(() => {
                const want = 1000;
                if (node.size[1] < want) node.setSize([node.size[0], want]);
            }, 25);

            // Advanced toggle — appended LAST.
            const btn = node.addWidget("button", "▸ advanced", null, () => {
                node._advOpen = !node._advOpen;
                btn.name = (node._advOpen ? "▾" : "▸") + " advanced";
                relayout();
            });
            btn.serialize = false;

            const relayout = () => {
                for (const n of ADVANCED) {
                    const w = widget(node, n);
                    node._advOpen ? show(w) : hide(w);
                }
                const sz = node.computeSize();
                node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
                node.setDirtyCanvas(true, true);
            };

            bindAutofill(node, "system_preset", "system_prompt", "/artfat_llm/system_preset");
            bindAutofill(node, "instruction_preset", "instruction", "/artfat_llm/instruction_preset");

            setTimeout(relayout, 20);
        };

        // Write the generated prompt back into the (real, Python) final_prompt field.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted && onExecuted.apply(this, arguments);
            if (!message || !message.text) return;
            const text = Array.isArray(message.text) ? message.text.join("\n\n") : message.text;
            const w = widget(this, "final_prompt");
            if (w) {
                w.value = text;
                this.setDirtyCanvas(true, true);
            }
        };
    },
});
