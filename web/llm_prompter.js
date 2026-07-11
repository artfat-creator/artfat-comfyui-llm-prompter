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

            // Field heights: main prompt in + final prompt out = 2x, other text boxes = 1.5x.
            const setH = (w, h) => { if (w) w.computeSize = (width) => [width || 220, h]; };
            setH(widget(node, "system_prompt"), 105);
            setH(instr, 140);
            setH(widget(node, "user_preset"), 105);
            setH(widget(node, "negative"), 105);
            setH(fp, 140);

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
