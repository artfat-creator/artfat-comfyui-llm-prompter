import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

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

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated && onCreated.apply(this, arguments);
            const node = this;
            node._advOpen = false;

            const disp = ComfyWidgets["STRING"](
                node, "final_prompt", ["STRING", { multiline: true }], app
            ).widget;
            if (disp.inputEl) {
                disp.inputEl.readOnly = true;
                disp.inputEl.style.opacity = "0.85";
                disp.inputEl.placeholder = "final prompt appears here after Queue";
            }
            disp.serialize = false;
            node._disp = disp;

            const en = widget(node, "llm_enabled");
            if (en) en.label = "⚡ LLM enabled";

            // The instruction box announces its role: task for the LLM when enabled,
            // or a plain positive-prompt box (CLIP Text Encode style) when disabled.
            const instr = widget(node, "instruction");
            const updateInstrHint = () => {
                if (!instr || !instr.inputEl) return;
                const off = en && en.value === false;
                instr.inputEl.placeholder = off
                    ? "▶ POSITIVE PROMPT — type it here (LLM off = plain CLIP Text Encode)"
                    : "task / instruction for the LLM (auto-filled from preset, editable)";
                node.setDirtyCanvas(true, true);
            };
            if (en) {
                const oc = en.callback;
                en.callback = (v) => {
                    if (oc) oc.call(en, v);
                    updateInstrHint();
                };
            }
            updateInstrHint();

            // Field heights via computeSize (litegraph reserves the space, no overlap):
            // main prompt in + final prompt out = 2x, the other text boxes = 1.5x.
            const setH = (w, h) => { if (w) w.computeSize = (width) => [width || 220, h]; };
            setH(widget(node, "system_prompt"), 105);
            setH(instr, 140);
            setH(widget(node, "user_preset"), 105);
            setH(widget(node, "negative"), 105);
            setH(disp, 140);

            const pfx = widget(node, "prefix");
            if (pfx) pfx.label = "prefix: trigger word";
            const sfx = widget(node, "suffix");
            if (sfx) sfx.label = "suffix: quality tags";

            const relayout = () => {
                for (const n of ADVANCED) {
                    const w = widget(node, n);
                    node._advOpen ? show(w) : hide(w);
                }
                const sz = node.computeSize();
                node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
                node.setDirtyCanvas(true, true);
            };

            const btn = node.addWidget("button", "▸ advanced", null, () => {
                node._advOpen = !node._advOpen;
                btn.name = (node._advOpen ? "▾" : "▸") + " advanced";
                relayout();
            });

            // Position ONLY the non-serialized helpers (disp, btn). Real widgets keep
            // their INPUT_TYPES order, so saved values never shift.
            // The advanced button is placed LAST so no hidden widget is ever the
            // terminal widget (which litegraph renders as a stray hanging row).
            const advSet = new Set(ADVANCED);
            const reals = node.widgets.filter((w) => w !== disp && w !== btn);
            const head = [];
            const adv = [];
            for (const w of reals) {
                if (advSet.has(w.name)) {
                    adv.push(w);
                    continue;
                }
                head.push(w);
                if (w.name === "negative") head.push(disp);
            }
            node.widgets = [...head, ...adv, btn];

            bindAutofill(node, "system_preset", "system_prompt", "/artfat_llm/system_preset");
            bindAutofill(node, "instruction_preset", "instruction", "/artfat_llm/instruction_preset");

            setTimeout(relayout, 20);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted && onExecuted.apply(this, arguments);
            if (!message || !message.text) return;
            const text = Array.isArray(message.text) ? message.text.join("\n\n") : message.text;
            if (this._disp) {
                this._disp.value = text;
                this.setDirtyCanvas(true, true);
            }
        };
    },
});
