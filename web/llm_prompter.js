import { app } from "../../scripts/app.js";

const ADVANCED = [
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

            // highlight the LLM on/off switch
            const en = widget(node, "llm_enabled");
            if (en) en.label = "⚡ LLM enabled";

            bindAutofill(node, "system_preset", "system_prompt", "/artfat_llm/system_preset");
            bindAutofill(node, "instruction_preset", "instruction", "/artfat_llm/instruction_preset");

            setTimeout(relayout, 20);
        };

        // show the final prompt on the node after it runs
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted && onExecuted.apply(this, arguments);
            if (!message || !message.text) return;
            const text = Array.isArray(message.text) ? message.text.join("\n\n") : message.text;
            let w = widget(this, "_final_prompt");
            if (!w) {
                w = this.addWidget("text", "_final_prompt", "", () => {}, { multiline: true });
                w.inputEl && (w.inputEl.readOnly = true);
                w.serialize = false;
            }
            w.value = text;
            this.setDirtyCanvas(true, true);
        };
    },
});
