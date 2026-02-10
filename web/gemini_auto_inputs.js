import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.GeminiImageGenerate.AutoExpandImages",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "GeminiImageGenerate") return;

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, link_info, ioSlot) {
            onConnectionsChange?.apply(this, arguments);
            if (type !== LiteGraph.INPUT) return;
            if (this._updatingInputs) return;

            requestAnimationFrame(() => {
                if (this._updatingInputs) return;
                this._updatingInputs = true;

                // Remove unconnected image_ inputs from end
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    if (this.inputs[i].name.startsWith("image_") && !this.inputs[i].link) {
                        this.removeInput(i);
                    }
                }

                // Renumber remaining image_ inputs sequentially
                let num = 2;
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i].name.startsWith("image_")) {
                        this.inputs[i].name = `image_${num}`;
                        num++;
                    }
                }

                // Add one spare image input at the end
                this.addInput(`image_${num}`, "IMAGE");

                this.setSize(this.computeSize());
                app.graph.setDirtyCanvas(true);
                this._updatingInputs = false;
            });
        };

        // Add initial spare input when node is created
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.addInput("image_2", "IMAGE");
            this.setSize(this.computeSize());
        };
    },
});
