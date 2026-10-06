/// <reference types="node" />
import { it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

// Execute the same worklet source loaded by the browser. Only the browser's
// processor/port boundary is simulated; queueing and rendering are real.
function worklet() {
	const messages: { type: string; samples?: number; finished?: boolean; state?: string }[] = [];
	interface Processor {
		port: { onmessage: (event: { data: unknown }) => void };
		process: (inputs: unknown[], outputs: Float32Array[][]) => boolean;
	}
	let ProcessorClass!: new (options: unknown) => Processor;
	const source = readFileSync(new URL("./use-tts-stream.ts", import.meta.url), "utf8")
		.split("const WORKLET_CODE = `")[1].split("`;", 1)[0];
	runInNewContext(source, {
		Float32Array,
		AudioWorkletProcessor: class {
			port = { postMessage: (message: typeof messages[number]) => messages.push(structuredClone(message)) };
		},
		registerProcessor: (_name: string, processor: typeof ProcessorClass) => { ProcessorClass = processor; },
	});
	const processor = new ProcessorClass({ processorOptions: { startThreshold: 256, resumeThreshold: 256 } });
	return {
		send: (data: unknown) => processor.port.onmessage({ data }),
		render: () => {
			const output = new Float32Array(128);
			const running = processor.process([], [[output]]);
			return { output, running };
		},
		clock: () => messages.filter(message => message.type === "playback").at(-1),
		state: () => messages.filter(message => message.type === "state").at(-1)?.state,
	};
}

it("advances the playback clock only for consumed PCM, never buffering silence", () => {
	const player = worklet();
	for (let i = 0; i < 20; i++) player.render();
	player.send({ type: "samples", samples: new Float32Array(384).fill(0.5) });
	for (let i = 0; i < 20; i++) player.render();
	assert.equal(player.clock()?.samples, 384);
	for (let i = 0; i < 40; i++) player.render();
	assert.equal(player.clock()?.samples, 384);
	player.send({ type: "samples", samples: new Float32Array(73).fill(0.25) });
	player.send({ type: "end" });
	const final = player.render();
	assert.equal(final.running, false);
	assert.equal(final.output[72], 0.25);
	assert.equal(final.output[73], 0);
	assert.deepEqual(player.clock(), { type: "playback", samples: 457, finished: true });
});

it("drains a finished clip shorter than the startup threshold", () => {
	const player = worklet();
	player.send({ type: "samples", samples: new Float32Array(61).fill(0.75) });
	assert.equal(player.render().output[0], 0);
	player.send({ type: "end" });
	const final = player.render();
	assert.equal(final.output[60], 0.75);
	assert.equal(final.output[61], 0);
	assert.equal(final.running, false);
	assert.equal(player.clock()?.samples, 61);
});

it("reports playing when audio resumes after an underrun", () => {
	const player = worklet();
	player.send({ type: "samples", samples: new Float32Array(384).fill(0.5) });
	for (let i = 0; i < 4; i++) player.render();
	assert.equal(player.state(), "buffering");
	player.send({ type: "samples", samples: new Float32Array(128).fill(0.25) });
	assert.equal(player.render().output[0], 0.25);
	assert.equal(player.state(), "playing");
});
