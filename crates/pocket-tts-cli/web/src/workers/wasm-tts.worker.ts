/// <reference lib="webworker" />

import type {
	WasmWorkerEvent,
	WasmWorkerRequest,
	WasmWorkerStatus,
	WasmWorkerVoiceInput,
} from "./wasm-tts-protocol";

declare const self: DedicatedWorkerGlobalScope;

import { configAssets, presetVoiceUrl, selectConfig } from "./model-config";

const rawConfigs = import.meta.glob<string>("../../../../pocket-tts/config/*.yaml", {
	query: "?raw", import: "default", eager: true,
});
const configs = Object.fromEntries(Object.entries(rawConfigs).map(([path, yaml]) => [
	path.substring(path.lastIndexOf("/") + 1, path.length - 5), yaml,
]));
let currentLanguage = "english";
let stockVoicesCompatible = true;

const PRESET_VOICES = [
	"alba",
	"anna",
	"azelma",
	"bill_boerst",
	"caro_davy",
	"charles",
	"cosette",
	"daan",
	"eponine",
	"estelle",
	"eve",
	"fantine",
	"george",
	"giovanni",
	"jane",
	"javert",
	"jean",
	"juergen",
	"lola",
	"marius",
	"mary",
	"michael",
	"paul",
	"peter_yearsley",
	"rafael",
	"stuart_bell",
	"vera",
] as const;

const encoder = new TextEncoder();

interface WasmChunkStats {
	samples?: number;
	compute_ms?: number;
	chunks_merged?: number;
}

interface WasmStreamLike {
	next_chunk_min_samples(minSamples: number): Float32Array | null | undefined;
	last_chunk_stats(): WasmChunkStats;
}

interface WasmModelLike {
	load_from_buffer(
		config: Uint8Array,
		weights: Uint8Array,
		tokenizer: Uint8Array,
		hasVoiceCloning?: boolean,
	): void;
	is_ready(): boolean;
	start_stream(text: string): WasmStreamLike;
	load_voice_from_buffer(wavBytes: Uint8Array): void;
	load_voice_from_safetensors(bytes: Uint8Array): void;
	readonly sample_rate: number;
}

interface WasmBindings {
	default: () => Promise<void>;
	WasmTTSModel: new () => WasmModelLike;
}

let bindings: WasmBindings | null = null;
let model: WasmModelLike | null = null;
let sampleRate = 24000;
let stopRequested = false;
let activeStreamToken = 0;

const postEvent = (event: WasmWorkerEvent, transfer: Transferable[] = []) => {
	self.postMessage(event, transfer);
};

const postStatus = (status: WasmWorkerStatus) => {
	postEvent({ kind: "status", status });
};

const postOk = (requestId: number, payload?: { sampleRate?: number }) => {
	postEvent({ kind: "rpc_ok", requestId, payload });
};

const postErr = (requestId: number, err: unknown) => {
	const message = err instanceof Error ? err.message : String(err);
	postEvent({ kind: "rpc_err", requestId, error: message });
};

const sleep = (ms: number) =>
	new Promise<void>((resolve) => setTimeout(resolve, ms));

const isPresetVoice = (
	voice: string,
): voice is (typeof PRESET_VOICES)[number] => {
	return (PRESET_VOICES as readonly string[]).includes(voice);
};

const ensureReadyModel = (): WasmModelLike => {
	if (!model?.is_ready()) {
		throw new Error("WASM model is not initialized yet.");
	}
	return model;
};

// Bump version to invalidate any stale cache from pre-fix builds
const CACHE_NAME = "pocket-tts-models-v3";

const fetchHF = async (
	url: string,
	hfToken: string,
	label: string,
): Promise<Uint8Array> => {
	// Check browser Cache API first
	try {
		const cache = await caches.open(CACHE_NAME);
		const cached = await cache.match(url);
		if (cached) {
			console.log(`[wasm-worker] Cache hit for ${label}: ${url}`);
			return new Uint8Array(await cached.arrayBuffer());
		}
	} catch {
		// Cache API unavailable (e.g. opaque origin), fall through to fetch
	}

	console.log(`[wasm-worker] Downloading ${label}: ${url}`);
	const headers: Record<string, string> = {};
	if (hfToken.trim()) {
		headers.Authorization = `Bearer ${hfToken.trim()}`;
	}
	const res = await fetch(url, { headers });
	if (!res.ok) {
		if (res.status === 401) {
			throw new Error(`HF auth required for ${label}`);
		}
		throw new Error(`Failed to fetch ${label} (${res.status})`);
	}

	// Store in cache for next time (clone response since body can only be read once)
	try {
		const cache = await caches.open(CACHE_NAME);
		await cache.put(url, res.clone());
	} catch {
		// Cache write failed (quota, etc.) — not fatal
	}

	return new Uint8Array(await res.arrayBuffer());
};

const fetchEmbedding = async (
	voice: string,
	hfToken: string,
): Promise<Uint8Array> => {
	return fetchHF(presetVoiceUrl(currentLanguage, voice), hfToken, `preset voice "${voice}"`);
};

const handleInit = async (
	message: Extract<WasmWorkerRequest, { kind: "init" }>,
) => {
	stopRequested = true;
	activeStreamToken += 1;
	model = null;
	const selected = selectConfig(configs, message.language ?? "english");
	currentLanguage = selected.name;
	const manual = message.manualAssets;
	stockVoicesCompatible = !manual?.configBytes && !manual?.weightsBytes;
	const configBytes = manual?.configBytes ?? encoder.encode(selected.yaml);
	// Fully supplied manual bundles need not declare remote asset URLs.
	const assets = () => configAssets(new TextDecoder().decode(configBytes));

	postStatus({
		phase: "initializing-runtime",
		progress: 10,
		message: "Loading WASM runtime...",
		source: null,
		ready: false,
		error: null,
	});

	if (!bindings) {
		const modulePath = `${message.wasmBase.replace(/\/+$/, "")}/pocket_tts.js`;
		bindings = (await import(/* @vite-ignore */ modulePath)) as WasmBindings;
	}
	await bindings.default();

	let source: "hf" | "manual" = "manual";
	let weightsBytes = manual?.weightsBytes;

	if (!weightsBytes) {
		postStatus({
			phase: "loading-assets",
			progress: 42,
			message: `Fetching ${currentLanguage} model weights...`,
			source: null,
			ready: false,
			error: null,
		});
		weightsBytes = await fetchHF(assets().weightsUrl, message.hfToken, "model weights");
		source = "hf";
	}

	let tokenizerBytes = manual?.tokenizerBytes;
	if (!tokenizerBytes || tokenizerBytes.byteLength === 0) {
		postStatus({
			phase: "loading-assets",
			progress: 60,
			message: `Fetching ${currentLanguage} tokenizer...`,
			source: null,
			ready: false,
			error: null,
		});
		tokenizerBytes = await fetchHF(assets().tokenizerUrl, message.hfToken, "tokenizer");
	}

	postStatus({
		phase: "compiling-model",
		progress: 78,
		message: "Compiling model in WASM...",
		source,
		ready: false,
		error: null,
	});

	console.log(
		`[wasm-worker] Loading model: language=${currentLanguage}, config=${configBytes.byteLength}b, weights=${weightsBytes.byteLength}b, tokenizer=${tokenizerBytes.byteLength}b`,
	);
	model = new bindings.WasmTTSModel();
	model.load_from_buffer(configBytes, weightsBytes, tokenizerBytes, !!manual?.weightsBytes);
	sampleRate = model.sample_rate;
	console.log(`[wasm-worker] Model ready: sampleRate=${sampleRate}`);

	postStatus({
		phase: "ready",
		progress: 100,
		message: "WASM model is ready.",
		source,
		ready: true,
		error: null,
	});

	postOk(message.requestId, { sampleRate });
};

const handlePrepareVoice = async (
	message: Extract<WasmWorkerRequest, { kind: "prepare_voice" }>,
) => {
	const readyModel = ensureReadyModel();
	const input: WasmWorkerVoiceInput = message.voice;

	if (input.kind === "wav") {
		readyModel.load_voice_from_buffer(input.wavBytes);
		postOk(message.requestId);
		return;
	}

	if (input.kind === "embedding") {
		readyModel.load_voice_from_safetensors(input.embeddingBytes);
		postOk(message.requestId);
		return;
	}

	if (!isPresetVoice(input.voice)) {
		throw new Error(`Unknown preset voice: ${input.voice}`);
	}
	if (!stockVoicesCompatible) {
		throw new Error("Preset voices require a released model bundle. Supply a matching embedding for manual weights or configs.");
	}

	const bytes = await fetchEmbedding(input.voice, input.hfToken);
	readyModel.load_voice_from_safetensors(bytes);
	postOk(message.requestId);
};

const handleStartStream = async (
	message: Extract<WasmWorkerRequest, { kind: "start_stream" }>,
) => {
	const readyModel = ensureReadyModel();

	stopRequested = false;
	const streamToken = ++activeStreamToken;

	const stream = readyModel.start_stream(message.text);
	let firstChunkSent = false;
	let chunkCount = 0;

	while (!stopRequested && streamToken === activeStreamToken) {
		const startChunkSamples = Math.max(320, Math.floor(sampleRate * 0.032));
		const steadyChunkSamples = Math.max(1024, Math.floor(sampleRate * 0.11));
		const targetSamples =
			chunkCount < 3 ? startChunkSamples : steadyChunkSamples;

		const chunk = stream.next_chunk_min_samples(targetSamples);
		if (chunk == null) {
			break;
		}

		if (!firstChunkSent) {
			firstChunkSent = true;
			postEvent({ kind: "stream_first_chunk" });
		}

		const stats = stream.last_chunk_stats();
		const computeMs =
			typeof stats.compute_ms === "number" ? stats.compute_ms : null;
		const mergedChunks =
			typeof stats.chunks_merged === "number" ? stats.chunks_merged : null;

		postEvent(
			{
				kind: "stream_chunk",
				chunk,
				computeMs,
				mergedChunks,
			},
			[chunk.buffer],
		);

		chunkCount += 1;
		if (chunkCount % 6 === 0) {
			await sleep(0);
		}
	}

	if (stopRequested || streamToken !== activeStreamToken) {
		throw new Error("abort");
	}

	postEvent({ kind: "stream_done" });
	postOk(message.requestId);
};

self.onmessage = (event: MessageEvent<WasmWorkerRequest>) => {
	const message = event.data;

	if (!message || typeof message !== "object") {
		return;
	}

	if (message.kind === "stop") {
		stopRequested = true;
		activeStreamToken += 1;
		return;
	}

	void (async () => {
		try {
			if (message.kind === "init") {
				await handleInit(message);
				return;
			}

			if (message.kind === "prepare_voice") {
				await handlePrepareVoice(message);
				return;
			}

			if (message.kind === "start_stream") {
				await handleStartStream(message);
				return;
			}
		} catch (err) {
			if (message.kind === "init") {
				const error = err instanceof Error ? err.message : String(err);
				postStatus({ phase: "error", progress: 0, message: error, source: null, ready: false, error });
			}
			if (message.kind === "start_stream") {
				const text = err instanceof Error ? err.message : String(err);
				postEvent({ kind: "stream_error", error: text });
			}
			postErr(message.requestId, err);
		}
	})();
};

export {};
