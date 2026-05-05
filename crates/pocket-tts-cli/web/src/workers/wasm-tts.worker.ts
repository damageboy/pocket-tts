/// <reference lib="webworker" />

import type {
	WasmWorkerEvent,
	WasmWorkerRequest,
	WasmWorkerStatus,
	WasmWorkerVoiceInput,
} from "./wasm-tts-protocol";

declare const self: DedicatedWorkerGlobalScope;

// v2 model URLs
const V2_REPO = "kyutai/pocket-tts-without-voice-cloning";
const V2_MODEL_REVISION = "d29db7978e464fb90cb3359ee0c69a273b9142cc";
const V2_VOICE_REVISION = "e041936c75475d350b405bc870bcf7c22da4e9e6";
let currentLanguage = "english";

interface LangMeta {
	layers: number;
	removeSemicolons: boolean;
	framesAfterEos: number | null;
}

const LANG_META: Record<string, LangMeta> = {
	english: { layers: 6, removeSemicolons: false, framesAfterEos: null },
	english_2026_04: { layers: 6, removeSemicolons: false, framesAfterEos: null },
	english_2026_01: { layers: 6, removeSemicolons: false, framesAfterEos: null },
	german: { layers: 6, removeSemicolons: true, framesAfterEos: null },
	italian: { layers: 6, removeSemicolons: false, framesAfterEos: null },
	spanish: { layers: 6, removeSemicolons: false, framesAfterEos: null },
	portuguese: { layers: 6, removeSemicolons: false, framesAfterEos: null },
	french_24l: { layers: 24, removeSemicolons: true, framesAfterEos: 8 },
	german_24l: { layers: 24, removeSemicolons: true, framesAfterEos: null },
	italian_24l: { layers: 24, removeSemicolons: false, framesAfterEos: null },
	portuguese_24l: { layers: 24, removeSemicolons: false, framesAfterEos: null },
	spanish_24l: { layers: 24, removeSemicolons: false, framesAfterEos: null },
};

const buildConfigYaml = (lang: string): string => {
	const meta = LANG_META[lang] ?? LANG_META.english;
	let yaml = `
flow_lm:
  insert_bos_before_voice: true
  dtype: float32
  flow:
    depth: 6
    dim: 512
  transformer:
    d_model: 1024
    hidden_scale: 4
    max_period: 10000
    num_heads: 16
    num_layers: ${meta.layers}
  lookup_table:
    dim: 1024
    n_bins: 4000
    tokenizer: sentencepiece
    tokenizer_path: dummy

mimi:
  dtype: float32
  sample_rate: 24000
  inner_dim: 32
  outer_dim: 512
  channels: 1
  frame_rate: 12.5
  seanet:
    dimension: 512
    channels: 1
    n_filters: 64
    n_residual_layers: 1
    ratios: [6, 5, 4]
    kernel_size: 7
    residual_kernel_size: 3
    last_kernel_size: 3
    dilation_base: 2
    pad_mode: constant
    compress: 2
  transformer:
    d_model: 512
    num_heads: 8
    num_layers: 2
    layer_scale: 0.01
    context: 250
    dim_feedforward: 2048
    input_dimension: 512
    output_dimensions: [512]
  quantizer:
    dimension: 32
    output_dimension: 512
`;
	if (meta.removeSemicolons) {
		yaml = `remove_semicolons: true\n` + yaml;
	}
	if (meta.framesAfterEos !== null) {
		yaml =
			`model_recommended_frames_after_eos: ${meta.framesAfterEos}\n` + yaml;
	}
	return yaml;
};

const PRESET_VOICES = [
	"alba",
	"anna",
	"azelma",
	"bill_boerst",
	"caro_davy",
	"charles",
	"cosette",
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
	path: string,
	revision: string,
	hfToken: string,
	label: string,
): Promise<Uint8Array> => {
	const url = `https://huggingface.co/${V2_REPO}/resolve/${revision}/${path}`;

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

const fetchWeights = async (
	_hfRepo: string,
	hfToken: string,
): Promise<{ bytes: Uint8Array; source: "local" | "hf" }> => {
	// Try local path first
	const localPath = "/model.safetensors";
	try {
		const localRes = await fetch(localPath);
		if (localRes.ok) {
			return {
				bytes: new Uint8Array(await localRes.arrayBuffer()),
				source: "local",
			};
		}
	} catch {
		// Ignore local fallback failures.
	}

	const bytes = await fetchHF(
		`languages/${currentLanguage}/model.safetensors`,
		V2_MODEL_REVISION,
		hfToken,
		"model weights",
	);
	return { bytes, source: "hf" };
};

const fetchTokenizer = async (hfToken: string): Promise<Uint8Array> => {
	// Try local path first
	try {
		const localRes = await fetch("/tokenizer.model");
		if (localRes.ok) {
			return new Uint8Array(await localRes.arrayBuffer());
		}
	} catch {
		// Ignore.
	}

	return fetchHF(
		`languages/${currentLanguage}/tokenizer.model`,
		V2_MODEL_REVISION,
		hfToken,
		"tokenizer",
	);
};

const fetchEmbedding = async (
	voice: string,
	_hfRepo: string,
	hfToken: string,
): Promise<Uint8Array> => {
	// Try local path first
	const localUrl = `/embeddings/${voice}.safetensors`;
	try {
		const localRes = await fetch(localUrl);
		if (localRes.ok) {
			return new Uint8Array(await localRes.arrayBuffer());
		}
	} catch {
		// Ignore local fallback failures.
	}

	const headers: Record<string, string> = {};
	if (hfToken.trim()) {
		headers.Authorization = `Bearer ${hfToken.trim()}`;
	}

	// v2 per-language voice embeddings
	const url = `https://huggingface.co/${V2_REPO}/resolve/${V2_VOICE_REVISION}/languages/${currentLanguage}/embeddings/${voice}.safetensors`;
	const res = await fetch(url, { headers });
	if (!res.ok) {
		if (res.status === 401) {
			throw new Error("HF auth required for preset voice fetch.");
		}
		throw new Error(
			`Failed to fetch voice "${voice}" (${res.status}). ` +
				`Check voice name with: pocket-tts voice-list`,
		);
	}

	return new Uint8Array(await res.arrayBuffer());
};

const handleInit = async (
	message: Extract<WasmWorkerRequest, { kind: "init" }>,
) => {
	stopRequested = true;
	activeStreamToken += 1;

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

	// Set language BEFORE fetching — weights/tokenizer/voice URLs depend on it
	currentLanguage = message.language ?? "english";

	const manual = message.manualAssets;

	let source: "local" | "hf" | "manual" = "manual";
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
		const fetched = await fetchWeights(message.hfRepo, message.hfToken);
		weightsBytes = fetched.bytes;
		source = fetched.source;
	}

	const configBytes =
		manual?.configBytes ?? encoder.encode(buildConfigYaml(currentLanguage));

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
		tokenizerBytes = await fetchTokenizer(message.hfToken);
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
	model.load_from_buffer(configBytes, weightsBytes, tokenizerBytes);
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

	const bytes = await fetchEmbedding(input.voice, input.hfRepo, input.hfToken);
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
			if (message.kind === "start_stream") {
				const text = err instanceof Error ? err.message : String(err);
				postEvent({ kind: "stream_error", error: text });
			}
			postErr(message.requestId, err);
		}
	})();
};

export {};
