/* tslint:disable */
/* eslint-disable */

/**
 * WASM-compatible TTS model wrapper
 */
export class WasmTTSModel {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Generate audio from text
     *
     * # Arguments
     * * `text` - Text to synthesize
     *
     * # Returns
     * Float32Array containing audio samples at 24kHz mono
     */
    generate(text: string): Float32Array;
    /**
     * Generate audio and return as base64-encoded WAV
     */
    generate_wav_base64(text: string): string;
    /**
     * Check if model is ready for generation
     */
    is_ready(): boolean;
    /**
     * Load model from ArrayBuffers
     *
     * # Arguments
     * * `config_yaml` - ArrayBuffer containing config.yaml
     * * `weights_data` - ArrayBuffer containing safetensors model weights
     * * `tokenizer_bytes` - ArrayBuffer containing the sentencepiece tokenizer.model
     */
    load_from_buffer(config_yaml: Uint8Array, weights_data: Uint8Array, tokenizer_bytes: Uint8Array): void;
    /**
     * Load voice from WAV audio buffer for voice cloning
     */
    load_voice_from_buffer(wav_bytes: Uint8Array): void;
    /**
     * Load voice from safetensors buffer (pre-calculated embedding)
     */
    load_voice_from_safetensors(bytes: Uint8Array): void;
    /**
     * Create a new WASM TTS model
     */
    constructor();
    /**
     * Start streaming audio generation from text
     *
     * Returns a stream object that yields Float32Array chunks.
     */
    start_stream(text: string): WasmTTSStream;
    /**
     * Get the sample rate of generated audio
     */
    readonly sample_rate: number;
}

/**
 * WASM-compatible streaming audio iterator
 */
export class WasmTTSStream {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get stats for the most recently produced chunk.
     *
     * Returns a JS object with keys:
     * - samples: number
     * - compute_ms: number
     * - chunks_merged: number
     */
    last_chunk_stats(): any;
    /**
     * Get the next chunk of audio samples.
     *
     * Returns None when the stream is complete.
     */
    next_chunk(): Float32Array | undefined;
    /**
     * Get a chunk with at least `min_samples` samples when available.
     *
     * This amortizes JS/WASM boundary overhead by combining multiple internal
     * stream frames into a larger output chunk.
     */
    next_chunk_min_samples(min_samples: number): Float32Array | undefined;
}

/**
 * Initialize console_error_panic_hook for better error messages in browser
 */
export function init(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmttsmodel_free: (a: number, b: number) => void;
    readonly __wbg_wasmttsstream_free: (a: number, b: number) => void;
    readonly wasmttsmodel_generate: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmttsmodel_generate_wav_base64: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmttsmodel_is_ready: (a: number) => number;
    readonly wasmttsmodel_load_from_buffer: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly wasmttsmodel_load_voice_from_buffer: (a: number, b: number, c: number) => [number, number];
    readonly wasmttsmodel_load_voice_from_safetensors: (a: number, b: number, c: number) => [number, number];
    readonly wasmttsmodel_new: () => number;
    readonly wasmttsmodel_sample_rate: (a: number) => number;
    readonly wasmttsmodel_start_stream: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmttsstream_last_chunk_stats: (a: number) => any;
    readonly wasmttsstream_next_chunk: (a: number) => [number, number, number];
    readonly wasmttsstream_next_chunk_min_samples: (a: number, b: number) => [number, number, number];
    readonly init: () => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
