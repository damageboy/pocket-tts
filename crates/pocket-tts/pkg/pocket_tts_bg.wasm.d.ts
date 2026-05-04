/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const __wbg_wasmttsmodel_free: (a: number, b: number) => void;
export const __wbg_wasmttsstream_free: (a: number, b: number) => void;
export const wasmttsmodel_generate: (a: number, b: number, c: number) => [number, number, number];
export const wasmttsmodel_generate_wav_base64: (a: number, b: number, c: number) => [number, number, number, number];
export const wasmttsmodel_is_ready: (a: number) => number;
export const wasmttsmodel_load_from_buffer: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
export const wasmttsmodel_load_voice_from_buffer: (a: number, b: number, c: number) => [number, number];
export const wasmttsmodel_load_voice_from_safetensors: (a: number, b: number, c: number) => [number, number];
export const wasmttsmodel_new: () => number;
export const wasmttsmodel_sample_rate: (a: number) => number;
export const wasmttsmodel_start_stream: (a: number, b: number, c: number) => [number, number, number];
export const wasmttsstream_last_chunk_stats: (a: number) => any;
export const wasmttsstream_next_chunk: (a: number) => [number, number, number];
export const wasmttsstream_next_chunk_min_samples: (a: number, b: number) => [number, number, number];
export const init: () => void;
export const __wbindgen_exn_store: (a: number) => void;
export const __externref_table_alloc: () => number;
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_free: (a: number, b: number, c: number) => void;
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
export const __externref_table_dealloc: (a: number) => void;
export const __wbindgen_start: () => void;
