export type WasmAssetSource = "local" | "hf" | "manual" | null;

export type WordTimestampEvent =
	| { kind: "word_start"; word: string; word_index: number; start_time: number }
	| { kind: "word_end"; word: string; word_index: number; start_time: number; end_time: number };

export interface TimestampBatch {
	audio: Float32Array;
	events: WordTimestampEvent[];
	start_time: number | null;
	end_time: number | null;
	chunks_merged: number;
	compute_ms: number;
}

export type WasmLoadPhase =
	| "idle"
	| "initializing-runtime"
	| "loading-assets"
	| "compiling-model"
	| "ready"
	| "error";

export interface WasmWorkerStatus {
	phase: WasmLoadPhase;
	progress: number;
	message: string;
	source: WasmAssetSource;
	ready: boolean;
	error: string | null;
}

export interface WasmWorkerManualAssets {
	configBytes?: Uint8Array;
	weightsBytes?: Uint8Array;
	tokenizerBytes?: Uint8Array;
}

export type WasmWorkerVoiceInput =
	| {
			kind: "preset";
			voice: string;
			hfRepo: string;
			hfToken: string;
	  }
	| {
			kind: "wav";
			wavBytes: Uint8Array;
	  }
	| {
			kind: "embedding";
			embeddingBytes: Uint8Array;
	  };

export type WasmWorkerRequest =
	| {
			kind: "init";
			requestId: number;
			wasmBase: string;
			hfRepo: string;
			hfToken: string;
			language?: string;
			manualAssets?: WasmWorkerManualAssets;
	  }
	| {
			kind: "prepare_voice";
			requestId: number;
			voice: WasmWorkerVoiceInput;
	  }
	| {
			kind: "start_stream";
			requestId: number;
			text: string;
			timestamps?: boolean;
	  }
	| {
			kind: "stop";
	  };

export type WasmWorkerEvent =
	| {
			kind: "status";
			status: WasmWorkerStatus;
	  }
	| {
			kind: "rpc_ok";
			requestId: number;
			payload?: {
				sampleRate?: number;
			};
	  }
	| {
			kind: "rpc_err";
			requestId: number;
			error: string;
	  }
	| {
			kind: "stream_first_chunk";
			requestId: number;
	  }
	| {
			kind: "stream_chunk";
			requestId: number;
			chunk: Float32Array;
			computeMs: number | null;
			mergedChunks: number | null;
	  }
	| {
			kind: "stream_words";
			requestId: number;
			events: WordTimestampEvent[];
	  }
	| {
			kind: "stream_done";
			requestId: number;
	  }
	| {
			kind: "stream_error";
			requestId: number;
			error: string;
	  };
