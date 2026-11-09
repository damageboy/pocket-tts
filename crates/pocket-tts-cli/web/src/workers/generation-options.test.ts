/// <reference types="node" />
import { test } from "node:test";
import assert from "node:assert/strict";
import { validateGenerationOptions, type GenerationOptions } from "./wasm-tts-protocol";

test("generation options preserve omitted defaults and explicit zero", () => {
    assert.deepEqual(validateGenerationOptions({}), { max_tokens: undefined, teacher_force_previous_chunk: undefined });
    assert.deepEqual(validateGenerationOptions({ max_tokens: 0, teacher_force_previous_chunk: true }), { max_tokens: 0, teacher_force_previous_chunk: true });
    assert.equal(validateGenerationOptions({ max_tokens: 25, teacher_force_previous_chunk: false }).max_tokens, 25);
});

test("worker rejects malformed generation options before starting inference", () => {
    for (const max_tokens of [-1, 1.5, NaN, Infinity, 0x100000000, "25", null]) {
        assert.throws(() => validateGenerationOptions({ max_tokens } as unknown as GenerationOptions), /max_tokens/);
    }
    for (const teacher_force_previous_chunk of [1, "false", null]) {
        assert.throws(() => validateGenerationOptions({ teacher_force_previous_chunk } as unknown as GenerationOptions), /teacher_force_previous_chunk/);
    }
});
