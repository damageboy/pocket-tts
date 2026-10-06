/// <reference types="node" />
import { it } from "node:test";
import assert from "node:assert/strict";
import { renderToStaticMarkup } from "react-dom/server";
import { WordReadAlong } from "./word-read-along";
import type { WordTimestampEvent } from "../../workers/wasm-tts-protocol";

const events: WordTimestampEvent[] = [
	{ kind: "word_start", word: "go", word_index: 0, start_time: 0.16 },
	{ kind: "word_end", word: "go", word_index: 0, start_time: 0.16, end_time: 0.40 },
	// Index 1 was not aligned. Do not shift the last repeated word onto it.
	{ kind: "word_start", word: "go", word_index: 2, start_time: 0.72 },
	{ kind: "word_end", word: "go", word_index: 2, start_time: 0.72, end_time: 1.04 },
];
const render = (playbackTime: number, playing = true) => renderToStaticMarkup(
	<WordReadAlong text="go, go… go!" events={events} playbackTime={playbackTime} playing={playing} />,
);

it("highlights by playback time, not timestamp arrival, with exclusive end boundaries", () => {
	for (const time of [0, 0.40, 0.65, 1.04]) assert(!render(time).includes('aria-current="true"'));
	assert.match(render(0.16), /data-word-index="0"[^>]*aria-current="true"/);
	assert.match(render(0.90), /data-word-index="2"[^>]*aria-current="true"/);
	assert(!render(0.90, false).includes('aria-current="true"'));
});

it("preserves punctuation and Unicode words without treating markup as HTML", () => {
	const html = renderToStaticMarkup(<WordReadAlong text={"<It's> café\u0301—l’été."}
		events={[{ kind: "word_start", word: "l’été", word_index: 2, start_time: 0.8 }]}
		playbackTime={0.9} playing />);
	assert(html.includes("&lt;"));
	assert(html.includes("café\u0301"));
	assert.match(html, /data-word-index="2"[^>]*aria-current="true"[^>]*>l’été<\/span>/);
});
