import { Fragment, useMemo } from "react";
import type { WordTimestampEvent } from "@/workers/wasm-tts-protocol";

interface WordReadAlongProps {
	text: string;
	events: WordTimestampEvent[];
	playbackTime: number;
	playing: boolean;
}

export function WordReadAlong({ text, events, playbackTime, playing }: WordReadAlongProps) {
	const words = useMemo(() => {
		// Match the Rust timestamp mapper's lexical words, not whitespace tokens.
		// Explicit pause markers are stripped by both streaming APIs.
		const source = text.replace(/\[pause:\d+(?:\.\d+)?(?:ms|s)\]/g, " ");
		const pattern = /[\p{L}\p{N}][\p{L}\p{N}\p{M}]*(?:[-‐‑'’][\p{L}\p{N}][\p{L}\p{N}\p{M}]*)*/gu;
		let end = 0;
		const parts: { before: string; word: string }[] = [];
		for (const match of source.matchAll(pattern)) {
			const before = source.slice(end, match.index);
			end = match.index + match[0].length;
			parts.push({ before, word: match[0] });
		}
		return { parts, after: source.slice(end) };
	}, [text]);
	const timings = useMemo(() => {
		const result = new Map<number, WordTimestampEvent>();
		for (const event of events) result.set(event.word_index, event);
		return result;
	}, [events]);

	return (
		<section aria-label="Timestamp read-along" className="space-y-3 rounded-lg border border-primary/20 bg-primary/5 p-4">
			<div className="flex flex-wrap items-center justify-between gap-2">
				<h3 className="text-sm font-semibold">Word timing · read along</h3>
				<span className="text-xs font-mono text-muted-foreground" data-playback-time={playbackTime}>
					{playbackTime.toFixed(2)}s · {timings.size} timed words
				</span>
			</div>
			<p className="whitespace-pre-wrap break-words text-lg leading-loose">
				{words.parts.map(({ before, word }, index) => {
					const event = timings.get(index);
					// Never highlight a different source word when normalization is ambiguous.
					const timing = event?.word === word ? event : undefined;
					const active = playing && timing && playbackTime >= timing.start_time &&
						(timing.kind === "word_start" || playbackTime < timing.end_time);
					return <Fragment key={index}>
						{before}<span data-word-index={index} aria-current={active ? "true" : undefined}
							title={timing ? `${timing.start_time.toFixed(2)}s – ${timing.kind === "word_end" ? `${timing.end_time.toFixed(2)}s` : "pending"}` : "No timestamp yet"}
							className={active ? "rounded bg-primary text-primary-foreground ring-2 ring-primary font-semibold" : undefined}
						>{word}</span>
					</Fragment>;
				})}{words.after}
			</p>
			<p className="text-xs text-muted-foreground">
				Follows audio playback, not generation. Approximate 80 ms boundaries;
				uncertain words may stay unhighlighted. Hover a word to inspect its timing.
			</p>
		</section>
	);
}
