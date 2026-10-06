import { Label } from "@/components/ui/label";
import { useEffect, useState } from "react";
import { FALLBACK_LANGUAGES, type LanguageEntry } from "@/lib/model-catalog";

const DEFAULT_TEXT: Record<string, string> = {
	english: "It's small enough to fit in your pocket.",
	french: "C'est assez petit pour tenir dans votre poche.",
	german: "Es ist klein genug, um in Ihre Tasche zu passen.",
	spanish: "Es lo suficientemente pequeño como para caber en tu bolsillo.",
	portuguese: "É pequeno o suficiente para caber no seu bolso.",
	italian: "È abbastanza piccolo da stare in tasca.",
	dutch: "Het is klein genoeg om in je zak te passen.",
};

export function defaultTextForLanguage(language: string): string {
	for (const [key, text] of Object.entries(DEFAULT_TEXT)) {
		if (language.includes(key)) return text;
	}
	return DEFAULT_TEXT.english;
}

interface LanguageSelectorProps {
	selectedLanguage: string;
	onLanguageChange: (
		language: string,
		defaultVoice: string,
		defaultText: string,
	) => void;
	disabled?: boolean;
}

export function LanguageSelector({
	selectedLanguage,
	onLanguageChange,
	disabled = false,
}: LanguageSelectorProps) {
	const [languages, setLanguages] =
		useState<LanguageEntry[]>(FALLBACK_LANGUAGES);

	useEffect(() => {
		fetch("/api/languages")
			.then((r) => r.json())
			.then((data) => {
				if (Array.isArray(data) && data.length > 0) {
					setLanguages(data);
				}
			})
			.catch(() => {
				/* use fallback */
			});
	}, []);

	const langFlag = (name: string) => {
		if (name.startsWith("english")) return "🇬🇧";
		if (name.startsWith("french")) return "🇫🇷";
		if (name.startsWith("german")) return "🇩🇪";
		if (name.startsWith("italian")) return "🇮🇹";
		if (name.startsWith("spanish")) return "🇪🇸";
		if (name.startsWith("portuguese")) return "🇧🇷";
		if (name.startsWith("dutch")) return "🇳🇱";
		return "🌍";
	};

	return (
		<div className="space-y-2">
			<Label className="text-muted-foreground text-xs uppercase tracking-wider font-semibold">
				Language
			</Label>
			<select
				value={selectedLanguage}
				onChange={(e) => {
					const lang = languages.find((l) => l.name === e.target.value);
					onLanguageChange(
						e.target.value,
						lang?.default_voice ?? "alba",
						defaultTextForLanguage(e.target.value),
					);
				}}
				disabled={disabled}
				className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
			>
				{languages.map((lang) => (
					<option key={lang.name} value={lang.name}>
						{langFlag(lang.name)} {lang.description}
						{lang.status === "preview" ? " (β preview)" : ""}
					</option>
				))}
			</select>
		</div>
	);
}
