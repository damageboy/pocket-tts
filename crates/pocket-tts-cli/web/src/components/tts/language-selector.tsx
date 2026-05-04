import { Label } from "@/components/ui/label";
import { useEffect, useState } from "react";

interface LanguageEntry {
	name: string;
	description: string;
	default_voice: string;
	layers: number;
	status: string;
}

const FALLBACK_LANGUAGES: LanguageEntry[] = [
	{
		name: "english",
		description: "English (latest)",
		default_voice: "alba",
		layers: 6,
		status: "production",
	},
	{
		name: "german",
		description: "German",
		default_voice: "juergen",
		layers: 6,
		status: "production",
	},
	{
		name: "italian",
		description: "Italian",
		default_voice: "giovanni",
		layers: 6,
		status: "production",
	},
	{
		name: "portuguese",
		description: "Portuguese",
		default_voice: "rafael",
		layers: 6,
		status: "production",
	},
	{
		name: "spanish",
		description: "Spanish",
		default_voice: "lola",
		layers: 6,
		status: "production",
	},
	{
		name: "french_24l",
		description: "French (24-layer)",
		default_voice: "estelle",
		layers: 24,
		status: "preview",
	},
	{
		name: "german_24l",
		description: "German (24-layer)",
		default_voice: "juergen",
		layers: 24,
		status: "preview",
	},
	{
		name: "italian_24l",
		description: "Italian (24-layer)",
		default_voice: "giovanni",
		layers: 24,
		status: "preview",
	},
	{
		name: "portuguese_24l",
		description: "Portuguese (24-layer)",
		default_voice: "rafael",
		layers: 24,
		status: "preview",
	},
	{
		name: "spanish_24l",
		description: "Spanish (24-layer)",
		default_voice: "lola",
		layers: 24,
		status: "preview",
	},
];

const DEFAULT_TEXT: Record<string, string> = {
	english: "It's small enough to fit in your pocket.",
	french: "C'est assez petit pour tenir dans votre poche.",
	german: "Es ist klein genug, um in Ihre Tasche zu passen.",
	spanish: "Es lo suficientemente pequeño como para caber en tu bolsillo.",
	portuguese: "É pequeno o suficiente para caber no seu bolso.",
	italian: "È abbastanza piccolo da stare in tasca.",
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
					// Filter out legacy entries for the UI
					setLanguages(
						data.filter((l: LanguageEntry) => l.status !== "legacy"),
					);
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
