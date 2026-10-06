import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { FALLBACK_VOICES, type VoiceEntry } from "@/lib/model-catalog";

interface VoiceSelectorProps {
	selectedVoice: string | null;
	customVoice: string;
	onVoiceSelect: (voice: string | null) => void;
	onCustomVoiceChange: (url: string) => void;
	customEnabled?: boolean;
	customLabel?: string;
	customPlaceholder?: string;
}

export function VoiceSelector({
	selectedVoice,
	customVoice,
	onVoiceSelect,
	onCustomVoiceChange,
	customEnabled = true,
	customLabel = "Or use a custom URL / Path",
	customPlaceholder = "hf://kyutai/tts-voices/voice.wav",
}: VoiceSelectorProps) {
	const [voices, setVoices] = useState<VoiceEntry[]>(FALLBACK_VOICES);

	useEffect(() => {
		fetch("/api/voices")
			.then((r) => r.json())
			.then((data) => {
				if (Array.isArray(data) && data.length > 0) setVoices(data);
			})
			.catch(() => {
				/* use fallback */
			});
	}, []);

	const genderIcon = (g: string) => (g === "f" ? "♀" : "♂");
	const langFlag = (l: string) => {
		const flags: Record<string, string> = {
			english: "🇬🇧",
			french: "🇫🇷",
			german: "🇩🇪",
			italian: "🇮🇹",
			spanish: "🇪🇸",
			portuguese: "🇧🇷",
			dutch: "🇳🇱",
		};
		return flags[l] || "🌍";
	};

	return (
		<div className="space-y-4">
			<Label className="text-muted-foreground text-xs uppercase tracking-wider font-semibold">
				Voice Selection
			</Label>
			<div className="grid grid-cols-4 gap-2">
				{voices.map((voice) => (
					<Button
						key={voice.name}
						variant={selectedVoice === voice.name ? "default" : "outline"}
						size="sm"
						className={cn(
							"capitalize transition-all duration-200 text-xs",
							selectedVoice === voice.name &&
								"shadow-md shadow-primary/20 scale-[1.02]",
						)}
						onClick={() => onVoiceSelect(voice.name)}
						title={`${voice.name} (${voice.gender === "f" ? "female" : "male"}, ${voice.style}, ${voice.language})`}
					>
						<span className="mr-1 opacity-60">{genderIcon(voice.gender)}</span>
						{voice.name.replace(/_/g, " ")}
						{voice.language !== "english" && (
							<span className="ml-1 opacity-60">
								{langFlag(voice.language)}
							</span>
						)}
					</Button>
				))}
			</div>
			{customEnabled && (
				<div className="space-y-2">
					<Label
						htmlFor="custom-voice"
						className="text-xs text-muted-foreground"
					>
						{customLabel}
					</Label>
					<Input
						id="custom-voice"
						placeholder={customPlaceholder}
						value={customVoice}
						onChange={(e) => onCustomVoiceChange(e.target.value)}
						className="bg-muted/30 border-muted-foreground/20 focus:border-primary/50"
					/>
				</div>
			)}
		</div>
	);
}
