/// <reference types="node" />

import { it } from "node:test";
import assert from "node:assert/strict";
import { renderToStaticMarkup } from "react-dom/server";
import { LanguageSelector, defaultTextForLanguage } from "./language-selector";
import { VoiceSelector } from "./voice-selector";

it("offers Dutch, French 6L and dated English releases in the offline catalog", () => {
  const html = renderToStaticMarkup(<LanguageSelector selectedLanguage="dutch" onLanguageChange={() => {}} />);
  for (const name of ["dutch", "dutch_24l", "french", "english_2026-09", "english_2026-09_24l", "english_2026-04_24l", "english_2026-01"]) {
    assert.ok(html.includes(`value="${name}"`), name);
  }
  assert.equal(defaultTextForLanguage("dutch_24l"), "Het is klein genoeg om in je zak te passen.");
  assert.equal(defaultTextForLanguage("english_2026-01"), defaultTextForLanguage("english"));
});

it("offers daan with Dutch voice metadata without an API", () => {
  const html = renderToStaticMarkup(<VoiceSelector selectedVoice="daan" customVoice="" onVoiceSelect={() => {}} onCustomVoiceChange={() => {}} />);
  assert.ok(html.includes("daan (male, reading, dutch)"));
  assert.ok(html.includes("🇳🇱"));
});
