#!/usr/bin/env python3
"""Offline-only timestamp calibration. See assets/timestamp-calibration-2026-11-09.

Separate commands make the heldout gate explicit: prepare, capture --split calibration,
transcribe --split calibration, rank, capture --split evaluation, transcribe --split
evaluation, evaluate. Use the pinned ASR Python only for transcribe. Native export
schema1 stores each segment's source units, all-head unit probabilities per emitted
frame, sample clock, voiced mask, and native events for replay verification.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "assets/timestamp-calibration-2026-11-09/protocol.json"
MODELS = ["english_2m_2026-10", "english_7m_2026-10", "english_18m_2026-10", "english"]
NUMBERS = {
    "12": ["twelve"],
    "5": ["five"],
    "20": ["twenty"],
    "2024": ["twentytwentyfour", "twothousandtwentyfour", "twothousandandtwentyfour"],
    "2025": ["twentytwentyfive", "twothousandtwentyfive", "twothousandandtwentyfive"],
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_protocol(path=PROTOCOL):
    data = json.loads(path.read_text())
    a, b = data["calibration"], data["evaluation"]
    assert not ({x["text"] for x in a} & {x["text"] for x in b}), "overlapping texts"
    assert len({x["id"] for x in a + b}) == len(a + b), "overlapping IDs"
    return data


def calibration_matches(record, identity):
    return record.get("promoted") is True and record.get("identity") == identity


def promotion_gate(old, new):
    failures = [
        key
        for key in (
            "start_mean",
            "start_p95",
            "end_mean",
            "end_p95",
            "pooled_mean",
            "pooled_p95",
        )
        if new[key] > old[key]
    ]
    if new["coverage"] < old["coverage"]:
        failures.append("coverage")
    return {"passed": not failures, "failures": failures}


def key(word):
    normalized = re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))
    return {"neighbours": "neighbors", "harbour": "harbor"}.get(normalized, normalized)


def match_reference(source, words):
    """Minimum-edit monotone mapping with explicit numeric equivalences.

    Every source word maps to at most one contiguous ASR span. No timing guessed
    for missing words. Insertions and deletions remain visible for adjudication.
    """
    n, m = len(source), len(words)
    dp = {(0, 0): (0, None)}

    def put(i, j, c, back):
        if (i, j) not in dp or c < dp[i, j][0]:
            dp[i, j] = (c, back)

    for i in range(n + 1):
        for j in range(m + 1):
            if (i, j) not in dp:
                continue
            cost = dp[i, j][0]
            if i < n and j < m:
                variants = {key(source[i]), *NUMBERS.get(key(source[i]), [])}
                for length in range(1, min(5, m - j) + 1):
                    if (
                        "".join(key(w["word"]) for w in words[j : j + length])
                        in variants
                    ):
                        put(i + 1, j + length, cost, (i, j, "match"))
            if i < n:
                put(i + 1, j, cost + 1, (i, j, "missing"))
            if j < m:
                put(i, j + 1, cost + 1, (i, j, "extra"))
    matched = {}
    missing = []
    extra = []
    i, j = n, m
    while i or j:
        pi, pj, kind = dp[i, j][1]
        if kind == "match":
            matched[pi] = (words[pj]["start"], words[j - 1]["end"])
        elif kind == "missing":
            missing.append(pi)
        else:
            extra.append(pj)
        i, j = pi, pj
    return matched, sorted(missing), sorted(extra)


def replay(data, heads):
    """Exact alignment branch semantics; exporter supplies probabilities and gate."""
    import numpy as np

    indices = [data["heads"].index(list(h)) for h in heads]
    events = []
    for segment in data["segments"]:
        units = segment["units"]
        word_units = [i for i, u in enumerate(units) if u["word"] is not None]
        nxt = 0
        opened = None

        def close(end):
            nonlocal opened
            pos, start = opened
            w = units[word_units[pos]]["word"]
            events.append(
                {"word": w["text"], "index": w["index"], "start": start, "end": end}
            )
            opened = None

        def begin(start):
            nonlocal opened, nxt
            opened = (nxt, start)
            nxt += 1

        for frame in segment["frames"]:
            # Match Rust f32 accumulation, selected-head order, then division.
            selected = np.asarray(frame["scores"], dtype=np.float32)[indices]
            scores = np.zeros(selected.shape[1], dtype=np.float32)
            for row in selected:
                scores += row
            scores /= np.float32(len(indices))
            start = frame["start"]
            if not frame["voiced"]:
                if opened:
                    pos, _ = opened
                    current = word_units[pos]
                    future = any(s > scores[current] for s in scores[current + 1 :])
                    final = pos + 1 == len(word_units)
                    punct = any(
                        u["word"] is None and not u["synthetic"]
                        for u in units[current + 1 :]
                    )
                    if future or (final and not punct):
                        close(start)
            elif opened is None:
                if nxt < len(word_units):
                    begin(start)
            elif nxt < len(word_units):
                current = word_units[opened[0]]
                if scores[word_units[nxt]] > scores[current] or (
                    scores[current] < np.float32(0.001)
                    and any(scores[k] > scores[current] for k in word_units[nxt + 1 :])
                ):
                    close(start)
                    begin(start)
        if opened:
            close(segment["end"])
    return events


def percentile(values, fraction):
    if not values:
        return float("inf")
    s = sorted(values)
    p = (len(s) - 1) * fraction
    lo = int(p)
    hi = math.ceil(p)
    return s[lo] + (s[hi] - s[lo]) * (p - lo)


def metrics(cases, heads):
    start = []
    end = []
    total = matched_count = 0
    details = []
    for data, asr in cases:
        source = [
            u["word"]["text"]
            for s in data["segments"]
            for u in s["units"]
            if u["word"] is not None
        ]
        ref, missing, extra = match_reference(source, asr["words"])
        events = replay(data, heads)
        matched = [e for e in events if e["index"] in ref]
        for e in matched:
            a, b = ref[e["index"]]
            start.append(abs(e["start"] - a))
            end.append(abs(e["end"] - b))
        total += len(source)
        matched_count += len(matched)
        repeated = [
            j
            for j in extra
            if j and key(asr["words"][j]["word"]) == key(asr["words"][j - 1]["word"])
        ]
        details.append(
            {
                "id": data["id"],
                "voice": data["voice"],
                "source_words": len(source),
                "matched_words": len(matched),
                "asr_missing": [source[i] for i in missing],
                "asr_extra": [asr["words"][i]["word"] for i in extra],
                "asr_repeated": [asr["words"][i]["word"] for i in repeated],
                "timestamp_missing": len(source) - len(events),
                "zero_duration_events": [e for e in events if e["end"] <= e["start"]],
            }
        )
    result = {
        "coverage": matched_count / total,
        "source_words": total,
        "matched_words": matched_count,
        "cases": details,
    }
    for name, values in [("start", start), ("end", end), ("pooled", start + end)]:
        result[name + "_mean"] = sum(values) / len(values) if values else float("inf")
        result[name + "_p95"] = percentile(values, 0.95)
    result["asr_missing"] = sum(len(d["asr_missing"]) for d in details)
    result["asr_extra"] = sum(len(d["asr_extra"]) for d in details)
    result["asr_repeated"] = sum(len(d["asr_repeated"]) for d in details)
    result["timestamp_missing"] = sum(d["timestamp_missing"] for d in details)
    return result


def comparison_diagnostics(old_cases, new_cases, old_heads, new_heads):
    """Post-gate diagnosis only: never used for head selection or promotion."""

    def rows(case, heads):
        data, asr = case
        source = [
            u["word"]["text"]
            for s in data["segments"]
            for u in s["units"]
            if u["word"] is not None
        ]
        refs, _, _ = match_reference(source, asr["words"])
        result = {}
        for event in replay(data, heads):
            if event["index"] not in refs:
                continue
            a, b = refs[event["index"]]
            result[event["index"]] = {
                **event,
                "asr_start": a,
                "asr_end": b,
                "start_error": abs(event["start"] - a),
                "end_error": abs(event["end"] - b),
                "trace_path": data["trace_path"],
                "wav_path": data["wav_path"],
                "id": data["id"],
                "voice": data["voice"],
            }
        return result

    def summary(items):
        start = [r["start_error"] for r in items]
        end = [r["end_error"] for r in items]
        result = {"matched_words": len(items)}
        for label, values in [("start", start), ("end", end), ("pooled", start + end)]:
            result[label + "_mean"] = sum(values) / len(values) if values else None
            result[label + "_p95"] = percentile(values, 0.95) if values else None
        return result

    cases = []
    old_common = []
    new_common = []
    worst = []
    for old_case, new_case in zip(old_cases, new_cases):
        assert (old_case[0]["id"], old_case[0]["voice"]) == (
            new_case[0]["id"],
            new_case[0]["voice"],
        )
        a = rows(old_case, old_heads)
        b = rows(new_case, new_heads)
        common = sorted(a.keys() & b.keys())
        old_common.extend(a[i] for i in common)
        new_common.extend(b[i] for i in common)
        cases.append(
            {
                "id": old_case[0]["id"],
                "voice": old_case[0]["voice"],
                "baseline": summary(list(a.values())),
                "target": summary(list(b.values())),
                "common_source_indices": common,
                "baseline_common": summary([a[i] for i in common]),
                "target_common": summary([b[i] for i in common]),
            }
        )
        worst.extend(
            dict(row, bundle=generation)
            for generation, items in [("baseline", a), ("target", b)]
            for row in items.values()
        )
    return {
        "use": "diagnostic only; does not override frozen promotion gate",
        "per_case": cases,
        "baseline_common": summary(old_common),
        "target_common": summary(new_common),
        "worst_boundaries": sorted(
            worst, key=lambda r: max(r["start_error"], r["end_error"]), reverse=True
        )[:20],
    }


def prepare(out):
    import yaml

    assert not (out / "provenance.json").exists(), (
        "refusing to overwrite frozen provenance"
    )
    protocol = load_protocol()
    anna = json.loads(
        Path("/private/tmp/pocket-tts-calibration-assets/anna-assets.json").read_text()
    )
    provenance = {
        "source_files": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                ROOT / "scripts/calibrate_timestamp_heads.py",
                *[
                    ROOT / "crates/pocket-tts/src" / f
                    for f in [
                        "timestamp_calibration.rs",
                        "tts_model.rs",
                        "audio.rs",
                        "text.rs",
                        "models/flow_lm.rs",
                        "timestamps/attention.rs",
                        "timestamps/alignment.rs",
                        "timestamps/text.rs",
                    ]
                ],
            ]
        },
        "cargo_lock_sha256": sha(ROOT / "Cargo.lock"),
        "protocol_sha256": sha(PROTOCOL),
        "asr_manifest": json.loads(
            Path(
                "/private/tmp/pocket-tts-calibration-assets/asr-manifest.json"
            ).read_text()
        ),
        "bundles": {},
    }
    for model in MODELS:
        for generation in ["baseline", "target"]:
            fixture = ROOT / "assets/upstream-2026-11-09" / generation / model
            manifest = json.loads((fixture / "manifest.json").read_text())
            config = (
                fixture
                / ("rust-config.yaml" if generation == "baseline" else "config.yaml")
            ).read_text()
            old = yaml.safe_load(
                (
                    ROOT
                    / "assets/upstream-2026-11-09/baseline"
                    / model
                    / "rust-config.yaml"
                ).read_text()
            )["timestamp_heads"]
            if generation == "target":
                config += "\ntimestamp_heads: " + json.dumps(old) + "\n"
            revision = manifest["voice_state"]["revision"]
            config += "\npreset_voice_revision: " + revision + "\n"
            bundle = out / generation / model
            bundle.mkdir(parents=True, exist_ok=True)
            (bundle / "config.yaml").write_text(config)
            voices = [
                dict(name="alba", **manifest["voice_state"]),
                dict(
                    name="anna",
                    **next(
                        a
                        for a in anna
                        if a["model"] == model and a["generation"] == generation
                    ),
                ),
            ]
            assets = {k: manifest[k] for k in ["weights", "tokenizer"]}
            for asset in list(assets.values()) + voices:
                assert sha(asset["path"]) == asset["sha256"], asset["path"]
            identity = {
                "config_sha256": sha(fixture / "config.yaml"),
                "protocol_sha256": sha(PROTOCOL),
                "weights": assets["weights"]["uri"],
                "tokenizer": assets["tokenizer"]["uri"],
                "preset_revision": revision,
                "heads": old,
                "modes": ["independent_segments"],
            }
            provenance["bundles"][generation + "/" + model] = {
                "identity": identity,
                "assets": assets,
                "voices": voices,
                "config_sha256": sha(bundle / "config.yaml"),
            }
            for split in ["calibration", "evaluation"]:
                write(
                    bundle / (split + "-job.json"),
                    {
                        "output": str(bundle / split),
                        "config": str(bundle / "config.yaml"),
                        "weights": assets["weights"]["path"],
                        "tokenizer": assets["tokenizer"]["path"],
                        "temperature": protocol["generation"]["temperature"],
                        "decode_steps": protocol["generation"]["decode_steps"],
                        "voices": voices,
                        "cases": [
                            dict(
                                c,
                                seeds={
                                    voice: int.from_bytes(
                                        hashlib.sha256(
                                            (
                                                c["id"] + ":" + voice + ":20261109"
                                            ).encode()
                                        ).digest()[:8],
                                        "big",
                                    )
                                    for voice in ["alba", "anna"]
                                },
                            )
                            for c in protocol[split]
                        ],
                    },
                )
    write(out / "provenance.json", provenance)


def read_cases(out, generation, model, split):
    cases = []
    for case in load_protocol()[split]:
        for voice in ["alba", "anna"]:
            directory = out / generation / model / split
            stem = case["id"] + "-" + voice
            data = json.loads((directory / (stem + ".json")).read_text())
            data["id"] = case["id"]
            data["trace_path"] = str(directory / (stem + ".json"))
            data["wav_path"] = str(directory / (stem + ".wav"))
            assert replay(data, data["native_heads"]) == data["native_events"], (
                "offline/native replay mismatch: " + stem
            )
            cases.append(
                (data, json.loads((directory / (stem + ".asr.json")).read_text()))
            )
    return cases


def rank(out, models=MODELS):
    for model in models:
        assert not any(
            (out / g / model / "evaluation").exists() for g in ["baseline", "target"]
        ), "cannot change selection after heldout capture starts"
        cases = read_cases(out, "target", model, "calibration")
        heads = cases[0][0]["heads"]
        old = cases[0][0]["native_heads"]
        ranked = [{"heads": [h], "metrics": metrics(cases, [h])} for h in heads]
        order = lambda x: (
            -x["metrics"]["coverage"],
            x["metrics"]["pooled_mean"],
            x["metrics"]["pooled_p95"],
            x["heads"],
        )
        ranked.sort(key=order)
        candidates = [old] + [
            list(pair)
            for pair in itertools.combinations([x["heads"][0] for x in ranked[:6]], 2)
        ]
        ranked.extend({"heads": hs, "metrics": metrics(cases, hs)} for hs in candidates)
        ranked.sort(key=order)
        write(
            out / model / "selection.json",
            {
                "protocol_sha256": sha(PROTOCOL),
                "heads": ranked[0]["heads"],
                "ranked": ranked,
                "frozen_before_evaluation": True,
            },
        )
        print(
            model,
            ranked[0]["heads"],
            ranked[0]["metrics"]["coverage"],
            ranked[0]["metrics"]["pooled_mean"],
            flush=True,
        )


def evaluate(out, models=MODELS):
    for model in models:
        selection = json.loads((out / model / "selection.json").read_text())
        assert selection["protocol_sha256"] == sha(PROTOCOL)
        old_cases = read_cases(out, "baseline", model, "evaluation")
        new_cases = read_cases(out, "target", model, "evaluation")
        old = metrics(old_cases, old_cases[0][0]["native_heads"])
        new = metrics(new_cases, selection["heads"])
        provenance = json.loads((out / "provenance.json").read_text())
        identity = dict(
            provenance["bundles"]["target/" + model]["identity"],
            heads=selection["heads"],
        )
        result = {
            "identity": identity,
            "promoted": False,
            "diagnostics": comparison_diagnostics(
                old_cases,
                new_cases,
                old_cases[0][0]["native_heads"],
                selection["heads"],
            ),
            "model": model,
            "heads": selection["heads"],
            "baseline": old,
            "target": new,
            "gate": promotion_gate(old, new),
            "selection_sha256": sha(out / model / "selection.json"),
            "protocol_sha256": sha(PROTOCOL),
            "audible_acceptance": "pending user testing",
            "ambiguity_adjudication": "pending review of missing/extra words and frame/ASR boundary evidence",
        }
        write(out / model / "evaluation.json", result)
        print(model, result["gate"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "capture", "transcribe", "rank", "evaluate"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["calibration", "evaluation"])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--native-test")
    parser.add_argument(
        "--asr",
        type=Path,
        default=Path("/private/tmp/pocket-tts-calibration-assets/asr"),
    )
    args = parser.parse_args()
    out = args.output.resolve()
    if args.command != "prepare":
        assert json.loads((out / "provenance.json").read_text())[
            "protocol_sha256"
        ] == sha(PROTOCOL), "frozen protocol changed"
    if args.command == "prepare":
        prepare(out)
    elif args.command == "rank":
        rank(out, [args.model] if args.model else MODELS)
    elif args.command == "evaluate":
        evaluate(out, [args.model] if args.model else MODELS)
    elif args.command == "capture":
        import os

        assert args.split and args.native_test
        for model in [args.model] if args.model else MODELS:
            if args.split == "evaluation":
                assert (out / model / "selection.json").is_file(), (
                    "freeze candidate before heldout generation"
                )
            for generation in ["baseline", "target"]:
                job = out / generation / model / (args.split + "-job.json")
                subprocess.run(
                    [
                        args.native_test,
                        "tts_model::timestamp_calibration::export_timestamp_calibration",
                        "--ignored",
                        "--exact",
                        "--nocapture",
                    ],
                    env=dict(
                        os.environ,
                        POCKET_TTS_CALIBRATION_JOB=str(job),
                        RAYON_NUM_THREADS="4",
                    ),
                    check=True,
                )
    elif args.command == "transcribe":
        from faster_whisper import WhisperModel

        assert args.split
        asr = WhisperModel(
            str(args.asr),
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            local_files_only=True,
        )
        for model in [args.model] if args.model else MODELS:
            for generation in ["baseline", "target"]:
                directory = out / generation / model / args.split
                expected = [
                    directory / (case["id"] + "-" + voice + ".wav")
                    for case in load_protocol()[args.split]
                    for voice in ["alba", "anna"]
                ]
                assert all(p.is_file() for p in expected), "capture incomplete: " + str(
                    directory
                )
                for wav in sorted(expected):
                    target = wav.with_suffix(".asr.json")
                    if target.exists():
                        continue
                    segments, info = asr.transcribe(
                        str(wav),
                        language="en",
                        beam_size=5,
                        word_timestamps=True,
                        condition_on_previous_text=False,
                    )
                    segments = list(segments)
                    write(
                        target,
                        {
                            "wav_sha256": sha(wav),
                            "text": " ".join(s.text for s in segments),
                            "words": [
                                {
                                    "word": w.word.strip(),
                                    "start": w.start,
                                    "end": w.end,
                                    "probability": w.probability,
                                }
                                for s in segments
                                for w in s.words
                            ],
                        },
                    )
                    print(target, flush=True)


if __name__ == "__main__":
    main()
