# Pocket TTS Metal profile artifacts (2026-05-06)

Workload: `pocket-tts-cli generate --language english --voice alba --temperature 0 --use-metal` using the long paragraph in `/tmp/pocket_tts_long_paragraph.txt`.

Large `.trace` bundles are kept in `/tmp/pocket_tts_profiles/` and are not copied here:

- `/tmp/pocket_tts_profiles/pocket_tts_metal_shader.trace`
- `/tmp/pocket_tts_profiles/pocket_tts_metal_cpu_time.trace`

Artifacts:

- `metal-shader-flamegraph.svg` — shader/GPU interval flamegraph (fallback mode; shader profiler rows were unavailable on this device/counter profile).
- `metal-system-cpu-flamegraph.svg` — CPU flamegraph extracted from the Metal System Trace.
- `metal-run-cpu-time-flamegraph.svg` — CPU Time Profiler flamegraph for the same Metal run.
- `gpu-summary.txt` — `trace-gpu.py` summary.
- `shader-hotspots.txt` — `trace-shader.py hotspots` summary.
- `cpu-time-summary.txt` — `trace-analyze.py summary` for CPU Time Profiler.
