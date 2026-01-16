# Run full Python vs Rust benchmark suite
$texts = @(
    "Hello world",
    "This is a medium length sentence for benchmarking.",
    "The quick brown fox jumps over the lazy dog. " * 3
)

# Ensure release build is up to date
Write-Host "Building Release..."
cargo build --release -p pocket-tts-cli

# Create temp directory
$tmpDir = ".bench_tmp"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

foreach ($i in 0..($texts.Length - 1)) {
    $text = $texts[$i]
    Write-Host "`n=== Benchmark $($i + 1): $($text.Length) chars ===" -ForegroundColor Cyan
    
    # Create batch files to avoid PowerShell quoting hell when passing to hyperfine
    $pyCmd = "uv run pocket-tts generate --text ""$text"" --output-path bench_py.wav"
    $rsCmd = "cargo run --release -p pocket-tts-cli -- generate --text ""$text"" --output bench_rs.wav"
    
    $pyBat = "$tmpDir\bench_py.bat"
    $rsBat = "$tmpDir\bench_rs.bat"
    
    Set-Content -Path $pyBat -Value $pyCmd
    Set-Content -Path $rsBat -Value $rsCmd
    
    # Verify content (debug)
    # Get-Content $pyBat
    
    hyperfine --warmup 1 --runs 3 `
        "$pyBat" `
        "$rsBat"
        
    # Cleanup logs/wavs if needed, but hyperfine overwrites
}

# Cleanup
Remove-Item -Recurse -Force $tmpDir
