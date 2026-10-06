# Stage 1: Build Web UI
FROM oven/bun:1 AS frontend-builder
WORKDIR /app/crates/pocket-tts-cli/web
COPY crates/pocket-tts/config /app/crates/pocket-tts/config
COPY crates/pocket-tts-cli/web ./
RUN bun install
RUN bun run build

# Stage 2: Build Rust
FROM rust:1.92-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace files
COPY ./ ./

# Copy built frontend assets from previous stage
COPY --from=frontend-builder /app/crates/pocket-tts-cli/web/dist ./crates/pocket-tts-cli/web/dist

# Build only the executable shipped in this image, not the Python bindings.
RUN cargo build --release --locked -p pocket-tts-cli

# =============================================================================
# Runtime
# =============================================================================
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/pocket-tts-cli /usr/local/bin/pocket-tts
COPY --from=builder /build/crates/pocket-tts/config /app/config

WORKDIR /app

# Pre-cache English and alba; the loader falls back to public preset-only weights
# without changing the configs or incorrectly enabling voice-cloning capability.
RUN pocket-tts generate --language english --text "Initialize cache" && rm -f output.wav

EXPOSE 8000

ENTRYPOINT ["pocket-tts"]
CMD ["serve", "--language", "english", "--host", "0.0.0.0", "--port", "8000"]
