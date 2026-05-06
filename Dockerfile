# Stage 1: Build Web UI
FROM oven/bun:1 AS frontend-builder
WORKDIR /app
COPY crates/pocket-tts-cli/web ./
RUN bun install
RUN bun run build

# Stage 2: Build Rust
FROM rust:1.92-bullseye AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace files
COPY ./ ./

# Copy built frontend assets from previous stage
COPY --from=frontend-builder /app/dist ./crates/pocket-tts-cli/web/dist

# Build the project in release mode
RUN cargo build --release

# =============================================================================
# Runtime
# =============================================================================
FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/pocket-tts-cli /usr/local/bin/pocket-tts
COPY --from=builder /build/crates/pocket-tts/config /app/config

# Patch configs to use the public (no-auth) model path as the default.
# This makes each language use weights_path_without_voice_cloning instead of weights_path.
RUN for cfg in /app/config/*.yaml; do \
        sed -i 's|^weights_path: hf://kyutai/pocket-tts/|#weights_path: hf://kyutai/pocket-tts/|' "$cfg" && \
        sed -i 's|^weights_path_without_voice_cloning:|weights_path:|' "$cfg"; \
    done

WORKDIR /app

# Pre-cache the default English model and alba voice during build
RUN pocket-tts generate --language english --text "Initialize cache" && rm -f output.wav

EXPOSE 8000

ENTRYPOINT ["pocket-tts"]
CMD ["serve", "--language", "english", "--host", "0.0.0.0", "--port", "8000"]
