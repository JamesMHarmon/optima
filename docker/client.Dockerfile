# Builds the Optima "client" binary (UGI/self-play/arena) and runs it on top of the
# same tensorflow/tensorflow:2.8.0-gpu image in both stages, so the CUDA/cuDNN runtime
# the Rust binary links against at build time is guaranteed to match what it runs
# against later - no separate CUDA base image to version-match by hand.
#
# Build and run on the SAME machine that will play the game: the Rust compiler's
# -C target-cpu=native flag (see ../.cargo/config.toml) optimizes for whatever CPU
# the build runs on.
#
# Usage:
#   docker build -f docker/client.Dockerfile -t optima-quoridor .
#   docker run --gpus all -it --rm optima-quoridor

# ---- builder ----
FROM tensorflow/tensorflow:2.8.0-gpu AS builder

ARG TENSORFLOW_VERSION=2.8.0

# This base image bakes in an NVIDIA CUDA apt repo whose signing key NVIDIA rotated
# in 2022, so apt-get update fails on it. We don't need it (CUDA is already in the
# base image; libtensorflow is fetched by direct download below), so drop it.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/tensorRT.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/libtensorflow.tar.gz \
        "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-${TENSORFLOW_VERSION}.tar.gz" \
    && tar -C /usr/local -xzf /tmp/libtensorflow.tar.gz \
    && rm /tmp/libtensorflow.tar.gz \
    && ldconfig

RUN curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --default-toolchain stable \
    && rm /tmp/rustup-init.sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

RUN cargo build --release -p client

# ---- runtime ----
FROM tensorflow/tensorflow:2.8.0-gpu

COPY --from=builder /usr/local/lib/libtensorflow.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libtensorflow_framework.so* /usr/local/lib/
RUN ldconfig

WORKDIR /app
COPY --from=builder /build/target/release/client ./client

ARG MODEL_FILE=10b256f_00568.tar.gz
COPY ${MODEL_FILE} ./${MODEL_FILE}

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV BOT_MODEL_DIR=/app
ENV BOT_MODEL_NAME=${MODEL_FILE}

ENTRYPOINT ["./client"]
CMD ["ugi"]
