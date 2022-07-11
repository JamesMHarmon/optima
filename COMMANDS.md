# Commands

## Add paths to CUDA dependencies when using Tensorflow with CUDA

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64

## Set Tensorflow logging level to only output Errors

export TF_CPP_MIN_LOG_LEVEL = 2

## Create Docker train image

cargo build --release && cp ./target/release/libreplay_buffer.so ./model_py/replay_buffer.so
docker build -f docker/train.Dockerfile -t quoridor_engine/train:latest .
docker run --rm -it -p 8888:8888 --gpus all --mount type=bind,source="${PWD}/Arimaa_runs",target=/Arimaa_runs -e TF_FORCE_GPU_ALLOW_GROWTH=true quoridor_engine/train:latest

cargo build --release && cp ./target/release/libreplay_buffer.so ./model_py/replay_buffer.so && docker cp model_py/ determined_brattain:/tf

## Limit Rayon threads

RAYON_NUM_THREADS = 16
TOKIO_THREADS = 16