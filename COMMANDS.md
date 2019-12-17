# Commands

## Add paths to CUDA dependencies when using Tensorflow with CUDA

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64

## Set Tensorflow logging level to only output Errors

export TF_CPP_MIN_LOG_LEVEL = 2
