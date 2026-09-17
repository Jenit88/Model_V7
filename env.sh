#!/bin/bash
# Environment for the Model_v7 (model_v7.py) training venv on this laptop.
# Shares the pcb62 venv with the V7 project -- same TF/Keras stack -- but
# keeps its own output directory and tmux session so the two never collide.
#   source env.sh
#
# Two things here are load-bearing and were found by diagnosis, not by taste:
#
# 1. LD_LIBRARY_PATH. opencv's loader replaces LD_LIBRARY_PATH with its own
#    directory, after which TensorFlow's dlopen of libcusolver.so.11 fails even
#    though the file is present in site-packages/nvidia/cusolver/lib. TF then
#    reports only "Cannot dlopen some GPU libraries" and silently falls back to
#    CPU. Exporting every site-packages/nvidia/*/lib up front makes the load
#    order irrelevant.
#
# 2. CUDA_CACHE_*. TensorFlow 2.21 ships no cubins for compute capability 12.0
#    (Blackwell), so every kernel is JIT-compiled from PTX on first use — TF
#    itself warns this "could take 30 minutes or longer". The result is cached
#    on disk, so that cost is paid once as long as the cache is big enough to
#    hold it; the 256 MB default is not.

PCB_ENV="${PCB_ENV:-$HOME/envs/pcb62}"
PCB_SITE="$PCB_ENV/lib/python3.11/site-packages"

_nvidia_libs=$(find "$PCB_SITE/nvidia" -maxdepth 3 -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${_nvidia_libs}${PCB_SITE}/tensorflow:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _nvidia_libs

export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
export CUDA_CACHE_MAXSIZE=4294967296
mkdir -p "$CUDA_CACHE_PATH"

# Keep the log readable; the PTX-JIT warning repeats per device init.
export TF_CPP_MIN_LOG_LEVEL=1
# oneDNN reorders float accumulation on the CPU ops; off for reproducibility.
export TF_ENABLE_ONEDNN_OPTS=0

# Dataset lives on the Windows drive (read once, at prepare time). The prepared
# memmapped arrays that every training step reads must be on WSL's ext4 disk:
# random access to a 10 GB memmap over the 9p /mnt/c mount is an order of
# magnitude slower and would starve the GPU.
export PCB_DATASET_ROOT="/mnt/c/Users/u117134/Desktop/Data_preprocessing/1000_images/Split_Data"
export PCB_ARRAY_DIR="${PCB_ARRAY_DIR:-$HOME/data/pcb_v62_arrays_repaired}"
export PCB_MODEL_OUTPUT_DIR="${PCB_MODEL_OUTPUT_DIR:-$HOME/Models/Model_v7_scratch_rtx}"
export PCB_TMUX_SESSION="${PCB_TMUX_SESSION:-v7}"

export PCB_PYTHON="$PCB_ENV/bin/python"

echo "pcb62 environment ready"
echo "  python  : $PCB_PYTHON"
echo "  dataset : $PCB_DATASET_ROOT"
echo "  arrays  : $PCB_ARRAY_DIR"
echo "  output  : $PCB_MODEL_OUTPUT_DIR"
