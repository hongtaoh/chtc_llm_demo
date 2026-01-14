#!/bin/bash
set -e

echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "========================================"

export PYTHONNOUSERSITE=1

# Access staging DIRECTLY
ENV_TARBALL="/staging/hhao9/llm.tar.gz"
MODEL_TARBALL="/staging/hhao9/my_models/Qwen.tar.gz"

# Extract conda env
echo "Extracting conda environment from staging..."
mkdir -p .conda_env
tar -xzf "$ENV_TARBALL" -C .conda_env

# Extract model
echo "Extracting model from staging..."
tar -xzf "$MODEL_TARBALL"

PYTHON_EXEC=".conda_env/bin/python"

# Setup HF
export HF_HOME="$(pwd)/hf_home"
export HF_OFFLINE=1
mkdir -p "$HF_HOME"

# Verify
echo "Verifying..."
"$PYTHON_EXEC" -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Run
export MODEL_ID="./Qwen-0.5B"
"$PYTHON_EXEC" infer.py

echo "Done!"