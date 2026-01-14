#!/bin/bash
set -e

MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME="Qwen-0.5B"

cd /staging/hhao9/my_models

rm -rf "$MODEL_NAME" "${MODEL_NAME}.tar.gz"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_REPO',
    local_dir='$MODEL_NAME'
)
"

tar -czf "${MODEL_NAME}.tar.gz" "$MODEL_NAME"
rm -rf "$MODEL_NAME"

echo "Done! Model at /staging/hhao9/my_models/${MODEL_NAME}.tar.gz"