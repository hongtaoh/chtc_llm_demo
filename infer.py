#!/usr/bin/env python3
"""
LLM Inference script for CHTC GPU Lab
"""
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    print("=" * 60)
    print("Environment Info")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Configuration
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    prompts_path = Path(os.environ.get("PROMPTS_FILE", "prompts.txt"))
    out_dir = Path(os.environ.get("OUT_DIR", "responses"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Job identity
    cluster = os.environ.get("CLUSTER", "local")
    process = os.environ.get("PROCESS", "0")
    out_path = out_dir / f"responses_{cluster}_{process}.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Device: {device}")
    print(f"Model: {model_id}")

    t0 = time.time()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True
    )

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()

    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    # Load prompts
    if not prompts_path.exists():
        print(f"ERROR: Prompts file not found: {prompts_path}")
        sys.exit(1)

    prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
    print(f"Found {len(prompts)} prompts")

    # Generation settings
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "200"))
    temperature = float(os.environ.get("TEMPERATURE", "0.8"))

    # Run inference
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"MODEL_ID: {model_id}\n")
        f.write(f"DEVICE: {device}\n")
        f.write(f"NUM_PROMPTS: {len(prompts)}\n")
        f.write(f"LOAD_TIME: {load_time:.2f}s\n")
        f.write(f"START_TIME: {time.ctime()}\n")
        f.write("=" * 80 + "\n\n")

        for i, prompt in enumerate(prompts, start=1):
            print(f"[{i}/{len(prompts)}] Processing: {prompt[:50]}...")
            gen_start = time.time()

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_time = time.time() - gen_start

            print(f"    Generated in {gen_time:.2f}s")

            f.write(f"[{i}] PROMPT:\n{prompt}\n\n")
            f.write(f"[{i}] RESPONSE:\n{decoded}\n")
            f.write(f"[{i}] TIME: {gen_time:.2f}s\n")
            f.write("\n" + "-" * 80 + "\n\n")

        total_time = time.time() - t0
        f.write(f"END_TIME: {time.ctime()}\n")
        f.write(f"TOTAL_SECONDS: {total_time:.2f}\n")

    print(f"Done! Total time: {total_time:.2f}s")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()