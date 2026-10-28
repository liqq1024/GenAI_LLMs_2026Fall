# Recommended Colab model and environment

Model ID: `Qwen/Qwen2.5-0.5B-Instruct` (downloaded on first execution, not bundled). Use a Colab T4 GPU if available; free-tier GPU access is not guaranteed. Python 3.10+ recommended. The starter installs `transformers`, `peft`, `accelerate`, and `pandas` and records versions. This is a teaching configuration, not a guarantee of identical runtime across Colab sessions.

Why this model: small enough for an introductory LoRA experiment, instruction-tuned, supported by Hugging Face and PEFT. The notebook uses causal-LM supervised loss only on the target response tokens. No paid API key is required.

Training defaults: LoRA r=8, alpha=16, dropout=0.05, target modules `q_proj`, `v_proj`; one epoch, batch size 2, gradient accumulation 4, max length 192, LR 2e-4. Reduce batch size to 1 if out of memory. For CPU-only runs, skip LoRA and request an institutional GPU or approved alternative.

Run order: upload `train.csv`, `test.csv`, and `evaluation.py` into Colab's `/content/`, then run all notebook cells. Colab files are ephemeral: download the resulting CSVs and adapter.
