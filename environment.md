# Measured environment and installation contract

Measured on Ubuntu 22.04.5, Python 3.10.12, NVIDIA driver 580.105.08,
RTX 4090 24 GB. Runtime Python is restricted to 3.10 (including installation).
vLLM 0.31.0; torch 2.13.0+cu130; transformers 5.17.0;
huggingface_hub 1.33.0; tokenizers 0.23.2; safetensors 0.8.0;
numpy 2.2.6; fastapi 0.136.3; uvicorn 0.54.0; requests 2.34.2.

`requirements.measured.lock` is the supplied 201-entry freeze. `resolve_env.py`
reproduces `requirements.lock` and `environment.resolved.json` offline from it.
All 201 entries are retained; the only override is torch 2.13.0 to the explicitly
supplied measured build 2.13.0+cu130. No packages were guessed to be uninstallable.
`install.sh` installs this exact version lock and runs `pip check`; no fresh
version resolution or silent downgrade is intended. The public CUDA 13.0 wheel
index is included for the measured torch build.

Lock SHA256: `32f0414b901856effaa2a6ff466cc9363c0ae70a7b6d5c6260a6433e184c3cdc`.
This is a version lock, not a wheel-hash lock: the supplied freeze has no artifact
hashes. Do not claim `--require-hashes` verification or a successful clean install.
The full dependency set, wheel availability and clean installation on Linux/Python
3.10 remain unverified in this task. Installation or `pip check` failure is fatal.
OS, driver and the model/processor files are outside pip's scope. The torch CUDA
build does not establish a separately installed system CUDA toolkit version.

Serving and selftest set VLLM_USE_FLASHINFER_SAMPLER=0 and
VLLM_ATTENTION_BACKEND=FLASH_ATTN, as in the supplied measurements. Both disable
usage telemetry and enable HF/transformers offline mode. Run preparation before
serving, with the installed environment's Python and network access.

The four required auxiliary files are preprocessor_config.json,
video_preprocessor_config.json, merges.txt and vocab.json. Missing files are copied
from Qwen/Qwen3.5-4B at the supplied immutable base revision; existing checkpoint
files are preserved. Weights come from wayfind/metask-jev-4b-policy-mix.

Package vllm_options are gpu_memory_utilization=0.85 and max_num_seqs=1,
conservative settings for the 32 GB evaluation GPU. The measured 24 GB RTX 4090
run used default options instead. This difference has NOT been cold-run verified
on the evaluation machine. The effective max_model_len is 34816, physical prompt
guard 32768 and soft context budget 16384. Preserve fitted prompt/limit identity.
