# metask-jev-rain-4B

Offline inference service for the official JevBench TypeSafe adapter. Use the code
commit given in the submission email. Model identity, data use, accounting and
measurement evidence are in [DISCLOSURE.md](DISCLOSURE.md).

## Requirements

Use Linux amd64 with Docker and NVIDIA Container Toolkit, one exclusively assigned
GPU with at least 24 GB VRAM and compute capability >=8.0, and NVIDIA driver 580 or
newer. The commands use the host's GPU 0. The runtime image is:

```text
vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
```

This is the linux/amd64 image of the official vLLM release tag `v0.31.0`.
The image has index digest
`sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`,
linux/amd64 manifest digest
`sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`
(used in the commands), and config digest
`sha256:c76d0e2225a4b1cb1e2109ace39639f55e714abd1a7a427acc8b0bbd7f6a83b3`
(the classic Docker image store shows the config digest as the image ID;
the containerd image store identifies images by manifest or index digest,
depending on the reference used when pulling).

The image, code, model folder, task files and harness must already be on the evaluation
host, with the exact repository digest used in step 1 resolvable locally.
The Docker client must not inject HTTP proxy variables into the container.
Running the service requires no installation and no network access. The
image supplies Python 3.12.3, vLLM 0.31.0 and the checked inference dependencies.

## Model folder

`MODEL` must contain the files of
[wayfind/metask-jev-4b-policy-mix](https://huggingface.co/wayfind/metask-jev-4b-policy-mix/tree/ea20fe85b28733b1522721dec119bec50947a869)
as regular files, mounted read-only. On the connected preparation host, download
that revision with `hf download wayfind/metask-jev-4b-policy-mix --revision ea20fe85b28733b1522721dec119bec50947a869 --local-dir "$MODEL"`.
A Hugging Face cache `snapshots/<sha>` directory contains relative links into
`../../blobs` and cannot be mounted on its own. No download runs during evaluation.

Startup checks SHA256 for `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `generation_config.json` and `chat_template.jinja`.
It checks `model.safetensors` against its size of 9,078,620,536 bytes; it does not
recompute the full weight SHA256 at startup. The package includes four auxiliary
files from
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
in `model_aux/`. Startup verifies their hashes and builds a symlink view under the
state directory. The view contains only the five identity files, the weight file
and the four auxiliary files. When all four auxiliary files are absent from
`MODEL`, it uses the package copies; when all four are present, their hashes must
match. It never writes to `MODEL`. Partial auxiliary sets, extra loading/tokenizer
files, and identity or auxiliary hash mismatches are errors.

## Run

Copy the commands as they are in the same Bash terminal. First set four absolute
host paths and the task list:

```bash
PKG=/abs/path/to/metask-jev-rain-4B          # this repository, mounted read-only
MODEL=/abs/path/to/metask-jev-4b-policy-mix  # the regular-file model folder above, mounted read-only
WORK=/abs/path/to/new-work-dir              # new for each evaluation; the only place files are written
JEVBENCH=/abs/path/to/jevbench               # the pinned harness checkout below, mounted read-only
TASKS=/jevbench/datasets/public/easy.jsonl,/jevbench/datasets/public/original.jsonl,/jevbench/datasets/public/hard.jsonl  # container paths
```

`JEVBENCH` must contain the official harness at
[bb05a335bc809e61b20c0f745d25499a82b326fc](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc).
`TASKS` is the comma-separated list of task files as seen inside the container:
a file below `$JEVBENCH` is below `/jevbench`, a file below `$WORK` is below `/work`.
The value above is the public set. Task paths cannot contain commas. `WORK` holds
nothing but any task files placed there and must be separate from the other mounts.
Use a local filesystem writable by the container user, supporting symbolic links
and executable/JIT artifacts; it must not be mounted `noexec`.

1. Start the service:

   ```bash
   docker run -d --name jev4b --init --gpus '"device=0"' --network none --read-only --shm-size 8g \
     -v "$PKG":/pkg:ro -v "$MODEL":/model:ro -v "$WORK":/work -v "$JEVBENCH":/jevbench:ro \
     --entrypoint bash vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b \
     /pkg/serve.sh --model /model --state-dir /work/state --run-id run1
   ```

2. Follow the log until the service is ready:

   ```bash
   docker logs -f jev4b
   ```

   Startup loads and compiles the model. When a line starting with `{"status": "ok"`
   appears, press Ctrl-C (this ends only the log display) and go to step 3. If the
   log ends before that line, the container has exited: the reason is in
   `docker logs jev4b` and `$WORK/state/run1-startup/attempt-*.log`; go to step 4.
   After correcting the cause, use a new `WORK` and restart from step 1.

3. Run the official harness in the same container, then summarize:

   ```bash
   docker exec -e PYTHONPATH=/jevbench jev4b python3 -B -m jevbench.cli run --adapter typesafe \
     --endpoint http://127.0.0.1:8000 --model metask-jev-rain-4B --key-env '' --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl --raw-dir /work/run1/raw \
     --manifest /work/run1/manifest.json --cap-usd 15 --price-in-per-m 0.03 --price-out-per-m 0.15
   docker exec -e PYTHONPATH=/jevbench jev4b python3 -B -m jevbench.cli summarize --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl
   ```

4. Stop and remove the container:

   ```bash
   docker stop -t 40 jev4b && docker rm jev4b
   ```

Each evaluation uses a new `WORK` and a new container (step 4 removes the previous
one); the run ID stays `run1` because `WORK` is new.

The service writes files under `/work/state`, plus shared memory in `/dev/shm`:
per-attempt quota state, startup logs,
the model view, temporary files, home directory and compilation caches. The
harness writes its evidence to `/work/run1`. Each evaluation uses a new `WORK`;
reusing a run ID in the same state directory is rejected. Keep
the GPU exclusive and send harness requests serially.

Expected health: `status="ok"`, `warnings=[]`, `model_key="metask-jev-4b"`,
`prefix_caching=true`, `supports_slow_session=true`. `info.runtime` reports the
loaded versions, CUDA and GPU; `info.attention_backend` records the observed
backend. The public system name remains `metask-jev-rain-4B`; the model key is the
weight and calibration identity. `/engine_stats` is a read-only counter endpoint.

The service binds to `127.0.0.1`; `docker exec` shares its network namespace.
Startup diagnostics persist at `/work/state/run1-startup/` inside the container,
which is `$WORK/state/run1-startup/` on the host. This includes `attempt-N.log` and
`attempts.jsonl`; these files survive container removal in step 4. Docker's own
logs remain available through `docker logs jev4b` after an exit until that removal.
Each startup attempt has a fixed 1800-second deadline. A timeout aborts startup
rather than advancing to another backend. Model, state and output paths must be
absolute; state and model directories may neither contain one another nor overlap
the package. The fixed `/work/state` and `run1` paths meet the 70 UTF-8-byte startup
temporary-path limit (a run ID under `/work/state` can be at most 46 ASCII bytes).
Do not inject cache/configuration path overrides into the container: inherited
overrides outside `/work/state` are rejected. The isolated model-view path changes
with the run and enters vLLM's compilation cache key; reuse of compiled artifacts
across runs, or a faster second startup, is not guaranteed.

Maintenance and validation procedures are in [MAINTAINER.md](MAINTAINER.md).
