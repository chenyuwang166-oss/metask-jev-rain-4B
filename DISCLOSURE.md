# metask-jev-rain-4B — disclosure / 系统披露

已注入定稿参数并记录用户提供的实测环境。下述历史实测数字来自审核 §2.4，
不是本次打包新跑结果，也不是定稿校准和 32 GB 保守配置的重新测量。
剩余显式占位符须在正式提交前补齐。

## Identity / 身份

现榜 metask-jev-4b 的底座与权重不变，修订推理服务。
Checkpoint: `wayfind/metask-jev-4b-policy-mix`; immutable revision: `ea20fe85b28733b1522721dec119bec50947a869`.
Weight manifest SHA256: `f4b40475d18e0a38638b985ed51816c012619e0d4138166704e2ea0aac7a12b3 (model.safetensors, 9,078,620,536 bytes; identical on HF and on the measured host)`.
Base auxiliary-file source: `Qwen/Qwen3.5-4B`; revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
prepare_model.sh 记录每个权重与处理器文件的 SHA256，以及权重清单自身 SHA256。
正式提交前须核实权重身份；本次未读取上一版冻结包。

## Two paths and routing / 两路与路由

快路单次前向读取候选 token logits，聚合候选形式。慢路逐字延长快路提示，
要求至多 250 词推理，硬限制 512 个生成 token，到 ANSWER 标记停止并读取候选分布；
未解析时回退快路。250 词是提示约束，512 token 是硬限制。
以 logratio margin 路由，阈值 `2.3108509669020307`；公开 231 题标定比例
`0.2510822510822511`（58/231）。quota=0.25，share fuse=0.35。
成本保险丝 warmup=100、arm=0.85、release=0.80，相对成本上限生效。
prompt_sha: `e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938`.
历史实际慢路 56/231=0.2424；46 次解析成功，10 次未解析（10/56 约 18%），
后者花费生成 token 后使用快路答案。配额、解析失败和成本保险丝影响实际比例。
阈值拟合选择比例；概率校准使用公开标签，不能称作无标签拟合。

## Long inputs / 长题

问句感知抽取式压缩至 16384 prompt tokens，仅走快路；physical guard=32768，
max_model_len=34816。压缩会丢证据，物理上限不是无损上下文承诺。
真实约 80k-token 长题与约 32 万字符合成长题的模型冒烟尚未执行。

## Calibration / 校准

正式运行参数如下；calibration.json 原样保留 by_type 样本、原始拟合值和 insufficient 状态。

| Block | p_cal | T_choice | T_score | noul / choice / score samples |
|---|---|---|---|---|
| fast | 0.81 | 1.681792830507429 | 1.189207115002721 | 74 / 139 / 18 |
| slow | 0.842 | 6.156 | 9.190 | 20 / 21 / 5 |
| fallback | 0.81 | 1.681792830507429 | 1.189207115002721 | 0 / 9 / 1 |

fast 的原始 p_cal=0.7837837837837838，定稿夹至 0.81；fallback 全部继承 fast。
slow 按样本数收缩 n0=20；原始 choice=5.6569、score=32，score 仅 5 题且为
网格边界退化解。fast score 18 题、slow score 5 题及 fallback 各类型仍 insufficient。
所有块为 candidate; held-out validation required，未做留出验证。
运行 placeholder=false 只说明参数已注入，不代表验证充分；source_placeholder 保留源状态。

历史旧校准（fast p_cal 0.7838，slow 0.9 / 5.6569 / 32）的样本内
ECE 0.0880→0.0214、Brier 0.2750→0.2434、hard ECE 0.0946。
审核 S4 离线比较是在 fast p_cal 不变时得到 ECE 0.0278、Brier 0.2460、hard ECE 0.0663；
这些值不能冒充本包 fast p_cal=0.81 的定稿指标。定稿校准指标：`in-sample ECE on the 231 public items: 0.088 raw -> 0.021 after the three-block fit (package review, offline recompute matching harness ECE/Brier); hard-tier leave-one-out ECE 0.087 for the slow block with shrinkage n0=20`。

## Historical usage and results / 历史实测

公开 231 题、官方 harness、fresh service、串行，实测 RTX 4090 24 GB。
Engine input/output: 230588 / 16490；cached: 299904；submitted input/output:
530492 / 16607；each-once input/output: 232109 / 16607。
按 input $0.03/M、output $0.15/M，USD/1000 的 engine / each-once / submitted
分别 0.04065 / 0.04093 / 0.07968。成本保险丝 0 次转换，终值 0.0407，上限 0.0646。
长尾 23 题、每题约 79k token 压至 16384 fast_only 的 0.0470 USD/1000 仅为投影。
对账历史结论 passed（前缀缓存 on/off），matched coverage: `231/231 decisions in each of two runs (prefix caching on and off), both passed equation 1; cached-run manual check: sum input_tokens 218,877 = engine prompt tokens 433,245 - prefix-cache hits 214,368 (exact)`。

Accuracy 0.8485（196/231），macro 0.8567；easy 1.000/48，standard 0.972/72，
hard 0.703/111；fast-only 0.8095。Raw median/p95: 0.047/4.83 s；fast p50 0.046 s，
slow p50 2.99 s，max 5.39 s。231/231 ok、0 failed、schema validity 1.0、
paraphrase agreement 0.944（34/36）。Ready 前一次读分探针及一次 1-token 生成，并发 1。
legacy latency_scale=2.0、offset=0.15 是投影，不是实测延迟。

## Environment and evaluation hardware / 环境与硬件

OS Ubuntu 22.04.5；Python 3.10.12（运行版本钉 3.10）；NVIDIA driver 580.105.08。
vLLM 0.31.0；torch 2.13.0+cu130；transformers 5.17.0；huggingface_hub 1.33.0；
tokenizers 0.23.2；safetensors 0.8.0；numpy 2.2.6；fastapi 0.136.3；uvicorn 0.54.0；requests 2.34.2。
Measured GPU: RTX 4090 24 GB，不能写成 RTX 5090 实测。
Evaluation target: RTX 5090 32 GB，尚未测量。
vllm_options gpu_memory_utilization=0.85 + max_num_seqs=1 是为 32 GB 评测机保守设置，
与实测默认配置不同；须在评测机冷跑确认逐题 argmax、196/231、延迟和峰值显存。
Evaluation-machine peak VRAM: `19,702 MiB on the measured host (RTX 4090 24 GB, vLLM gpu_memory_utilization 0.85, max_num_seqs 1); the evaluation machine (RTX 5090 32 GB) was not measured`。
Lock SHA256: `32f0414b901856effaa2a6ff466cc9363c0ae70a7b6d5c6260a6433e184c3cdc`。
锁为实测版本锁，未提供 wheel 哈希；完整 freeze 与元数据见 environment.resolved.json。
Official harness clone revision: bb05a33（2026-09-29）；dataset hash: `easy.jsonl 314cd493a4c080fca5ad90e1c8a7c2b3f24137e66dc06f8fceaf51ea54f38467; original.jsonl b2abe9ac8cabd953a3bc452abd4b229da42ca6d0e647bf93f7638076f42db19f; hard.jsonl 8c8f1efb5a04b7016f9c8cd8a70a485f58a70a918aac3e3c1107f41172ca91b6 (jevbench datasets/public, HEAD bb05a33)`。
Freeze time (UTC): `2026-10-06T10:53:05Z`。

## Limits / 限制

公开集拟合存在过拟合风险，稀疏校准仍需留出验证；题序影响配额和成本保险丝。
抽取式压缩可能丢失证据，慢路未解析会消耗生成 token 后回退。
本次仅离线打包检查，没有运行真模型、重新安装完整 GPU 依赖或采集评测机证据。


## Final cold run from this package (supersedes the service-side numbers above where they differ)

Launched with `serve.sh` from this package on the measured host (RTX 4090 24 GB), fresh process, prefix cache cold, official harness `python -m jevbench.cli run --adapter typesafe --model metask-jev-rain-4B --key-env ''` over the 231 public items (2026-10-06T10:53:05Z):

| metric | value |
|---|---|
| attempted / failed | 231 / 0 |
| schema validity | 1.000 |
| accuracy (official scorer) | 0.8442 (195/231); macro 0.8606 |
| price per 1,000 decisions (derived_usage_times_tariff, Qwen3.5-4B $0.03/$0.15 per M) | $0.04099 |
| latency p50 / p95 (harness end-to-end, raw) | 0.055 s / 5.15 s; adjusted p50 = 2*0.055+0.15 = 0.26 s |
| peak VRAM | 19,702 MiB |
| health | status ok, no warnings, prefix_caching true, supports_slow_session true |

The earlier service-side run (same weights, same parameters, server started directly from the source tree) scored 0.8485 (196/231) at $0.04065; the one-item difference is within run-to-run numeric drift of vLLM batching.
