"""Lazy model backends; importing this module never imports or loads a model.

Both backends return unmasked next-token scores. vLLM returns raw log
probabilities, which differ from logits by one common additive constant; the
form-set logsumexp and subsequent label temperature are invariant to it.
"""

from __future__ import annotations

import math
import logging
import json
import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable


class BackendError(RuntimeError):
    """An unavailable or incompatible backend, with no approximation fallback."""


def _candidate_ids(values: Iterable[int]) -> list[int]:
    ids = list(dict.fromkeys(values))
    if not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("candidate_ids must contain nonnegative integer token IDs")
    return ids


def _max_new_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    return value


def _check_scores(scores: dict[int, float]) -> dict[int, float]:
    if any(math.isnan(value) or value == math.inf for value in scores.values()):
        raise BackendError("backend returned NaN or positive infinite candidate scores")
    return scores


def generation_audit(config):
    """Compare explicit service settings; never silently inherit checkpoint policy."""
    report={'verified':False,'status':'unavailable','differences':{},'sha256':None}
    model_path=config['model'].get('path')
    path=Path(os.path.expanduser(model_path))/'generation_config.json' if model_path else None
    if path is not None and path.is_file():
        try:
            data=path.read_bytes();reference=json.loads(data)
            if not isinstance(reference,dict):raise ValueError('Expected mapping')
            report['sha256']=hashlib.sha256(data).hexdigest()
            def safe(v):
                if v is None or isinstance(v,bool) or isinstance(v,(int,float)) and math.isfinite(v):return v
                if isinstance(v,list):return [safe(x) for x in v]
                return 'unsupported value'
            for key,value in config['generation'].items():
                if key not in ('seed','reasoning_words','max_new_tokens','wall_timeout_seconds') and key in reference and reference[key]!=value:
                    report['differences'][key]={'checkpoint':safe(reference[key]),'configured':safe(value)}
            report.update(verified=not report['differences'],status='mismatch' if report['differences'] else 'matched_explicit_keys')
        except (OSError,ValueError,TypeError):report['status']='unreadable'
    if not report['verified']:logging.warning('Generation reference %s; explicit configured parameters remain active',report['status'])
    return report


def install_forward_counter(backend):
    backend.forward_counters={'forward_tokens':0,'prefill_tokens':0,'uncounted_forwards':0}
    def count_forward(module,args,kwargs):
        try:
            ids=kwargs.get('input_ids',args[0] if args else None)
            if ids is None:ids=kwargs.get('inputs_embeds')
            if ids is None:raise ValueError('No input tensor')
            n=int(ids.shape[1]);past=kwargs.get('past_key_values')
            empty=past is None or (callable(getattr(past,'get_seq_length',None)) and past.get_seq_length()==0)
            backend.forward_counters['forward_tokens']+=n
            if empty:backend.forward_counters['prefill_tokens']+=n
        except Exception:
            backend.forward_counters['uncounted_forwards']+=1
            logging.warning('HF forward count unavailable; usage estimated')
    target=backend.model.get_base_model() if callable(getattr(backend.model,'get_base_model',None)) else backend.model
    hook=getattr(target,'register_forward_pre_hook',None)
    backend.stats_available=callable(hook)
    if backend.stats_available:backend.stats_hook=hook(count_forward,with_kwargs=True)
    else:logging.warning('HF forward hook unavailable; usage estimated')


class TokenizerMixin:
    tokenizer: Any

    def scan_digit_tokens(self):
        vocab=self.tokenizer.get_vocab() if hasattr(self.tokenizer,'get_vocab') else {}
        self.digit_start_ids={token for token in set(vocab.values())
                              if re.match(r'^\d',self.decode([token]).lstrip())}

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def prompt_ids(self, system: str, text: str, enable_thinking: bool = False) -> list[int]:
        # Match cache_hidden_v2: the instructions and context share a user turn.
        messages = [{"role": "user", "content": system + "\n\n" + text}]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return self.encode(prompt)


class TransformersBackend(TokenizerMixin):
    """Direct single-forward readout, and a separate bounded draft generation."""

    readout_output_tokens = 0

    def __init__(self, config: dict[str, Any]):
        model_cfg = config["model"]
        self.generation = dict(config["generation"])
        self.generation_reference=generation_audit(config)
        self.marker = config.get("prompt", {}).get("answer_marker", "ANSWER:")
        if model_cfg["quant"] not in {"int8", "bf16"}:
            raise ValueError("model.quant must be int8 or bf16")
        if not model_cfg["path"]:
            raise ValueError("model.path is required before loading a model")
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise BackendError("transformers backend requires torch and transformers") from exc
        self.torch = torch
        torch.manual_seed(self.generation["seed"])
        # expanduser keeps a Hub identifier's slash intact on Windows.
        path = os.path.expanduser(model_cfg["path"])
        cfg = transformers.AutoConfig.from_pretrained(path)
        declared_quant = getattr(cfg, "quantization_config", None)
        if declared_quant:
            if not isinstance(declared_quant, dict):
                declared_quant = declared_quant.to_dict()
            if model_cfg["quant"] == "bf16":
                raise BackendError("bf16 requires an unquantized checkpoint; base config declares quantization")
            if (declared_quant.get("quant_method") != "bitsandbytes"
                    or not declared_quant.get("load_in_8bit")
                    or declared_quant.get("load_in_4bit")):
                raise BackendError("int8 requires an unquantized base or a BitsAndBytes 8bit checkpoint")
        archs = list(getattr(cfg, "architectures", None) or [])
        if any("Qwen3_5ForConditionalGeneration" in name for name in archs):
            model_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
            if model_cls is None:
                raise BackendError("installed transformers lacks Qwen3_5ForConditionalGeneration")
        else:
            model_cls = transformers.AutoModelForCausalLM
        kwargs = {
            "dtype": torch.bfloat16,
            "device_map": model_cfg["device_map"],
            "low_cpu_mem_usage": True,
            "attn_implementation": model_cfg["attn_implementation"],
        }
        if model_cfg["quant"] == "int8":
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        self.scan_digit_tokens()
        self.model = model_cls.from_pretrained(path, **kwargs).eval()
        if model_cfg.get("adapter"):
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise BackendError("--adapter requires peft for the transformers backend") from exc
            self.model = PeftModel.from_pretrained(
                self.model, str(Path(model_cfg["adapter"]).expanduser()),
            ).eval()
        # A fresh config prevents checkpoint-specific suppression, forced tokens,
        # sampling filters or minimum-length policies from silently changing the
        # configured draft. Preserve special-token metadata, including EOS lists.
        special = {}
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            for source in (getattr(self.model, "generation_config", None),
                           getattr(self.model, "config", None), self.tokenizer):
                value = getattr(source, name, None)
                if value is not None:
                    special[name] = value
                    break
        self.prefix_caching=False
        install_forward_counter(self)
        self.draft_generation_config = transformers.GenerationConfig(**special, **{k:v for k,v in self.generation.items()
            if k not in ("reasoning_words","seed","max_new_tokens","wall_timeout_seconds")})
        eos = special.get("eos_token_id", [])
        self.eos_ids = eos if isinstance(eos, list) else [eos]

    def engine_stats(self):
        return {'source':'hf_forward_hook','prefix_caching':False,
                'counters':dict(self.forward_counters) if self.stats_available else {}}

    def open_session(self, ids):
        return CachedSession(self, ids)

    def slow_session(self, ids, unparsed_policy, meter, max_new_tokens=None):
        count=_max_new_tokens(max_new_tokens or self.generation['max_new_tokens'])
        tensor=self._input(ids)
        meter['prompt']+=len(ids); meter['engine_submitted']+=len(ids)
        meter['engine_input']=meter.get('engine_input',0)+len(ids)
        meter['engine_calls']=meter.get('engine_calls',0)+1
        meter['hf_prefill_tokens']=meter.get('hf_prefill_tokens',0)+len(ids)
        class MeterStreamer:
            first=True
            def put(inner,value):
                if inner.first: inner.first=False; return
                n=value.shape[-1]
                meter['output']+=n; meter['draft']+=n
                meter['engine_generated']=meter.get('engine_generated',0)+n
                if not getattr(inner,'counted',False):
                    meter['generate_nonempty_calls']=meter.get('generate_nonempty_calls',0)+1; inner.counted=True
            def end(inner): pass
        streamer=MeterStreamer()
        g=self.generation
        kwargs=dict(input_ids=tensor,attention_mask=self.torch.ones_like(tensor),
            generation_config=self.draft_generation_config,max_new_tokens=count,
            max_time=g.get("wall_timeout_seconds",60),
            do_sample=g['do_sample'],num_beams=1,num_return_sequences=1,
            repetition_penalty=g['repetition_penalty'],use_cache=True,
            return_dict_in_generate=True,stop_strings=[self.marker],tokenizer=self.tokenizer,
            streamer=streamer,pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if g['do_sample']: kwargs.update(temperature=g['temperature'],top_p=g['top_p'],top_k=g['top_k'])
        check_deadline(meter)
        try:
            with self.torch.inference_mode(): output=self.model.generate(**kwargs)
        except Exception:
            meter['engine_usage_unknown']=True
            raise
        check_deadline(meter)
        generated=output.sequences[0,len(ids):].tolist()
        # Transformers streams sampled IDs even when the next forward raises.
        if meter['draft']!=len(generated): raise BackendError('Generation streamer accounting mismatch')
        parsed=bool(re.search(r'(?m)^[ \t]*'+re.escape(self.marker)+r'$',self.decode(generated)))
        if not parsed and unparsed_policy=='fast': return None,False
        cache=output.past_key_values
        if cache is None or not generated: raise BackendError('Generation returned no reusable cache')
        session=object.__new__(CachedSession)
        session.backend,session.cache,session.length=self,cache,len(ids)+len(generated)-1
        session._log_probs=None
        session.meter=meter
        # Standard generate cache is one sampled token behind sequences.
        session.advance(generated[-1:])
        if not parsed:
            forced=self.encode('\n'+self.marker)
            meter['output']+=len(forced);meter['forced']+=len(forced)
            session.advance(forced)
        return session,parsed

    def _input(self, ids: Iterable[int]):
        values = list(ids)
        if not values:
            raise ValueError("prompt token IDs cannot be empty")
        embedding = self.model.get_input_embeddings()
        device = embedding.weight.device
        if getattr(device, "type", None) == "meta":
            device = self.model.device
        return self.torch.tensor([values], dtype=self.torch.long, device=device)

    def next_logits(self, ids: Iterable[int], candidate_ids: Iterable[int]) -> dict[int, float]:
        candidates = _candidate_ids(candidate_ids)
        tensor = self._input(ids)
        with self.torch.inference_mode():
            try:
                output = self.model(input_ids=tensor, use_cache=False, logits_to_keep=1)
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                output = self.model(input_ids=tensor, use_cache=False)
            logits = output.logits[0, -1, :]
            if max(candidates) >= logits.shape[0]:
                raise BackendError("candidate token ID exceeds the model output vocabulary")
            index = self.torch.tensor(candidates, dtype=self.torch.long, device=logits.device)
            selected = logits.index_select(0, index).float().cpu().tolist()
        return _check_scores(dict(zip(candidates, selected)))

    def generate(self, ids: Iterable[int], max_new_tokens: int) -> list[int]:
        count = _max_new_tokens(max_new_tokens)
        tensor = self._input(ids)
        generation = self.generation
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        kwargs = {
            "input_ids": tensor,
            "attention_mask": self.torch.ones_like(tensor),
            "generation_config": self.draft_generation_config,
            "max_new_tokens": count,
            "do_sample": generation["do_sample"],
            "num_beams": 1,
            "num_return_sequences": 1,
            "repetition_penalty": generation["repetition_penalty"],
            "use_cache": True,
            "return_dict_in_generate": False,
            "pad_token_id": pad_id,
        }
        if generation["do_sample"]:
            kwargs.update(temperature=generation["temperature"],
                          top_p=generation["top_p"], top_k=generation["top_k"])
        with self.torch.inference_mode():
            output = self.model.generate(**kwargs)
        return output[0, tensor.shape[1]:].tolist()


class VLLMBackend(TokenizerMixin):
    """Exact candidate readout through explicit raw logprob token IDs.

    The engine produces one probe token to expose the next-token distribution;
    callers must include readout_output_tokens in measured output-token usage.
    Normal probes gather only explicit candidate IDs. Full-vocabulary output
    is reserved for one logged recovery probe if a backend omits candidates.
    This implementation supports bf16 and an explicitly checked AWQ checkpoint.
    It rejects int8 because vLLM's
    BnB loader settings do not establish equivalence to the reference int8 base.
    """

    readout_output_tokens = 1
    engine_metering = True

    def __init__(self, config: dict[str, Any]):
        model_cfg = config["model"]
        if model_cfg["quant"] not in ("bf16","awq"):
            raise BackendError("vllm backend supports bf16 or AWQ; use transformers for int8")
        if not model_cfg["path"]:
            raise ValueError("model.path is required before loading a model")
        options = dict(model_cfg["vllm_options"])
        if config.get("limits"):
            options.setdefault("max_model_len",config["limits"]["physical_max_prompt_tokens"]+2048)
        reserved = {
            "model", "tokenizer", "dtype", "quantization", "quantization_config",
            "max_logprobs", "logprobs_mode", "seed", "enable_lora",
            "generation_config", "override_generation_config", "logits_processors",
            "skip_tokenizer_init", "speculative_config", "enable_prefix_caching", "disable_log_stats",
        }
        overlap = sorted(reserved.intersection(options))
        if overlap:
            raise ValueError("model.vllm_options cannot override readout contract: " + ", ".join(overlap))
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise BackendError("vllm backend requires an importable compatible vllm installation") from exc
        self.SamplingParams = SamplingParams
        self.marker=config.get("prompt",{}).get("answer_marker","ANSWER:")
        self.lock=threading.RLock()
        self.generation = dict(config["generation"])
        self.generation_reference=generation_audit(config)
        self.lora_request = None
        if model_cfg.get("adapter"):
            try:
                from vllm.lora.request import LoRARequest
                self.lora_request = LoRARequest(
                    "submission_v2_adapter", 1, str(Path(model_cfg["adapter"]).expanduser()),
                )
            except (ImportError, TypeError, ValueError) as exc:
                raise BackendError("installed vllm does not support the requested LoRA adapter API") from exc
        try:
            # These are correctness invariants, not configurable routing policy.
            self.engine = LLM(
                model=os.path.expanduser(model_cfg["path"]), dtype="bfloat16",
                quantization="awq" if model_cfg["quant"]=="awq" else None, seed=self.generation["seed"],
                max_logprobs=-1, logprobs_mode="raw_logprobs",
                generation_config="vllm", enable_lora=bool(self.lora_request), disable_log_stats=False,
                enable_prefix_caching=config.get("model",{}).get("enable_prefix_caching",True), **options,
            )
            # quantization=None permits checkpoint auto-detection in vLLM. Do
            # not label that run bf16 if the checkpoint activates quantization.
            active_model_cfg = self.engine.llm_engine.model_config
            if model_cfg["quant"]=="bf16" and active_model_cfg.quantization is not None:
                raise BackendError("bf16 requires an unquantized checkpoint; vllm activated quantization")
            if model_cfg['quant']=='awq' and active_model_cfg.quantization not in ('awq','awq_marlin'):
                raise BackendError('AWQ profile requires active AWQ quantization')
            self.quantization_kernel=active_model_cfg.quantization
            cache_cfg=getattr(getattr(self.engine.llm_engine,'vllm_config',None),'cache_config',None)
            self.prefix_caching=getattr(cache_cfg,'enable_prefix_caching',None)
            self.tokenizer = self.engine.get_tokenizer()
            self.scan_digit_tokens()
            self.probe_params = SamplingParams(
                max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1,
                min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
                repetition_penalty=1.0, logprobs=-1, detokenize=False,
                seed=self.generation["seed"],
            )
        except Exception as exc:
            raise BackendError(
                "vllm initialization failed; selected precision/architecture/adapter and full-vocabulary "
                "raw_logprobs support are required. No approximate fallback is used."
            ) from exc

    def engine_stats(self):
        counters={}
        accepted={'prompt_tokens','generation_tokens','prefix_cache_queries','prefix_cache_hits','num_preemptions','request_success'}
        for metric in self.engine.get_metrics():
            name=metric.name.removeprefix('vllm:').removesuffix('_total')
            if name not in accepted: continue
            value=getattr(metric,'value',None)
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                counters[name]=counters.get(name,0)+value
        return {'source':'vllm_metrics','prefix_caching':self.prefix_caching,'counters':counters}

    def _completion(self, ids: Iterable[int], sampling_params: Any, meter=None, kind="probes"):
        check_deadline(meter)
        values = list(ids)
        if not values:
            raise ValueError("prompt token IDs cannot be empty")
        kwargs = {"use_tqdm": False}
        if self.lora_request is not None:
            kwargs["lora_request"] = self.lora_request
        try:
            if meter is not None: meter['engine_submitted']+=len(values)
            # TokensPrompt is a TypedDict; its runtime representation is this dict.
            # TODO(vllm 0.31): installed-signature test is skipped without vllm.
            with self.lock:
                check_deadline(meter)
                result = self.engine.generate(
                    [{"prompt_token_ids": values}], sampling_params, **kwargs,
                )
        except TimeoutError:
            raise
        except Exception as exc:
            if meter is not None: meter['engine_usage_unknown']=True
            raise BackendError("vllm request failed; verify its architecture, adapter and logprobs support") from exc
        if len(result) != 1 or len(result[0].outputs) != 1:
            raise BackendError("vllm did not return exactly one completion")
        if meter is not None:
            n=len(result[0].outputs[0].token_ids)
            meter['output']+=n;meter[kind]+=n
            cached=getattr(result[0],'num_cached_tokens',None)
            if cached is None: meter['cached_unknown']=True
            else: meter['cached']+=int(cached)
            meter['engine_calls']=meter.get('engine_calls',0)+1
            meter['engine_generated']=meter.get('engine_generated',0)+n
            meter['engine_input']=meter.get('engine_input',0)+len(values)-int(cached or 0)
        check_deadline(meter)
        return result[0].outputs[0]

    def next_logits(self, ids: Iterable[int], candidate_ids: Iterable[int], meter=None) -> dict[int, float]:
        candidates = _candidate_ids(candidate_ids)
        values = list(ids)
        scores = {}
        # v0.31 supports explicit gather from RAW logprobs (before masks).
        # Chunking preserves the same full-vocabulary normalizer for all IDs.
        for start in range(0, len(candidates), 128):
            wanted = candidates[start:start + 128]
            kwargs = dict(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1,
                          repetition_penalty=1.0, detokenize=False,
                          seed=self.generation['seed'])
            try:
                params = self.SamplingParams(**kwargs, allowed_token_ids=wanted,
                    logprob_token_ids=wanted, logprobs=len(wanted))
            except TypeError:
                params = self.SamplingParams(**kwargs, logprobs=min(50, max(len(wanted), 1)))
            output = self._completion(values, params, meter)
            distribution = output.logprobs[0] if output.logprobs else None
            if len(output.token_ids) != 1:
                raise BackendError('Probe must produce exactly one token')
            if distribution is None or any(t not in distribution for t in wanted):
                logging.warning('Candidate readout incomplete; one full-vocabulary recovery probe')
                if meter is not None:
                    meter.setdefault('warnings', []).append('candidate_full_vocab_recovery')
                output = self._completion(values, self.probe_params, meter)
                distribution = output.logprobs[0] if output.logprobs else None
                if len(output.token_ids) != 1 or distribution is None or any(t not in distribution for t in candidates):
                    raise BackendError('vllm omitted candidate token logprobs after recovery')
                return _check_scores({t: float(distribution[t].logprob) for t in candidates})
            scores.update({t: float(distribution[t].logprob) for t in wanted})
        return _check_scores(scores)

    def open_session(self, ids, meter=None):
        return VLLMPrefixSession(self,ids,meter if meter is not None else new_meter())

    def slow_session(self, ids, unparsed_policy, meter, max_new_tokens=None):
        count=_max_new_tokens(max_new_tokens or self.generation['max_new_tokens'])
        g=self.generation; sampling=g['do_sample']
        params=self.SamplingParams(max_tokens=count,temperature=g['temperature'] if sampling else 0.0,
            top_p=g['top_p'] if sampling else 1.0,top_k=(g['top_k'] or -1) if sampling else -1,
            repetition_penalty=g['repetition_penalty'],seed=g['seed'],
            stop=[self.marker],include_stop_str_in_output=True)
        meter['prompt']+=len(ids)
        out=self._completion(ids,params,meter,'draft')
        gen=list(out.token_ids)
        parsed=(getattr(out,'stop_reason',None)==self.marker and
                getattr(out,'finish_reason',None)=='stop' and
                bool(re.search(r'(?m)^[ \t]*'+re.escape(self.marker)+r'$',self.decode(gen))))
        if not parsed:
            if unparsed_policy=='fast': return None,False
            forced=self.encode('\n'+self.marker);gen+=forced
            meter['output']+=len(forced);meter['forced']+=len(forced)
        return VLLMPrefixSession(self,list(ids)+gen,meter),parsed

    def generate(self, ids: Iterable[int], max_new_tokens: int) -> list[int]:
        count = _max_new_tokens(max_new_tokens)
        generation = self.generation
        sampling = generation["do_sample"]
        # HF uses top_k=0 for unrestricted sampling; vLLM uses top_k=-1.
        top_k = generation["top_k"] if generation["top_k"] > 0 else -1
        params = self.SamplingParams(
            max_tokens=count, temperature=generation["temperature"] if sampling else 0.0,
            top_p=generation["top_p"] if sampling else 1.0,
            top_k=top_k if sampling else -1,
            repetition_penalty=generation["repetition_penalty"],
            seed=generation["seed"], detokenize=False,
        )
        output = self._completion(ids, params)
        if len(output.token_ids) > count:
            raise BackendError("vllm generated more tokens than max_new_tokens")
        return list(output.token_ids)


def create_backend(config: dict[str, Any]):
    name = config["model"]["backend"]
    if name == "transformers":
        return TransformersBackend(config)
    if name == "vllm":
        return VLLMBackend(config)
    raise ValueError("model.backend must be transformers or vllm")

class CachedSession:
    """Single prefill followed exclusively by KV-backed token continuations.

    Branches own their cache, including recurrent attention state. Models that
    cannot copy/cache fail into the service's fast path rather than re-prefill.
    """
    def __init__(self, backend, ids):
        self.backend = backend
        self.cache = None
        self.length = 0
        self.advance(list(ids))

    def advance(self, ids):
        if not ids:
            return
        backend = self.backend
        meter=getattr(self,'meter',None)
        if meter is not None:
            check_deadline(meter)
            meter['engine_input']+=len(ids);meter['engine_submitted']+=len(ids)
        kwargs = dict(input_ids=backend._input(ids), use_cache=True, logits_to_keep=1)
        if self.cache is not None:
            kwargs['past_key_values'] = self.cache
        try:
            with backend.torch.inference_mode():
                output = backend.model(**kwargs)
        except Exception:
            if meter is not None:meter['engine_usage_unknown']=True
            raise
        self.cache = output.past_key_values
        if self.cache is None:
            raise BackendError('Model did not return a reusable cache')
        self.length += len(ids)
        self._logits = output.logits[0, -1, :].float()
        self._log_probs = None

    def top_token(self):
        return int(self._logits.argmax().item())

    def log_probs(self, ids):
        if self._log_probs is None:
            self._log_probs=self._logits.log_softmax(dim=-1)
        index=self.backend.torch.tensor(list(ids),dtype=self.backend.torch.long,device=self._logits.device)
        return _check_scores(dict(zip(ids,self._log_probs.index_select(0,index).cpu().tolist())))

    def non_digit_logprob(self, digit_ids):
        if self._log_probs is None: self._log_probs=self._logits.log_softmax(dim=-1)
        ids=[i for i in digit_ids if i<self._logits.shape[0]]
        if not ids: return 0.0
        torch=self.backend.torch
        index=torch.tensor(ids,dtype=torch.long,device=self._logits.device)
        # Sum excluded mass on device; return one scalar, never vocab logits.
        mass=self._log_probs.index_select(0,index).exp().sum().clamp(max=1.0)
        return float(torch.log1p(-mass).item())

    def fork(self):
        import copy
        branch = object.__new__(type(self))
        branch.backend, branch.length = self.backend, self.length
        branch.meter=getattr(self,'meter',None)
        branch.cache = copy.deepcopy(self.cache)
        branch._logits = self._logits
        branch._log_probs = self._log_probs
        return branch


def check_deadline(meter):
    if meter and (meter.get('cancel_event') is not None and meter['cancel_event'].is_set() or
                  meter.get('deadline') is not None and time.monotonic()>=meter['deadline']):
        raise TimeoutError('Slow deadline exceeded')


def new_meter():
    return dict(prompt=0,output=0,cached=0,engine_submitted=0,draft=0,forced=0,probes=0,
                engine_calls=0,engine_generated=0,engine_input=0,generate_nonempty_calls=0,hf_prefill_tokens=0)


class VLLMPrefixSession:
    """Immutable prefix branches; APC owns KV storage and recomputation."""
    handles_meter=True
    def __init__(self,backend,ids,meter):
        self.backend,self.ids,self.meter=backend,list(ids),meter
        self._logits=None

    @property
    def logits(self):
        if self._logits is None:
            out=self.backend._completion(self.ids,self.backend.probe_params,self.meter)
            if not out.logprobs or out.logprobs[0] is None:
                raise BackendError('Missing full-vocabulary probe distribution')
            self._logits=_check_scores({t:float(lp.logprob) for t,lp in out.logprobs[0].items()})
        return self._logits

    def log_probs(self,ids):
        return self.backend.next_logits(self.ids, ids, self.meter)

    def non_digit_logprob(self, digit_ids):
        if not digit_ids: return 0.0
        scores = self.log_probs(sorted(digit_ids))
        mass = min(1.0, math.fsum(math.exp(v) for v in scores.values()))
        return math.log1p(-mass) if mass < 1 else -math.inf

    def fork(self):
        branch=VLLMPrefixSession(self.backend,self.ids,self.meter)
        branch._logits=self._logits
        return branch

    def advance(self,ids):
        # These are candidate continuation tokens, separate from engine probes.
        self.ids+=list(ids);self._logits=None
        self.meter['output']+=len(ids);self.meter['probes']+=len(ids)
