"""Rendering, candidate aggregation and service logic, without model imports."""
from __future__ import annotations

import json
import math
import re
import string
from pathlib import Path

import threading
if __package__:
    from .backends import new_meter, check_deadline
    from .quota import QuotaStateError
else:
    from backends import new_meter, check_deadline
    from quota import QuotaStateError

ROOT = Path(__file__).resolve().parent
YES_FORMS = ["yes", "Yes", "YES", " yes", " Yes", " YES", "true", "True", " true", " True", "Y", " Y"]
NO_FORMS = ["no", "No", "NO", " no", " No", " NO", "false", "False", " false", " False", "N", " N"]
ALIASES = {"boolean": "noul", "enum": "choice", "noul": "noul", "choice": "choice", "score": "score"}


def external_output(path, *, create_parent=True):
    """Validate an explicit external path; read-only callers disable mkdir."""
    result = Path(path)
    if not result.is_absolute():
        raise ValueError("External paths must be absolute")
    result = result.resolve()
    if result.is_relative_to(ROOT) or ROOT.is_relative_to(result):
        raise ValueError("External paths must be separate from the package")
    if create_parent:
        result.parent.mkdir(parents=True, exist_ok=True)
    return result


def load_json(path):
    def reject(value):
        raise ValueError("Non-finite JSON value")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def load_config(path=None):
    config = load_json(path or ROOT / "config.json")
    apply_profile(config)
    validate_config(config)
    return config


def apply_profile(config):
    import copy
    profile=config.get('profiles',{}).get(config['model']['model_key'])
    if profile is None: raise ValueError('No profile for model_key')
    for section,values in profile.items():
        config.setdefault(section,{}).update(copy.deepcopy(values))
    return apply_budget_tier(config)


def apply_budget_tier(config, tier=None):
    r=config['routing'];tier=tier or r.get('tier','normal')
    if tier not in ('normal','hard','long'):raise ValueError('Unknown quota tier')
    r['tier']=tier
    tiers=r.get('tiers',{})
    values=tiers.get(tier,{})
    for key in ('reasoning_words','max_new_tokens'):
        if key in tiers.get('hard',{}):config['generation'][key]=tiers['hard'][key]
    if 'max_prompt_tokens' in values:config['limits']['max_prompt_tokens']=values['max_prompt_tokens']
    for key in ('reasoning_words','max_new_tokens'):
        if key in values:config['generation'][key]=values[key]
    # Forced long is a runtime admission switch, preserving fast identity.
    return config


def finite_number(value, name, *, low=None, high=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (low is not None and value < low) or (high is not None and value > high) or (positive and value <= 0):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def validate_config(c):
    if __package__: from .quota import cost_settings
    else: from quota import cost_settings
    cost_settings(c)
    if c['model']['model_key'] not in c.get('profiles',{}):raise ValueError('No profile for model_key')
    if c.get('usage',{}).get('basis','engine') not in ('engine','each_once','submitted'):
        raise ValueError('Invalid usage basis')
    m, r, g, p, s = (c[k] for k in ("model", "routing", "generation", "prompt", "server"))
    if m["backend"] not in ("transformers", "vllm") or m["quant"] not in ("int8", "bf16", "awq"):
        raise ValueError("Unsupported backend or precision")
    if m['backend']=='vllm':
        neutral={'suppress_tokens':None,'begin_suppress_tokens':None,'bad_words_ids':None,
            'forced_bos_token_id':None,'forced_eos_token_id':None,'forced_decoder_ids':None,
            'no_repeat_ngram_size':0,'min_new_tokens':0,'min_length':0,'length_penalty':1.0,'renormalize_logits':False}
        if any(g.get(k,v)!=v for k,v in neutral.items()):raise ValueError('Unsupported vLLM decode override')
    if r["mode"] not in ("routed", "fast_only", "uniform"):
        raise ValueError("Unsupported routing mode")
    tiers=r.get('tiers',{})
    if tiers:
        if set(tiers)!={'normal','hard','long'} or r.get('tier','normal') not in tiers:raise ValueError('Require normal/hard/long tiers')
        if tiers['long'].get('fast_only') is not True:raise ValueError('Long tier must be fast only')
        for tier in ('normal','long'):
            value=tiers[tier].get('max_prompt_tokens')
            if type(value) is not int or not 0<value<=c['limits']['physical_max_prompt_tokens']:raise ValueError('Invalid tier input budget')
        for key in ('reasoning_words','max_new_tokens'):
            if type(tiers['hard'].get(key)) is not int or tiers['hard'][key]<=0:raise ValueError('Invalid hard draft budget')
    finite_number(r["margin_threshold"], "margin_threshold", low=0)
    if r["margin_kind"]=="probdiff" and r["margin_threshold"]>1:
        raise ValueError("probdiff threshold must be <=1")
    if r["margin_kind"] not in ("logratio", "probdiff"):
        raise ValueError("Invalid margin_kind")
    if r.get('long_item_slow_policy','fast_only') not in ('fast_only','allow'):
        raise ValueError('Invalid long item policy')
    if r["unparsed_policy"] not in ("fast", "force_answer"):
        raise ValueError("Invalid unparsed_policy")
    finite_number(r["quota"], "quota", low=0, high=1)
    finite_number(r["fuse"], "fuse", low=0, high=1)
    if r["quota"] > r["fuse"]:
        raise ValueError("quota must not exceed fuse")
    if r["mode"] == "uniform" and (r["quota"] != 1 or r["fuse"] != 1):
        raise ValueError("uniform requires explicit quota=1 and fuse=1; it is an unbounded draft experiment")
    for key in ("reasoning_words", "max_new_tokens"):
        if type(g[key]) is not int or g[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(g["seed"]) is not int or type(g["do_sample"]) is not bool:
        raise ValueError("Invalid generation seed or do_sample")
    finite_number(g.get("wall_timeout_seconds",60),"slow wall timeout",positive=True)
    finite_number(g["temperature"], "generation temperature", positive=True)
    finite_number(g["top_p"], "top_p", positive=True, high=1)
    finite_number(g["repetition_penalty"], "repetition_penalty", positive=True)
    if type(g["top_k"]) is not int or g["top_k"] < 0:
        raise ValueError("top_k must be a nonnegative integer")
    for key in ("max_request_bytes", "max_questions", "port"):
        if type(s[key]) is not int or s[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if s["port"] > 65535:
        raise ValueError("Invalid port")
    finite_number(s["socket_timeout_seconds"], "socket timeout", positive=True)
    if not isinstance(p["answer_marker"], str) or not p["answer_marker"].strip() or "\n" in p["answer_marker"]:
        raise ValueError("answer_marker must be a nonempty single line")
    if type(p["enable_thinking"]) is not bool or not isinstance(p["system"], str):
        raise ValueError("Invalid prompt settings")
    for template in p["cot_instructions"].values():
        template.format(reasoning_words=g["reasoning_words"])
    for value in (c["limits"]["max_prompt_tokens"], r["slow_max_prompt_tokens"]):
        if type(value) is not int or value <= 0:
            raise ValueError("Token limits must be positive integers")
    limits=c['limits']
    for name in ('physical_max_prompt_tokens','oom_retry_prompt_tokens','state_chunk_tokens','precompression_trigger_chars_per_token','estimated_chars_per_token'):
        if type(limits[name]) is not int or limits[name]<=0: raise ValueError('Invalid compression limit: '+name)
    for name in ('state_head_tokens','state_tail_tokens','compression_reserve_tokens'):
        if type(limits[name]) is not int or limits[name]<0: raise ValueError('Invalid compression limit: '+name)
    if limits['physical_max_prompt_tokens']<limits['max_prompt_tokens']:
        raise ValueError('Physical maximum must cover soft prompt budget')
    finite_number(c["limits"]["truncation_head_frac"], "head fraction", low=0, high=1)
    if c.get("interface", {}).get("boolean_keys", "request") not in ("request", "true_false", "yes_no"):
        raise ValueError("Unknown boolean key style")


class UnsupportedRequest(ValueError):
    """Structurally valid request outside the supported semantic domain (422)."""


class ItemFailure(RuntimeError):
    """An individual item could not be processed (HTTP 422)."""


class ServiceFailure(RuntimeError):
    """An unrecoverable service failure (HTTP 503)."""


ITEM_FAILURE_LIMIT = 3


def unrecoverable_engine_error(exc):
    """Recognize wrapped vLLM engine failures without importing the runtime."""
    pending, seen = [exc], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        classes = type(current).__mro__
        if isinstance(current, ServiceFailure) or any(cls.__name__ == 'EngineDeadError' for cls in classes):
            return True
        vllm_classes = {cls.__name__ for cls in classes
                        if cls.__module__ == 'vllm' or cls.__module__.startswith('vllm.')}
        if 'VLLMServerError' in vllm_classes and 'EngineGenerateError' not in vllm_classes:
            return True
        pending.extend((current.__cause__, current.__context__))
    return False


def choice_codes(count):
    if count<=26:return list(string.ascii_uppercase[:count])
    if count>676:raise UnsupportedRequest('At most 676 choices supported')
    return [a+b for a in string.ascii_uppercase for b in string.ascii_uppercase][:count]


def normalize_question(question):
    if not isinstance(question, dict):
        raise ValueError("Each question must be an object")
    qtype = ALIASES.get(question.get("type"))
    if qtype is None:
        raise UnsupportedRequest("Unsupported question type")
    instructions = question.get("instructions", question.get("description"))
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Question needs instructions or description")
    criteria = question.get("criteria")
    q = {"request_type": question["type"], "type": qtype, "instructions": instructions, "criteria": criteria}
    labels = None
    if qtype == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an object")
    elif qtype == "score":
        if not isinstance(criteria,list): raise ValueError("score criteria must be an ordered list")
        if not criteria: raise UnsupportedRequest("score criteria is empty")
    else:
        labels = question.get("labels")
        if labels is None:
            labels = list(criteria) if isinstance(criteria, (dict, list)) else None
        if not isinstance(labels,list): raise ValueError("choice needs labels or ordered criteria")
        if not 1<=len(labels)<=676: raise UnsupportedRequest("choice needs 1..676 labels")
        if any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != len(labels):
            raise ValueError("choice labels must be distinct nonempty strings")
        if criteria is not None and not isinstance(criteria, (dict, list)):
            raise ValueError("choice criteria must be an object or list")
    return q, labels


def normalize_request(payload, max_questions):
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError("Request needs state and questions")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= max_questions:
        raise ValueError("questions must be a nonempty object within max_questions")
    tasks = []
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Question names must be nonempty strings")
        q, labels = normalize_question(question)
        tasks.append((name, {"state": payload["state"], "question": q, "labels": labels}))
    return tasks


def letter_forms(c):
    return [c, " " + c, c.lower(), " " + c.lower(), c + ".", "(" + c]


def digit_forms(d):
    return [d, " " + d, d + "."]


def render(task):
    # Source-faithful copy of cache_hidden_v2.render; do not deduplicate forms.
    q = task["question"]; qtype = q["type"]; crit = q.get("criteria")
    state = task["state"] if isinstance(task["state"], str) else json.dumps(task["state"], ensure_ascii=False)
    cq = "Context:\n" + state + "\n\nQuestion: " + q["instructions"]
    if qtype == "noul":
        labels = ["no", "yes"]; c = crit or {}; groups = {"yes": YES_FORMS, "no": NO_FORMS}
        parts = [cq] + (["Answer yes if: " + str(c.get("true", "Yes")) + "\nAnswer no if: " + str(c.get("false", "No"))] if (c.get("true") or c.get("false")) else []) + ["Answer with one word: yes or no."]
    elif qtype == "score":
        labels = [str(i) for i in range(len(crit))]; groups = {l: digit_forms(l) for l in labels}
        parts = [cq, "Levels:\n" + "\n".join(str(i) + ": " + str(t) for i, t in enumerate(crit)), "Answer with one number from 0 to " + str(len(crit) - 1) + "."]
    else:
        labels = list(task["labels"]); rubric = crit if isinstance(crit, dict) else {}; codes = choice_codes(len(labels))
        groups = {l: letter_forms(c) for c, l in zip(codes, labels)}
        parts = [cq, "Options:\n" + "\n".join(c + ". " + str(l) + ((": " + str(rubric[l])) if rubric.get(l) else "") for c, l in zip(codes, labels)), "Answer with the letter of the best option."]
    return labels, "\n\n".join(parts), groups


def parse_answer(generation, qtype="noul", labels=None):
    """Audit only: preserve cache_cot's explicit-last and standalone fallback."""
    if qtype == "noul":
        matches = re.findall(r"ANSWER:\s*(yes|no)", generation, re.I) or re.findall(r"\b(yes|no)\b", generation, re.I)
        return matches[-1].lower() if matches else None
    if qtype == "choice":
        mapping = dict(zip(choice_codes(len(labels or [])), labels or []))
        explicit, fallback = r"ANSWER:\s*([A-Z]{1,2})\b", r"\b([A-Z]{1,2})\b"
    elif qtype == "score":
        mapping = {str(label): str(label) for label in (labels or [])}
        explicit, fallback = r"ANSWER:\s*(\d+)", r"(?<![\w.+-])(\d+)(?!\w|\.\d)"
    else:
        raise ValueError("Unsupported question type")
    matches = re.findall(explicit, generation)
    if matches and matches[-1] in mapping:
        return mapping[matches[-1]]
    matches = [symbol for symbol in re.findall(fallback, generation) if symbol in mapping]
    return mapping[matches[-1]] if matches else None


def readout_prefix(generation, marker):
    matches = list(re.finditer(r"(?m)^[ \t]*" + re.escape(marker), generation))
    if matches:
        return generation[:matches[-1].end()], False
    return generation.rstrip() + "\n" + marker, True


def logsumexp(values):
    top = max(values)
    if top == -math.inf: return top
    return top + math.log(sum(math.exp(x - top) for x in values))


def candidate_ids(backend, groups):
    # Keep multiplicity even when two textual forms map to the same token id.
    result = {}
    for label, forms in groups.items():
        tokens = [backend.encode(form) for form in forms]
        result[label] = [ids[0] for ids in tokens if len(ids) == 1]
    if not any(result.values()):
        raise ValueError("Tokenizer has no single-token candidate forms")
    return result


def aggregate(logits, groups):
    for token in {t for ids in groups.values() for t in ids}:
        if token not in logits or not math.isfinite(logits[token]):
            raise RuntimeError("Backend omitted a candidate or returned a non-finite logit")
    return {label: logsumexp([logits[t] for t in ids]) if ids else -1e9 for label, ids in groups.items()}


def validate_calibration(calibration, model_key, path="fast"):
    model = calibration.get("models", {}).get(model_key)
    if model is None:
        raise ValueError("Calibration lacks selected model_key")
    block = model.get(path)
    if block is None:
        raise ValueError("Calibration lacks selected route")
    for kind in ("choice", "score"):
        finite_number(block["temperature_by_kind"][kind], kind + " temperature", positive=True)
    if block["temperature_by_kind"]["noul"] != "one_bin":
        raise ValueError("noul calibration must be one_bin")
    finite_number(block["noul_calibration"]["p_cal"], "p_cal", low=0, high=1)
    return block


def probabilities(labels, logits, qtype, block):
    if qtype == "noul":
        # Deterministic tie: earliest in render's label order, namely no.
        winner = max(labels, key=logits.__getitem__)
        p_cal = min(1 - 1e-6, max(0.5, block["noul_calibration"]["p_cal"]))
        return {label: p_cal if label == winner else 1 - p_cal for label in labels}
    temperature = block["temperature_by_kind"][qtype]
    # Subtract before dividing to avoid overflow from small positive T.
    top = max(logits.values())
    weights = {label: math.exp((logits[label] - top) / temperature) for label in labels}
    total = sum(weights.values())
    floored = {label: max(1e-6, weights[label] / total) for label in labels}
    return {label: value / sum(floored.values()) for label, value in floored.items()}


# Rule sections are protected conservatively. Unknown text outside a recognized
# section is narrative; explicit rule/policy/definition/criteria subtrees are sacred.
_RULE = re.compile(r'(?im)^\s*(?:R\d+\s*[:：]|(?:rules?|polic(?:y|ies)|definitions?|criteria|规则|規則|定义|定義|政策|判定标准)\s*[:：])')

_RULE_KEY = re.compile(r'^\s*(?:rules?|polic(?:y|ies)|definitions?|criteria|application_rule_set|规则|規則|定义|定義|政策|标准|判定标准)\s*$', re.I)

def contains_rule_subtree(value):
    if isinstance(value,str):return bool(_RULE.search(value))
    if isinstance(value,dict):return any(_RULE_KEY.fullmatch(str(k)) or contains_rule_subtree(v) for k,v in value.items())
    if isinstance(value,list):return any(contains_rule_subtree(v) for v in value)
    return False


OMISSION = '\n[... state middle omitted ...]\n'


def lexical_terms(text):
    words = re.findall(r'[a-zA-Z_][\w-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]', text.lower())
    # Character bigrams preserve Chinese query signal without a segmenter.
    words += re.findall(r'(?=([\u4e00-\u9fff]{2}))', text)
    return set(words)


def shrink_state(state, keep_chars, head_fraction=0.6, *, backend=None, query='',
                 head_tokens=512, tail_tokens=512, chunk_tokens=200):
    """Extract query-relevant chunks in original order, protecting explicit rules.

    The budget is tokens when a backend is supplied, otherwise characters.
    A contiguous omitted range produces exactly one marker.
    """
    encode = backend.encode if backend else lambda x: list(x)
    decode = backend.decode if backend else lambda x: ''.join(x)
    query_terms = lexical_terms(query)
    important = {w for w in query_terms if any(c.isdigit() for c in w)}
    important |= {w.lower() for w in re.findall(r'\b(?:[A-Z][a-z]+|[A-Z][A-Z0-9_]+)\b',query)}
    def text_shrink(text, budget):
        if len(encode(text)) <= budget:
            return text
        # Slice original paragraphs including their delimiters; never make a
        # delimiter into an independent compressible paragraph.
        spans=[]; rule_section=False
        for match in re.finditer(r'.+?(?:\n[ \t]*\n|\Z)',text,re.S):
            part=match.group()
            heading=re.match(r'^\s*(?:#{1,6}\s*)?([^\n:：]{1,60})[:：]\s*(?:\n|$)',part)
            if heading:
                title=heading.group(1)
                if _RULE_KEY.search(title): rule_section=True
                elif re.fullmatch(r'(?i)narrative|events?|history|state|context|data|observations?|背景|叙述|事件|状态|记录',title.strip()): rule_section=False
            protected=rule_section or bool(_RULE.search(part))
            if spans and spans[-1][1]==protected: spans[-1]=(spans[-1][0]+part,protected)
            else: spans.append((part,protected))
        merged=spans
        encoded=[(part,keep,encode(part)) for part,keep in merged]
        total=sum(len(ids) for _,_,ids in encoded)
        head_end=min(head_tokens,total);tail_start=max(head_end,total-tail_tokens)
        blocks=[];cursor=0
        for part,protected,ids in encoded:
            if protected:
                blocks.append((part,len(ids),True));cursor+=len(ids)
                continue
            # Sentence/newline boundaries first, word boundaries for oversized
            # sentences. Never decode an arbitrary token slice into a block.
            sentences=re.findall(r'.*?(?:[.!?。！？](?=\s|$)|\n|$)',part,re.S)
            pieces=[]
            for sentence in sentences:
                if not sentence:continue
                if len(encode(sentence))<=chunk_tokens:pieces.append(sentence)
                else:
                    words=re.findall(r'\s+|\S+\s*',sentence)
                    chunk=''
                    for word in words:
                        if chunk and len(encode(chunk+word))>chunk_tokens:
                            pieces.append(chunk);chunk=''
                        chunk+=word
                    if chunk:pieces.append(chunk)
            chunk=''
            def add(value):
                nonlocal cursor
                size=len(encode(value))
                edge=cursor<head_end or cursor+size>tail_start
                blocks.append((value,size,edge));cursor+=size
            for piece in pieces:
                if chunk and len(encode(chunk+piece))>chunk_tokens:
                    add(chunk);chunk=''
                chunk+=piece
            if chunk:add(chunk)
        selected={i for i,(_,_,keep) in enumerate(blocks) if keep}
        marker_cost=len(encode(OMISSION))
        used=sum(blocks[i][1] for i in selected)+marker_cost*(len(selected)+1)
        ranked=[]
        for i,(part,size,keep) in enumerate(blocks):
            terms=lexical_terms(part); overlap=query_terms & terms
            score=(len(overlap)+2*len(important & terms))/max(1,math.sqrt(len(terms)))
            if not keep: ranked.append((-score,i))
        for _,i in sorted(ranked):
            if used+blocks[i][1]+marker_cost <= budget:
                selected.add(i); used+=blocks[i][1]+marker_cost
        def join(indices):
            out=[]; omitted=False
            for i,(part,_,_) in enumerate(blocks):
                if i in indices:
                    if omitted: out.append(OMISSION)
                    out.append(part); omitted=False
                else: omitted=True
            if omitted: out.append(OMISSION)
            return ''.join(out)
        result=join(selected)
        # Recover conservative omission-marker reserves using adjacent blocks.
        while len(encode(result))<.8*budget:
            neighbors=sorted({j for i in selected for j in (i-1,i+1) if 0<=j<len(blocks) and j not in selected})
            added=False
            for j in neighbors:
                candidate=join(selected|{j})
                if len(encode(candidate))<=budget:
                    selected.add(j);result=candidate;added=True
            if not added:break
        if len(encode(result)) <= budget: return result
        # Protected material survives the soft budget. For an unusually small
        # budget shrink only narrative edges globally, retaining the rules.
        if any(keep for _,keep in merged):
            protected_text=''.join(part for part,keep in merged if keep)
            allowance=max(0,budget-len(encode(protected_text))-marker_cost*(len(merged)+1))
            if not allowance: return protected_text
            n=sum(len(encode(part)) for part,keep in merged if not keep)
            h=int(allowance*head_fraction);t=allowance-h
            out=[];offset=0;omitted=False
            for part,keep in merged:
                if keep:
                    if omitted:out.append(OMISSION);omitted=False
                    out.append(part);continue
                ids=encode(part)
                head=ids[:max(0,min(len(ids),h-offset))]
                tail=ids[max(0,n-t-offset):] if offset+len(ids)>n-t else []
                if head:
                    if omitted:out.append(OMISSION);omitted=False
                    out.append(decode(head))
                if len(head)+len(tail)<len(ids):omitted=True
                if tail:
                    if omitted:out.append(OMISSION);omitted=False
                    out.append(decode(tail))
                offset+=len(ids)
            if omitted:out.append(OMISSION)
            return ''.join(out)
        ids=encode(text); available=max(0,budget-marker_cost)
        h=int(available*head_fraction); t=available-h
        return decode(ids[:h])+OMISSION+(decode(ids[-t:]) if t else '')

    if isinstance(state,str): return text_shrink(state,max(0,keep_chars))
    if isinstance(state,dict):
        protected={k:v for k,v in state.items() if _RULE_KEY.search(str(k)) or
                   (not isinstance(v,(dict,list)) and len(str(v))<=200)}
        rest=[k for k in state if k not in protected]
        remaining=max(0,keep_chars-len(encode(json.dumps(protected,ensure_ascii=False))))
        sizes={k:len(encode(json.dumps(state[k],ensure_ascii=False))) for k in rest}
        total=max(1,sum(sizes.values()))
        return {k:v if k in protected else shrink_state(v,int(remaining*sizes[k]/total),head_fraction,
                backend=backend,query=query,head_tokens=head_tokens,tail_tokens=tail_tokens,chunk_tokens=chunk_tokens)
                for k,v in state.items()}
    if isinstance(state,list):
        if len(encode(json.dumps(state,ensure_ascii=False)))<=keep_chars: return state
        if not state: return state
        # Budget whole entries, protect explicit rule entries; preserve both ends.
        selected={0,len(state)-1}|{i for i,v in enumerate(state) if contains_rule_subtree(v)}
        ranked=sorted(range(1,len(state)-1),key=lambda i:(-len(lexical_terms(str(state[i])) & query_terms),i))
        used=sum(len(encode(json.dumps(state[i],ensure_ascii=False))) for i in selected)
        for i in ranked:
            size=len(encode(json.dumps(state[i],ensure_ascii=False)))
            if i not in selected and used+size+len(OMISSION)<keep_chars: selected.add(i); used+=size
        protected_entries={i for i in selected if isinstance(state[i],str) and _RULE.search(state[i])}
        protected_size=sum(len(encode(json.dumps(state[i],ensure_ascii=False))) for i in protected_entries)
        narrative_size=max(1,sum(len(encode(json.dumps(state[i],ensure_ascii=False))) for i in selected-protected_entries))
        free=max(0,keep_chars-protected_size-len(selected)*len(OMISSION))
        result=[]; omitted=False
        for i,v in enumerate(state):
            if i in selected:
                if omitted: result.append(OMISSION.strip())
                if i not in protected_entries and len(str(v))>200:
                    size=len(encode(json.dumps(v,ensure_ascii=False)))
                    v=shrink_state(v,int(free*size/narrative_size),head_fraction,backend=backend,
                        query=query,head_tokens=head_tokens,tail_tokens=tail_tokens,chunk_tokens=chunk_tokens)
                result.append(v);omitted=False
            else: omitted=True
        return result
    return state


def prepare_task(task, backend, config):
    def prompts(value):
        labels,text,groups=render(value)
        instruction=config['prompt']['cot_instructions'][value['question']['type']].format(
            reasoning_words=config['generation']['reasoning_words']).replace('ANSWER:',config['prompt']['answer_marker'])
        system,think=config['prompt']['system'],config['prompt']['enable_thinking']
        return labels,groups,backend.prompt_ids(system,text,think),backend.prompt_ids(system,text+'\n'+instruction,think)
    limits=config['limits']; budget=limits['max_prompt_tokens'];normal_budget=budget
    long_budget=min(budget,config['routing'].get('tiers',{}).get('long',{}).get('max_prompt_tokens',budget))
    query=task['question']['instructions']+' '+json.dumps(task['question'].get('criteria'),ensure_ascii=False)+' '+str(task.get('labels') or '')
    raw=task['state'] if isinstance(task['state'],str) else json.dumps(task['state'],ensure_ascii=False)
    precompressed=len(raw)>limits['precompression_trigger_chars_per_token']*budget
    source=task
    if precompressed:budget=long_budget
    if precompressed:
        ratio=limits['estimated_chars_per_token']
        source=dict(task,state=shrink_state(task['state'],budget*ratio,limits['truncation_head_frac'],query=query,
            head_tokens=limits['state_head_tokens']*ratio,tail_tokens=limits['state_tail_tokens']*ratio,
            chunk_tokens=limits['state_chunk_tokens']*ratio))
    original=prompts(source); original_count=len(original[2])
    long_input=precompressed or original_count>normal_budget
    if long_input:budget=long_budget
    current,prepared=source,original
    refilled=False
    if precompressed and original_count<.8*budget:
        fixed=len(prompts(dict(task,state=''))[2])
        refill=shrink_state(task['state'],max(0,budget-fixed-limits['compression_reserve_tokens']),
            limits['truncation_head_frac'],backend=backend,query=query,
            head_tokens=limits['state_head_tokens'],tail_tokens=limits['state_tail_tokens'],chunk_tokens=limits['state_chunk_tokens'])
        candidate=dict(task,state=refill);candidate_prompts=prompts(candidate)
        if len(candidate_prompts[2])<=budget:
            current,prepared=candidate,candidate_prompts;refilled=True
    if original_count>budget:
        fixed=len(prompts(dict(task,state=''))[2])
        query=task['question']['instructions']+' '+json.dumps(task['question'].get('criteria'),ensure_ascii=False)+' '+str(task.get('labels') or '')
        allowance=max(0,budget-fixed-limits['compression_reserve_tokens'])
        for _ in range(3):
            current=dict(task,state=shrink_state(source['state'],allowance,limits['truncation_head_frac'],
                backend=backend,query=query,head_tokens=limits['state_head_tokens'],
                tail_tokens=limits['state_tail_tokens'],chunk_tokens=limits['state_chunk_tokens']))
            prepared=prompts(current); size=len(prepared[2])
            if size<=budget: break
            allowance=max(0,allowance-(size-budget)-limits['compression_reserve_tokens'])
        # Compression cannot remove sacred rule/question/option material. The
        # final physical guard always bounds tensors, with explicit metadata.
    kept=len(prepared[2]); physical=limits['physical_max_prompt_tokens']
    hard_cut=kept>physical
    if hard_cut:
        def cut(ids):
            h=int(physical*limits['truncation_head_frac']);t=physical-h
            return ids if len(ids)<=physical else ids[:h]+ids[-t:]
        prepared=prepared[:2]+(cut(prepared[2]),cut(prepared[3]))
    return current,prepared,{'original_prompt_tokens':original_count if not precompressed else None,
        'original_prompt_tokens_estimate':math.ceil(len(raw)/limits['estimated_chars_per_token']) if precompressed else original_count,
        'long_input':long_input,'character_precompressed':precompressed,'adjacent_refill':refilled,
        'kept_prompt_tokens':len(prepared[2]),
        'slow_prompt_exceeds_budget':len(prepared[3])>min(budget,config['routing']['slow_max_prompt_tokens']),
        'truncated':current['state']!=task['state'] or hard_cut,
        'protected_content_exceeds_budget':kept>budget,'physical_truncated':hard_cut}


def sequence_scores(session, backend, groups, meter=None):
    """Full-vocabulary chain probabilities; copies branch KV, never prompt prefill.

    Keep every form (including repeated tokenizations) with its original weight.
    """
    cache = {(): session}
    continuation_tokens = 0
    normalized = {}
    path_scores = {(): 0.0}
    def score(ids):
        nonlocal continuation_tokens
        prefix = ()
        for token in ids:
            check_deadline(meter)
            child = prefix + (token,)
            if child not in path_scores:
                parent = cache[prefix]
                if prefix not in normalized:
                    wanted=sorted({encoded[len(prefix)] for forms in groups.values() for form in forms
                        if (encoded:=backend.encode(form))[:len(prefix)]==list(prefix) and len(encoded)>len(prefix)})
                    if hasattr(parent,'log_probs'):
                        normalized[prefix]=parent.log_probs(wanted)
                    else:
                        logits=parent.logits; lse=logsumexp(list(logits.values()))
                        normalized[prefix]={t:logits[t]-lse for t in wanted if t in logits}
                value=normalized[prefix].get(token)
                if value is None or not math.isfinite(value): raise RuntimeError('Missing candidate token')
                path_scores[child]=path_scores[prefix]+value
            prefix = child
            if prefix not in cache and len(prefix) < len(ids):
                branch = cache[prefix[:-1]].fork()
                if meter is not None and not getattr(branch, 'handles_meter', False):
                    meter['output'] += 1
                    meter['probes'] += 1
                    if getattr(branch,'meter',None) is None:
                        meter['engine_input']=meter.get('engine_input',0)+1; meter['engine_submitted']+=1
                branch.advance([token])
                continuation_tokens += 1
                cache[prefix] = branch
        return path_scores[prefix]
    digit_ids=getattr(backend,'digit_start_ids',None)
    if digit_ids is None:
        # Synthetic tokenizer fallback only; real backends scan the vocabulary.
        digit_ids={ids[0] for d in string.digits if len(ids:=backend.encode(d))==1}
    terminals={}
    def termination(ids):
        nonlocal continuation_tokens
        key=tuple(ids)
        if key not in terminals:
            if key not in cache:
                branch=cache[key[:-1]].fork()
                if meter is not None and not getattr(branch,'handles_meter',False):
                    meter['output']+=1;meter['probes']+=1
                    if getattr(branch,'meter',None) is None:
                        meter['engine_input']=meter.get('engine_input',0)+1;meter['engine_submitted']+=1
                branch.advance([key[-1]]);continuation_tokens+=1;cache[key]=branch
            se=cache[key]
            if hasattr(se,'non_digit_logprob'):
                value=se.non_digit_logprob(digit_ids)
            else:
                logits=se.logits; z=logsumexp(list(logits.values()))
                allowed=[v for t,v in logits.items() if t not in digit_ids]
                value=logsumexp(allowed)-z if allowed else -math.inf
            terminals[key]=value
        return terminals[key]
    result={}
    for label,forms in groups.items():
        is_prefix=label.isdigit() and any(other!=label and other.startswith(label) for other in groups)
        values=[]
        for form in forms:
            ids=backend.encode(form)
            if not ids: continue
            value=score(ids)
            if is_prefix and form.strip().isdigit(): value+=termination(ids)
            values.append(value)
        result[label]=logsumexp(values)
    return result, continuation_tokens


def slow_readout(backend, ids, groups, config, meter):
    """One autoregressive session: one prompt prefill, cached draft/marker/readout."""
    if hasattr(backend,'slow_session'):
        session,parsed=backend.slow_session(ids,config['routing']['unparsed_policy'],meter)
        if session is None: return None,meter['draft'],False,False,0
        # Marginalize textual whitespace forms; no full-vocabulary CPU copy.
        scores,probes=sequence_scores(session,backend,groups,meter)
        return scores,meter['draft']+meter['forced'],not parsed,False,probes
    meter['prompt'] += len(ids)
    meter['engine_submitted'] += len(ids)
    meter['engine_input']=meter.get('engine_input',0)+len(ids)
    meter['engine_calls']=meter.get('engine_calls',0)+1
    session = backend.open_session(ids)
    import random
    rng = random.Random(config['generation']['seed'])
    generated = []
    marker = config['prompt']['answer_marker']
    found = False
    for _ in range(config['generation']['max_new_tokens']):
        check_deadline(meter)
        logits = dict(session.logits)
        g = config['generation']
        for seen in set(generated):
            value = logits[seen]
            logits[seen] = value * g['repetition_penalty'] if value < 0 else value / g['repetition_penalty']
        token = max(logits, key=logits.get)
        if g['do_sample']:
            ordered = sorted(logits, key=logits.get, reverse=True)
            if g['top_k']:
                ordered = ordered[:g['top_k']]
            weights = [math.exp((logits[t]-logits[token])/g['temperature']) for t in ordered]
            mass, cutoff = 0, sum(weights)*g['top_p']
            end = 0
            while end < len(weights) and mass < cutoff:
                mass += weights[end]
                end += 1
            token = rng.choices(ordered[:end], weights=weights[:end])[0]
        generated.append(token)
        meter['output'] += 1
        meter['draft'] += 1
        meter['engine_generated']=meter.get('engine_generated',0)+1
        session.advance([token])
        text = backend.decode(generated)
        if re.search(r'(?m)^[ \t]*' + re.escape(marker) + r'$', text):
            found = True
            break
        if token in getattr(backend, 'eos_ids', ()):
            break
    if not found and config['routing']['unparsed_policy'] == 'fast':
        return None, len(generated), False, False, 0
    if not found:
        suffix = backend.encode('\n' + marker)
        meter['output'] += len(suffix)
        meter['forced'] += len(suffix)
        meter['engine_input']=meter.get('engine_input',0)+len(suffix);meter['engine_submitted']+=len(suffix)
        session.advance(suffix)
        generated.extend(suffix)
    # Whitespace is a form branch, never a top-1 destructive advance.
    whitespace = False
    scores, probes = sequence_scores(session, backend, groups, meter)
    return scores, len(generated), not found, whitespace, probes


def label_margin(labels, scores, kind="logratio"):
    """Aggregated label distribution at T=1, before any calibration/floor."""
    if len(labels) < 2:
        return math.inf
    values = sorted((scores[label] for label in labels), reverse=True)
    if kind == 'logratio':
        return values[0] - values[1]
    if kind != 'probdiff':
        raise ValueError('Invalid margin kind')
    weights = [math.exp(v-values[0]) for v in values]
    return (weights[0]-weights[1])/sum(weights)


def fast_readout(backend, ids, labels, groups, meter):
    meter["prompt"]+=len(ids)
    if not getattr(backend,"engine_metering",False):
        meter["engine_submitted"]+=len(ids);meter['engine_input']+=len(ids)
        meter['hf_prefill_tokens']+=len(ids)
    try:
        tokens = candidate_ids(backend, groups)
    except ValueError:
        tokens = {label: [] for label in labels}
    probes = 0
    if any(not values for values in tokens.values()):
        session=backend.open_session(ids,meter=meter) if getattr(backend,'engine_metering',False) else backend.open_session(ids)
        if hasattr(session,'cache'): session.meter=meter
        scores, probes = sequence_scores(session, backend, groups, meter)
    else:
        wanted = sorted({t for values in tokens.values() for t in values})
        logits=backend.next_logits(ids,wanted,meter=meter) if getattr(backend,'engine_metering',False) else backend.next_logits(ids,wanted)
        scores = aggregate(logits, tokens)
        if not getattr(backend,'engine_metering',False):
            meter['output']+=backend.readout_output_tokens;meter['probes']+=backend.readout_output_tokens
            meter['engine_generated']+=backend.readout_output_tokens
    return scores, probes + backend.readout_output_tokens


def release_failed_session(backend):
    import gc
    gc.collect()
    torch=getattr(backend,'torch',None)
    cuda=getattr(torch,'cuda',None)
    if cuda is not None:
        try: cuda.empty_cache()
        except Exception: pass


def fast_attempt_with_retry(backend,task,config,ids,labels,groups,meter):
    try:
        scores,out=fast_readout(backend,ids,labels,groups,meter)
        return scores,out,None
    except Exception as exc:
        if unrecoverable_engine_error(exc):
            raise
        cuda=getattr(getattr(backend,'torch',None),'cuda',None)
        oom_type=getattr(cuda,'OutOfMemoryError',MemoryError)
        if not isinstance(exc,(MemoryError,oom_type)) and 'out of memory' not in str(exc).lower():
            raise
    # Leave the failed forward traceback before clearing allocator state.
    release_failed_session(backend)
    import copy
    reduced=copy.deepcopy(config)
    budget=min(config['limits']['oom_retry_prompt_tokens'],max(1,config['limits']['max_prompt_tokens']//2))
    reduced['limits']['max_prompt_tokens']=budget
    reduced['limits']['physical_max_prompt_tokens']=budget
    _,(_,new_groups,new_ids,_),length=prepare_task(task,backend,reduced)
    scores,out=fast_readout(backend,new_ids,labels,new_groups,meter)
    return scores,out,length


def slow_attempt(backend,ids,groups,config,meter):
    # Catch in this frame, then leave except before collecting. The failed
    # readout/generate traceback (and its KV references) no longer exists.
    try:
        return slow_readout(backend,ids,groups,config,meter)
    except Exception as exc:
        if unrecoverable_engine_error(exc):
            raise ServiceFailure('unrecoverable engine failure') from exc
        timed_out=isinstance(exc,TimeoutError)
    release_failed_session(backend)
    if timed_out: raise TimeoutError('Slow deadline exceeded')
    raise RuntimeError('Slow session failed')


def bounded_slow_attempt(backend,ids,groups,config,meter,model_lock, *, on_service_failure=None):
    """Bound response wait; an in-flight synchronous engine call drains under lock.

    No worker may start another engine call after cancellation. Timeout usage is
    explicitly estimated and reconciliation rejects the run.
    """
    import queue,time
    result=queue.Queue(maxsize=1)
    timeout=config['generation'].get('wall_timeout_seconds',60)
    meter['cancel_event']=threading.Event();meter['deadline']=time.monotonic()+timeout
    def work():
        try:
            with model_lock:
                check_deadline(meter)
                value=slow_attempt(backend,ids,groups,config,meter)
                check_deadline(meter)
            result.put((True,value))
        except Exception as exc:
            fatal = unrecoverable_engine_error(exc)
            if fatal and on_service_failure is not None:
                on_service_failure()
            result.put((False,ServiceFailure if fatal else type(exc)))
    worker=threading.Thread(target=work,daemon=True);worker.start()
    try: ok,value=result.get(timeout=timeout)
    except queue.Empty:
        meter['cancel_event'].set();meter['engine_usage_unknown']=True
        raise TimeoutError('Slow wall timeout; in-flight call draining')
    if not ok:
        if value is ServiceFailure: raise ServiceFailure('unrecoverable engine failure')
        if value is TimeoutError: raise TimeoutError('Slow wall timeout')
        raise RuntimeError('Slow session failed')
    return value


def threshold_identity(config):
    import hashlib
    import inspect
    # The renderer source binds label layout, and settings bind both prompts.
    canonical=json.dumps({'prompt':{k:config['prompt'][k] for k in ('system','enable_thinking')},'render':inspect.getsource(render),'choice_codes':inspect.getsource(choice_codes),
        'generation':{k:v for k,v in config['generation'].items() if k not in ('max_new_tokens','reasoning_words','wall_timeout_seconds')}},sort_keys=True,ensure_ascii=False)
    m=config['model']
    return dict(model_key=m['model_key'],quant=m['quant'],adapter=m.get('adapter'),
                prompt_sha=hashlib.sha256(canonical.encode('utf-8')).hexdigest(),backend=m['backend'],
                margin_kind=config['routing']['margin_kind'],max_new_tokens=config['generation']['max_new_tokens'],
                reasoning_words=config['generation']['reasoning_words'],
                max_prompt_tokens=config['limits']['max_prompt_tokens'],request_contract='typesafe.criteria.v1')


def validate_threshold(config,allow_placeholder=False):
    identity=threshold_identity(config)
    fit=config['routing'].get('margin_threshold_fitted_for',{})
    reasons=[]
    if fit.get('placeholder',True): reasons.append('placeholder')
    if any(fit.get(k)!=identity[k] for k in ('model_key','quant','adapter','prompt_sha','margin_kind','max_prompt_tokens','request_contract')):
        reasons.append('identity mismatch')
    if fit.get('backend')!=identity['backend']: reasons.append('backend mismatch')
    if reasons and config['routing']['mode']=='routed' and not allow_placeholder:
        raise ValueError('Unfitted margin threshold: '+', '.join(reasons))
    return ['margin threshold degraded: '+', '.join(reasons)] if reasons else []


class Service:
    def __init__(self, config, calibration, backend, counter, *, allow_placeholder_threshold=False):
        validate_config(config)
        threshold_warnings=validate_threshold(config,allow_placeholder_threshold)
        self.model_lock = threading.RLock()
        self._failure_lock = threading.Lock()
        self.item_failure_streak = 0
        self.dead = False
        self.config, self.backend, self.counter = config, backend, counter
        counter.configure_cost(config)
        key = config['model']['model_key']
        self.calibration = {path: validate_calibration(calibration, key, path) for path in ('fast', 'slow')}
        model = calibration['models'][key]
        self.warnings = ['calibration placeholder: refit required'] if (
            model.get('placeholder') or any(b.get('placeholder') for b in self.calibration.values())) else []
        self.warnings.extend(threshold_warnings)
        if not config['routing'].get('cost_fuse',{}).get('enabled',True):
            self.warnings.append('cost fuse explicitly disabled')
        if 'fallback' in model:
            self.calibration['fallback']=validate_calibration(calibration,key,'fallback')
        else:
            self.calibration['fallback']=self.calibration['fast']
            self.warnings.append('fallback calibration missing: using fast block')

    def _mark_dead(self):
        with self._failure_lock:
            self.dead = True

    def is_dead(self):
        if self.dead:
            return True
        # Read the runtime's existing flag; this performs no engine RPC.
        resource = self.backend
        for name in ('engine', 'llm_engine', 'engine_core', 'resources'):
            resource = getattr(resource, name, None)
        if getattr(resource, 'engine_dead', False):
            self._mark_dead()
        return self.dead

    def _raise_if_service_failure(self, exc):
        if unrecoverable_engine_error(exc) or self.is_dead():
            self._mark_dead()
            raise ServiceFailure('unrecoverable engine failure') from exc

    def _item_failed(self):
        with self._failure_lock:
            self.item_failure_streak += 1
            if self.item_failure_streak >= ITEM_FAILURE_LIMIT:
                self.dead = True
            return self.item_failure_streak

    def _item_succeeded(self):
        with self._failure_lock:
            self.item_failure_streak = 0

    def health(self):
        dead = self.is_dead()
        warnings = list(self.warnings)
        if dead:
            warnings.append('service failure')
        reference=getattr(self.backend,'generation_reference',None)
        if reference and not reference['verified']:warnings.append('generation reference '+reference['status'])
        if hasattr(self.backend,'forward_counters') and not getattr(self.backend,'stats_available',False):warnings.append('HF forward hook unavailable; usage estimated')
        if getattr(self.backend,'forward_counters',{}).get('uncounted_forwards',0):warnings.append('HF uncounted forwards; usage estimated')
        if getattr(self.counter, 'memory_only', False):
            warnings.append('quota persistence unavailable: memory counts, fast only')
        return {'status': 'error' if dead else 'warning' if warnings else 'ok', 'warnings': warnings,
                'model_key': self.config['model']['model_key'], 'mode':self.config['routing']['mode'],
                'quota_tier':self.config['routing'].get('tier','normal'), 'budget_tiers':self.config['routing'].get('tiers',{}),
                'threshold_identity':threshold_identity(self.config),
                'cost_fuse':self.counter.cost_fuse.snapshot(self.config['routing']['quota']),
                'accuracy_only':self.config.get('bench',{}).get('accuracy_only',False),
                'generation_reference':reference,'quantization_kernel':getattr(self.backend,'quantization_kernel',None),
                'prefix_caching':getattr(self.backend,'prefix_caching',None),
                'supports_slow_session':callable(getattr(self.backend,'slow_session',None)) or callable(getattr(self.backend,'open_session',None))}

    def engine_stats(self):
        with self.model_lock:
            return self.backend.engine_stats()

    def warmup(self):
        ids = self.backend.prompt_ids(self.config['prompt']['system'], 'Ready?', False)
        self.backend.next_logits(ids, [self.backend.encode('yes')[0]])
        self.backend.generate(ids, 1)
        meter=new_meter()
        with self.model_lock:
            if callable(getattr(self.backend,'slow_session',None)):
                session,_=self.backend.slow_session(ids,'force_answer',meter,
                    max_new_tokens=self.config['server']['warmup_slow_tokens'])
                if session is not None:
                    branch=session.fork();branch.advance(self.backend.encode('yes'))
                    if hasattr(branch,'log_probs'): branch.log_probs([self.backend.encode('yes')[0]])
                    else: _=branch.logits
                    del branch,session
            elif callable(getattr(self.backend,'open_session',None)):
                import copy
                warm=copy.deepcopy(self.config)
                warm['generation']['max_new_tokens']=self.config['server']['warmup_slow_tokens']
                warm['routing']['unparsed_policy']='force_answer'
                slow_attempt(self.backend,ids,{'yes':['yes',' yes']},warm,meter)

    def _one(self, task):
        with self.counter.accounting():
            if self.is_dead():
                raise ServiceFailure('service failure')
            try:
                answer,detail=self._one_impl(task)
                if self.is_dead():
                    raise ServiceFailure('service failure')
                detail['threshold_identity']=threshold_identity(self.config)
                detail['cost_fuse']=self.counter.record_cost(detail['usage_bases'][detail['accounting']],detail['path']=='slow')
            except ItemFailure as exc:
                if self._item_failed() >= ITEM_FAILURE_LIMIT:
                    raise ServiceFailure('consecutive item failures') from exc
                raise
            except ServiceFailure:
                self._mark_dead()
                raise
            self._item_succeeded()
            return answer,detail

    def _one_impl(self, task):
        c, backend = self.config, self.backend
        uncounted_before=getattr(backend,'forward_counters',{}).get('uncounted_forwards',0)
        labels, _, groups = render(task)
        qtype = task['question']['type']
        detail = {'path':'fast', 'route_reason':'error', 'router_score':0.0,
                  'prompt_tokens':0, 'completion_tokens':0, 'n_new_tokens':0,
                  'prompt_tokens_details':{'cached_tokens':0,'engine_submitted_tokens':0},
                  'fast_scores':None,'slow_scores':None,'budget_tier':'normal',
                  'readout_output_tokens':0, 'forced_answer':False, 'draft_prediction':None}
        warning = None
        meters = []
        try:
            kept_task, (_, groups, fast_ids, slow_ids), length = prepare_task(task, backend, c)
            detail.update(length)
            long_item=length.get('long_input',False) or c['routing'].get('tier')=='long' or length.get('slow_prompt_exceeds_budget',False) or any(length[k] for k in ('truncated','character_precompressed','protected_content_exceeds_budget','physical_truncated'))
            hard=long_item if c['routing'].get('long_item_slow_policy','fast_only')=='fast_only' else (
                length['character_precompressed'] or (length['original_prompt_tokens'] or 0)>c['routing']['slow_max_prompt_tokens'] or length['protected_content_exceeds_budget'])
            hard=hard or c['routing'].get('tier')=='long' or length.get('slow_prompt_exceeds_budget',False)
            detail['long_item']=long_item
            r = c['routing']
            fast_scores = None
            fast_meter=new_meter()
            meters.append(fast_meter)
            try:
                with self.model_lock:
                    fast_scores, out, retry_length = fast_attempt_with_retry(backend,task,c,fast_ids,labels,groups,fast_meter)
                if retry_length is not None:
                    detail.update(retry_length)
                    detail['fast_oom_retry']=True
                    hard=True

            except Exception as exc:
                self._raise_if_service_failure(exc)
                raise ItemFailure('item could not be processed') from exc
            detail['prompt_tokens']+=fast_meter['prompt']
            detail['completion_tokens']+=fast_meter['output']
            detail['readout_output_tokens']+=fast_meter['probes']
            detail['prompt_tokens_details'].update(cached_tokens=fast_meter['cached'],engine_submitted_tokens=fast_meter['engine_submitted'])
            if fast_meter.get('engine_usage_unknown'): detail['usage_estimated']=True
            if fast_meter.get('cached_unknown'): detail['prompt_tokens_details']['cached_tokens_known']=False
            detail['fast_prompt_tokens']=fast_meter['prompt']
            detail['raw_label_scores']=detail['fast_scores']=fast_scores
            margin = label_margin(labels, fast_scores, r['margin_kind']) if fast_scores is not None else None
            detail.update(fast_margin=margin if margin is None or math.isfinite(margin) else None,
                          fast_margin_infinite=margin == math.inf, margin_kind=r['margin_kind'])
            wants_slow = fast_scores is not None and not hard and (
                r['mode'] == 'uniform' or (r['mode'] == 'routed' and margin < r['margin_threshold']))
            wants_slow = wants_slow and (callable(getattr(backend,'slow_session',None)) or callable(getattr(backend,'open_session',None)))
            detail['budget_tier']='long' if hard else 'hard' if wants_slow else 'normal'
            detail['admission_quota'],detail['admission_quota_source']=self.counter.cost_fuse.quota(r['quota'])
            with self.counter.decision(wants_slow, quota=detail['admission_quota']) as decision:
                detail.update(path=decision.path, route_reason='oom_retry' if detail.get('fast_oom_retry') else 'length' if hard else decision.reason,
                              quota_at_reservation=decision.snapshot)
                scores, block = fast_scores, 'fast'
                if decision.path == 'slow':
                    meter = new_meter()
                    meters.append(meter)
                    try:
                        scores, count, forced, whitespace, probes = bounded_slow_attempt(backend,slow_ids,groups,c,meter,self.model_lock,on_service_failure=self._mark_dead)
                        detail['slow_raw_label_scores']=detail['slow_scores']=scores
                        block = 'slow' if scores is not None else 'fallback'
                        detail['route'] = ('slow_forced_answer' if forced else 'slow') if scores is not None else 'slow_unparsed_fast'
                        if scores is None:
                            scores = fast_scores
                        detail.update(n_new_tokens=count, forced_answer=forced, whitespace_step=whitespace)
                    except Exception as exc:
                        self._raise_if_service_failure(exc)
                        scores, block = fast_scores, 'fallback'
                        warning = 'slow_failed: fast answer kept'
                        detail.update(fallback='slow_to_fast', route='slow_timeout_fast' if isinstance(exc,TimeoutError) else 'slow_failed_fast')
                        if isinstance(exc,TimeoutError):
                            warning='slow_timeout: fast answer kept'
                            detail['usage_estimated']=True
                            # Detach response accounting from the draining worker.
                            meter=dict(meter);meters[-1]=meter
                    finally:
                        if meter.get('engine_usage_unknown'): detail['usage_estimated']=True
                        if meter.get('cached_unknown'): detail['prompt_tokens_details']['cached_tokens_known']=False
                        detail['prompt_tokens'] += meter['prompt']
                        detail['completion_tokens'] += meter['output']
                        detail['n_new_tokens'] = meter['draft'] + meter['forced']
                        detail['readout_output_tokens'] += meter['probes']
                        detail['prompt_tokens_details']['cached_tokens'] += meter['cached']
                        detail['prompt_tokens_details']['engine_submitted_tokens'] += meter['engine_submitted']
                detail['calibration_block']=block
                if scores is None:
                    raise ItemFailure('item could not be processed')
                probs = probabilities(labels, scores, qtype, self.calibration.get(block, self.calibration['fast']))
            detail['quota_after_completion'] = decision.snapshot
        except (ItemFailure, QuotaStateError, ServiceFailure):
            raise
        except Exception as exc:
            self._raise_if_service_failure(exc)
            raise ItemFailure('item could not be processed') from exc
        if (getattr(backend,'forward_counters',{}).get('uncounted_forwards',0)>uncounted_before
                or hasattr(backend,'forward_counters') and not getattr(backend,'stats_available',False)):
            detail['usage_estimated']=True
            warning='HF count unavailable: usage estimated'
        detail['backend_warnings']=[w for m in meters for w in m.get('warnings',[])]
        if any(m.get('cached_unknown') for m in meters):detail['usage_estimated']=True
        submitted=sum(m['engine_submitted'] for m in meters)
        cached=sum(m['cached'] for m in meters)
        generated=sum(m.get('engine_generated',m['output']) for m in meters)
        engine_input=sum(m.get('engine_input',m['engine_submitted']-m['cached']) for m in meters)
        if detail.get('usage_estimated'):
            engine_input=max(engine_input,submitted)+c['generation']['max_new_tokens']
            generated+=c['generation']['max_new_tokens']
        detail['input_tokens_details']=dict(cached_tokens=cached,engine_submitted_tokens=submitted,
            engine_calls=sum(m.get('engine_calls',0) for m in meters),engine_generated_tokens=generated,
            cached_tokens_known=not any(m.get('cached_unknown') for m in meters),
            generate_nonempty_calls=sum(m.get('generate_nonempty_calls',0) for m in meters),
            hf_prefill_tokens=sum(m.get('hf_prefill_tokens',0) for m in meters))
        detail['usage_bases']={
            'engine':dict(input_tokens=engine_input,output_tokens=generated),
            'each_once':dict(input_tokens=detail['prompt_tokens'],output_tokens=detail['completion_tokens']),
            'submitted':dict(input_tokens=submitted,output_tokens=detail['completion_tokens'])}
        if detail.get('usage_estimated'):
            for values in detail['usage_bases'].values():
                values['input_tokens']=max(values['input_tokens'],engine_input)
                values['output_tokens']=max(values['output_tokens'],generated)
        detail['accounting']=c.get('usage',{}).get('basis','engine')
        detail['kept_prompt_tokens_max'] = detail.get('kept_prompt_tokens', 0)
        detail['total_tokens'] = detail['prompt_tokens'] + detail['completion_tokens']
        detail['prediction'] = max(labels, key=probs.get)
        request_type = task['question']['request_type']
        if qtype == 'noul':
            style = c.get('interface', {}).get('boolean_keys', 'request')
            if style == 'true_false' or (style == 'request' and request_type == 'boolean'):
                probs = {'false':probs['no'], 'true':probs['yes']}
        answer = {'type':request_type, 'probabilities':probs}
        if qtype == 'noul': answer['noul'] = float(probs.get('yes', probs.get('true')))
        elif qtype == 'choice': answer['choice'] = detail['prediction']
        if warning:
            answer['warning'] = warning
        return answer, detail

    def answer(self, payload):
        if self.is_dead():
            raise ServiceFailure('service failure')
        tasks = normalize_request(payload, self.config['server']['max_questions'])
        answers, details, errors, noul, choice = {}, {}, {}, {}, {}
        for name, task in tasks:
            answer, detail = self._one(task)
            answers[name], details[name] = answer, detail
            if 'warning' in answer:
                errors[name] = answer['warning']
            if task['question']['type'] == 'noul':
                noul[name] = answer['probabilities'].get('true', answer['probabilities'].get('yes'))
            else:
                choice[name] = detail['prediction']
        basis=self.config.get('usage',{}).get('basis','engine')
        alternatives={b:{k:sum(d['usage_bases'][b][k] for d in details.values())
            for k in ('input_tokens','output_tokens')} for b in ('engine','each_once','submitted')}
        p,o=alternatives[basis]['input_tokens'],alternatives[basis]['output_tokens']
        token_details={k:sum(d['input_tokens_details'][k] for d in details.values()) for k in
            ('cached_tokens','engine_submitted_tokens','engine_calls','engine_generated_tokens','generate_nonempty_calls','hf_prefill_tokens')}
        token_details['cached_tokens_known']=all(d['input_tokens_details']['cached_tokens_known'] for d in details.values())
        result={'answers':answers,'model':self.config['model']['name'],
            'usage':dict(input_tokens=p,output_tokens=o,prompt_tokens=p,completion_tokens=o,total_tokens=p+o,
                cost_fuse=next(reversed(details.values()))['cost_fuse'],
                accounting=basis,usage_estimated=any(d.get('usage_estimated',False) for d in details.values()),
                alternatives=alternatives,questions=details,input_tokens_details=token_details,
                prompt_tokens_details=token_details,token_accounting=basis)}
        for key, values in (('noul',noul),('choice',choice)):
            if values:
                result[key] = next(iter(values.values())) if len(tasks) == 1 else values
        if errors:
            result['partial_errors'] = errors
        return result
