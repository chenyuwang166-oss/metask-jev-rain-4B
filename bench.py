"""Sequential, model-free HTTP benchmark client for the submission v2 API."""

import argparse
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PUBLIC_FILES = (("easy.jsonl", "easy"), ("original.jsonl", "standard"), ("hard.jsonl", "hard"))
TYPES = ("noul", "choice", "score")
ALIASES = {"boolean": "noul", "enum": "choice"}
QUESTION_NAME = "decision"


def canonical_type(value):
    value = ALIASES.get(value, value)
    if value not in TYPES:
        raise ValueError("unsupported question type")
    return value


def _read_rows(path):
    with Path(path).expanduser().open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            yield row


def normalize_task(task, tier=None, require_expected=False):
    """Keep the source's label order, including choice tie-breaking order."""
    if not isinstance(task.get("id"), str) or not task["id"]:
        raise ValueError("task id must be a nonempty string")
    if "state" not in task or not isinstance(task.get("question"), dict):
        raise ValueError("task needs state and question")
    question = dict(task["question"])
    qtype = canonical_type(question.get("type"))
    question["request_type"] = question.get("request_type",question["type"])
    question["type"] = qtype
    instructions = question.get("instructions", question.get("description"))
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("question needs nonempty instructions or description")
    question["instructions"] = instructions
    question.pop("description", None)
    criteria = question.get("criteria")
    if qtype == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an object")
        labels = ["no", "yes"]
    elif qtype == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("score criteria must be a nonempty list")
        labels = [str(index) for index in range(len(criteria))]
    else:
        labels = task.get("labels", question.get("labels"))
        if labels is None and isinstance(criteria, dict):
            labels = list(criteria)
        if labels is None and isinstance(criteria, list):
            labels = criteria
        if not isinstance(labels, list) or not 1 <= len(labels) <= 676:
            raise ValueError("choice requires between 1 and 676 labels")
        labels = list(labels)
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("choice labels must be nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("choice labels must be distinct")
        question["labels"] = labels
    expected = task.get("expected")
    if expected is not None:
        if qtype == "noul" and isinstance(expected, bool):
            expected = "yes" if expected else "no"
        elif qtype == "score":
            expected = str(expected)
        if expected not in labels:
            raise ValueError("expected answer is not a valid label")
    elif require_expected:
        raise ValueError("public source task is missing expected answer")
    result = dict(task)
    result.update(question=question, type=qtype, labels=labels, expected=expected)
    result["tier"] = tier if tier is not None else task.get("tier", "custom")
    return result


def load_jsonl_tasks(path):
    tasks = [normalize_task(row) for row in _read_rows(path)]
    _require_unique(tasks)
    if not tasks:
        raise ValueError("input contains no tasks")
    return tasks


def _require_unique(rows):
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("duplicate task id")
        seen.add(row["id"])


def load_public_tasks(index, public, expected_count):
    """Use only id/type/tier from the index; all task content comes from public."""
    metadata = []
    for cached in _read_rows(index):
        # Do not use cached expected labels, logits, predictions or prompt content.
        item = {key: cached.get(key) for key in ("id", "type", "tier")}
        if not isinstance(item["id"], str) or not item["id"]:
            raise ValueError("index id must be a nonempty string")
        item["type"] = canonical_type(item["type"])
        if item["tier"] not in {tier for _, tier in PUBLIC_FILES}:
            raise ValueError("index tier is not a public tier")
        metadata.append(item)
    _require_unique(metadata)
    if len(metadata) != expected_count:
        raise ValueError("index task count does not match expected_public_count")
    by_id = {}
    for filename, tier in PUBLIC_FILES:
        for row in _read_rows(Path(public).expanduser() / filename):
            if "tier" in row and row["tier"] != tier:
                raise ValueError("public task tier conflicts with its source file")
            task = normalize_task(row, tier=tier, require_expected=True)
            if task["id"] in by_id:
                raise ValueError("duplicate public task id")
            by_id[task["id"]] = task
    selected = []
    for meta in metadata:
        task = by_id.get(meta["id"])
        if task is None:
            raise ValueError("indexed task missing from public sources")
        if task["type"] != meta["type"] or task["tier"] != meta["tier"]:
            raise ValueError("indexed type or tier does not match public source")
        selected.append(task)
    return selected


def request_payload(task, original_types=False, model=None, label_order=False):
    question=task['question']
    q={'type':question['type'],'instructions':question['instructions']}
    if question.get('criteria') is not None: q['criteria']=question['criteria']
    if original_types:q['type']=question.get('request_type',question['type'])
    if label_order and task['type']=='choice': q['labels']=list(task['labels'])
    return {'state':task['state'],'model':model,'questions':{QUESTION_NAME:q}}


def endpoint_url(endpoint):
    parts = urllib.parse.urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("endpoint must be an HTTP base URL or the systemone route")
    if parts.username or parts.password:
        raise ValueError("endpoint must not contain credentials")
    path = parts.path.rstrip("/")
    if path not in {"", "/v1/systemone"}:
        raise ValueError("endpoint path must be empty or /v1/systemone")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/v1/systemone", "", ""))


def post_json(endpoint, payload, timeout):
    request = urllib.request.Request(endpoint_url(endpoint),
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_prediction(response, task):
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise ValueError("response must contain answers")
    if not isinstance(response.get("model"), str) or not response["model"]:
        raise ValueError("response must identify its model")
    answer = response["answers"].get(QUESTION_NAME)
    if not isinstance(answer, dict) or canonical_type(answer.get("type")) != task["type"]:
        raise ValueError("response question type mismatch")
    if answer.get("warning") or response.get("partial_errors"):
        raise ValueError("degraded API response")
    probabilities = answer.get("probabilities")
    if task["type"]=="noul" and isinstance(probabilities,dict) and set(probabilities)=={"true","false"}:
        probabilities={"yes":probabilities["true"],"no":probabilities["false"]}
    if not isinstance(probabilities, dict) or set(probabilities) != set(task["labels"]):
        raise ValueError("response probabilities must have exactly the requested labels")
    values = list(probabilities.values())
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("response contains invalid probabilities")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("response probabilities must sum to one")
    # Match official score_task: lexicographically smallest label wins ties.
    return max(sorted(task["labels"]), key=lambda label: probabilities[label]), probabilities


def _token_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _safe_usage_json(value):
    """Keep malformed optional metadata from making result JSON unwritable."""
    if value is None or isinstance(value, (str, bool, int)):
        return value, False
    if isinstance(value, float):
        return (value, False) if math.isfinite(value) else (None, True)
    if isinstance(value, list):
        parts = [_safe_usage_json(item) for item in value]
        return [item for item, _ in parts], any(changed for _, changed in parts)
    if isinstance(value, dict):
        result, changed = {}, False
        for key, item in value.items():
            if not isinstance(key, str):
                changed = True
                continue
            result[key], item_changed = _safe_usage_json(item)
            changed = changed or item_changed
        return result, changed
    return None, True


def extract_usage(response):
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    usage, sanitized = _safe_usage_json(usage)
    questions = usage.get("questions", {})
    detail = questions.get(QUESTION_NAME, {}) if isinstance(questions, dict) else {}
    detail = detail if isinstance(detail, dict) else {}
    prompt = _token_count(usage.get("input_tokens", usage.get("prompt_tokens")))
    completion = _token_count(usage.get("output_tokens", usage.get("completion_tokens")))
    total = _token_count(usage.get("total_tokens"))
    n_new_tokens = _token_count(detail.get("n_new_tokens"))
    complete = prompt is not None and completion is not None and total is not None
    complete = complete and total == prompt + completion
    complete = complete and n_new_tokens is not None
    if usage.get("accounting") is None:complete=complete and n_new_tokens<=completion if complete else False
    complete = complete and not sanitized and not detail.get("usage_estimated", False) and not usage.get("usage_estimated",False)
    path = detail.get("path")
    if not isinstance(path, str) or path not in {"fast", "slow"}:
        path = None
    router_score = detail.get("router_score")
    if isinstance(router_score, bool) or not isinstance(router_score, (int, float)) or not math.isfinite(router_score):
        router_score = None
    return {"usage": usage, "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "usage_complete": bool(complete), "usage_sanitized": sanitized,
            "n_new_tokens": n_new_tokens, "path": path, "router_score": router_score,
            "fast_prompt_tokens":detail.get("fast_prompt_tokens"), "prefix_caching":response.get("prefix_caching"),
            "fast_scores":detail.get("fast_scores",detail.get("raw_label_scores")),
            "slow_scores":detail.get("slow_scores",detail.get("slow_raw_label_scores")),
            "slow_raw_label_scores":detail.get("slow_raw_label_scores"),
            "calibration_block":detail.get("calibration_block"),"threshold_identity":detail.get("threshold_identity"),
            "budget_tier":detail.get("budget_tier"), "cost_fuse":detail.get("cost_fuse"),
            "raw_label_scores":detail.get("raw_label_scores"), "fast_margin":detail.get("fast_margin"), "margin_kind":detail.get("margin_kind")}


def run_benchmark(tasks, endpoint, config, client=post_json, clock=time.perf_counter, writer=None, concurrency=1):
    """One request per task, no retries; ordered rows, latency includes each call."""
    if type(concurrency) is not int or concurrency<1: raise ValueError('concurrency must be positive')
    if concurrency>1:
        from concurrent.futures import ThreadPoolExecutor
        def one(task): return run_benchmark([task],endpoint,config,client,clock)[0]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows=list(pool.map(one,tasks))
        for row in rows: row["concurrency"]=concurrency
        if writer is not None:
            for row in rows: writer.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            writer.flush()
        return rows
    rows = []
    timeout = config["bench"]["timeout_seconds"]
    for task in tasks:
        row = {"request_order":"labels" if config.get("bench",{}).get("label_order",False) else "criteria", "id": task["id"], "type": task["type"], "tier": task.get("tier"),
               "benchmark_set":task.get("benchmark_set"), "concurrency":concurrency, "expected": task.get("expected"), "labels": task["labels"], "pred": None,
               "probabilities": None, "ok": False, "error": None, "model": None}
        row.update(extract_usage({}))
        start = clock()
        try:
            response = client(endpoint, request_payload(task,config.get("bench",{}).get("original_types",False),config["model"]["name"],config.get("bench",{}).get("label_order",False)), timeout)
        except Exception as exc:
            # Exception text can include an endpoint or credentials; never save it.
            row["error"] = {"type": type(exc).__name__, "message": "request failed"}
            row["latency_seconds"] = clock() - start
        else:
            row["latency_seconds"] = clock() - start
            row.update(extract_usage(response))
            if isinstance(response, dict) and isinstance(response.get("model"), str):
                row["model"] = response["model"]
            try:
                row["pred"], row["probabilities"] = parse_prediction(response, task)
                row["ok"] = True
            except (ValueError, KeyError, TypeError) as exc:
                row["error"] = {"type": type(exc).__name__, "message": "invalid API response"}
        row["correct"] = bool(row["ok"] and row["pred"] == row["expected"]) if row["expected"] is not None else None
        rows.append(row)
        if writer is not None:
            writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            writer.flush()
    return rows


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(rows, config, expected_count=None):
    count = len(rows)
    expected_count = count if expected_count is None else expected_count
    successes = sum(bool(row["ok"]) for row in rows)
    complete = count > 0 and count == expected_count and successes == count
    usage_complete = count > 0 and all(row["usage_complete"] for row in rows)
    routing_complete = count > 0 and all(row["path"] in {"fast", "slow"} for row in rows)
    new_tokens_complete = count > 0 and all(row["n_new_tokens"] is not None for row in rows)
    latencies = [row["latency_seconds"] for row in rows]
    p50 = statistics.median(latencies) if latencies else None
    price_key = config["pricing"]["model_key"]
    observed_models = sorted({row["model"] for row in rows if isinstance(row.get("model"), str) and row["model"]})
    price_model_matches = count > 0 and all(config["pricing"].get("submission_models",{}).get(row.get("model"),row.get("model")) == price_key for row in rows)
    prices = config["pricing"]["models"].get(price_key, {})
    input_price = prices.get("input_per_million")
    output_price = prices.get("output_per_million")
    price_known = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                      and math.isfinite(value) and value >= 0 for value in (input_price, output_price))
    total_prompt = sum(row["prompt_tokens"] for row in rows) if usage_complete else None
    total_completion = sum(row["completion_tokens"] for row in rows) if usage_complete else None
    cost = (total_prompt * input_price + total_completion * output_price) / 1_000_000 if usage_complete and price_known and price_model_matches else None
    cost_per_1000 = cost / count * 1000 if cost is not None else None
    by_type = {}
    for qtype in TYPES:
        subset = [row for row in rows if row["type"] == qtype]
        labeled = [row for row in subset if row["expected"] is not None]
        correct = sum(row["correct"] is True for row in labeled)
        by_type[qtype] = {"count": len(subset), "labeled_count": len(labeled), "correct": correct,
                          "accuracy": correct / len(subset) if subset and len(labeled) == len(subset) else None}
    limits = config["limits"]
    scale=config.get('bench',{}).get('latency_scale',1)
    offset=config.get('bench',{}).get('latency_offset_seconds',0)
    scaled_p50=p50*scale+offset if p50 is not None else None
    margin=config.get('pricing',{}).get('margin',0)
    if not isinstance(margin,(float,int)) or not 0<=margin<1: raise ValueError('pricing.margin must be in [0,1)')
    cost_limit=limits['cost_per_1000_decisions_usd']*(1-margin)
    eligible = None
    if complete and usage_complete and routing_complete and new_tokens_complete and cost_per_1000 is not None:
        eligible = scaled_p50 <= limits["median_latency_seconds"] and cost_per_1000 <= cost_limit
    slow_share = sum(row["path"] == "slow" for row in rows) / count if routing_complete else None
    fast_q = 0.5 / (1 - slow_share) if slow_share is not None and slow_share < 1 else None
    fast_latencies = [row["latency_seconds"] for row in rows if row["path"] == "fast" and row["ok"]]
    path_stats = {}
    for path in ("fast", "slow"):
        subset = [row for row in rows if row["path"] == path and row["ok"]]
        times = [row["latency_seconds"] for row in subset]
        path_stats[path] = {"count":len(subset), "p50":percentile(times, .5),
                            "p75":percentile(times,.75),"mean":statistics.mean(times) if times else None,
                            "scaled_p75":percentile(times,.75)*scale+offset if times else None,
                            "scaled_mean":statistics.mean(times)*scale+offset if times else None,
                            "p90":percentile(times, .9), "p99":percentile(times, .99),
                            'average_prompt_tokens':statistics.mean(r['prompt_tokens'] for r in subset) if subset and all(r['prompt_tokens'] is not None for r in subset) else None,
                            'average_completion_tokens':statistics.mean(r['completion_tokens'] for r in subset) if subset and all(r['completion_tokens'] is not None for r in subset) else None}
    tail = config.get("bench", {}).get("long_tail", {"count":23, "prompt_tokens":79000, "total_decisions":1624})
    tail_kept = min(tail["prompt_tokens"], limits.get("max_prompt_tokens", 16000))
    tail_cost = tail["count"] * tail_kept * input_price / 1e6 if price_known else None
    projection = {"count":tail["count"], "original_prompt_tokens_each":tail["prompt_tokens"],
                  "kept_prompt_tokens_each":tail_kept, "path":"fast", "output_tokens_each":0,
                  "estimated_cost_usd":tail_cost,
                  "untruncated_fast_cost_usd":tail["count"] * tail["prompt_tokens"] * input_price / 1e6 if price_known else None,
                  "assumption":"fast-only long-tail projection; final physical guard can discard protected content",
                  "projected_cost_per_1000_decisions_usd":
                      ((tail["total_decisions"]-tail["count"]) * cost/count + tail_cost) / tail["total_decisions"] * 1000
                      if cost is not None and count and tail_cost is not None else None}
    cost_basis = {'A':cost_per_1000,'B':None,'C':None}
    accounting_details = [row.get('usage',{}).get('prompt_tokens_details') for row in rows]
    if cost is not None and count and all(isinstance(d,dict) and
            all(type(d.get(k)) is int and d[k]>=0 for k in ('cached_tokens','engine_submitted_tokens'))
            and d['cached_tokens']<=d['engine_submitted_tokens'] for d in accounting_details):
        submitted=sum(d['engine_submitted_tokens'] for d in accounting_details)
        cached=sum(d['cached_tokens'] for d in accounting_details)
        cost_basis['B']=(submitted*input_price+total_completion*output_price)/1e6/count*1000
        if all(d.get('cached_tokens_known',True) for d in accounting_details):
            cost_basis['C']=((submitted-cached)*input_price+total_completion*output_price)/1e6/count*1000
    for name,legacy in [('engine','C'),('each_once','A'),('submitted','B')]:
        values=[r.get('usage',{}).get('alternatives',{}).get(name) for r in rows]
        if cost is not None and count and all(isinstance(v,dict) for v in values):
            cost_basis[legacy]=sum(v['input_tokens']*input_price+v['output_tokens']*output_price for v in values)/1e6/count*1000
        cost_basis[name]=cost_basis[legacy]
    tail_tiers={}
    for budget in tail.get('projection_budgets',[]):
        extra=tail['count']*budget*input_price/1e6 if price_known else None
        tail_tiers[str(budget)]={'count':tail['count'],'prompt_tokens_each':budget,
            'fast_only_estimated_cost_usd':extra,
            'projected_cost_per_1000_by_basis':{k:((tail['total_decisions']-tail['count'])*v/1000+extra)
                /tail['total_decisions']*1000 if v is not None and extra is not None else None for k,v in cost_basis.items()}}
    speed_rows=[r for r in rows if r.get('benchmark_set')=='open-set' and r.get('tier') in ('standard','judge') and r.get('concurrency',1)==1]
    speed={'count':len(speed_rows),'scope':'explicit open-set standard+judge serial',
           'scaled_p50':None,'scaled_p95':None,'complete':bool(speed_rows) and all(r['ok'] for r in speed_rows)}
    if speed['complete']:
        speed.update(scaled_p50=percentile([r['latency_seconds'] for r in speed_rows],.5)*scale+offset,
                     scaled_p95=percentile([r['latency_seconds'] for r in speed_rows],.95)*scale+offset)
    q_latency=percentile(fast_latencies,fast_q) if fast_q is not None and fast_q<=1 else None
    budget_stats={}
    for tier in ('normal','hard','long'):
        subset=[r for r in rows if r.get('budget_tier')==tier]
        labeled=[r for r in subset if r.get('correct') is not None]
        known=bool(subset) and price_known and all(r['usage_complete'] for r in subset)
        budget_stats[tier]={'count':len(subset),'accuracy':sum(r['correct'] for r in labeled)/len(labeled) if labeled else None,
            'cost_usd':sum(r['prompt_tokens']*input_price+r['completion_tokens']*output_price for r in subset)/1e6 if known else None}
    by_tier={}
    for tier in sorted({str(r.get('tier')) for r in rows}):
        subset=[r for r in rows if str(r.get('tier'))==tier];labeled=[r for r in subset if r.get('correct') is not None]
        by_tier[tier]={'count':len(subset),'accuracy':sum(r['correct'] for r in labeled)/len(labeled) if labeled else None}
    p_projection=project_cost(rows,config,config.get('bench',{}).get('projection_prompt_tokens'))
    accuracy_only=config.get('bench',{}).get('accuracy_only',False)
    if accuracy_only:eligible=None
    return {"accuracy_only":accuracy_only,"by_tier":by_tier,"P_projection":p_projection,
            "cost_gate":('UNKNOWN' if cost_per_1000 is None or accuracy_only else 'PASS' if cost_per_1000<=.85*cost_limit else 'WARN' if cost_per_1000<=cost_limit else 'FAIL'),
            "p75_latency_seconds":percentile(latencies,.75),"mean_latency_seconds":statistics.mean(latencies) if latencies else None,
            "scaled_p75_latency_seconds":percentile(latencies,.75)*scale+offset if latencies else None,
            "scaled_mean_latency_seconds":statistics.mean(latencies)*scale+offset if latencies else None,
            "by_budget_tier":budget_stats,"official_speed_subset":speed,"fast_required_quantile_meets_raw_limit":q_latency<=limits.get('raw_median_latency_seconds',.54) if q_latency is not None else None,
            "count": count, "expected_count": expected_count, "successful_count": successes,
            "failed_count": count - successes, "complete": complete,
            "by_path":path_stats, "long_tail_projection":projection,
            "cost_per_1000_by_basis":cost_basis, "long_tail_by_budget":tail_tiers,
            "fast_required_quantile":fast_q,
            "fast_required_quantile_latency_seconds":percentile(fast_latencies, fast_q) if fast_q is not None and fast_q <= 1 else None,
            "fast_quantile_assumption":"all slow decisions exceed median target; successful fast samples only",
            "successful_p50_latency_seconds":percentile([r["latency_seconds"] for r in rows if r["ok"]], .5),
            "latency_scale":scale,"latency_offset_seconds":offset,"scaled_p50_latency_seconds":scaled_p50,
            "pricing_margin":margin,"cost_limit_after_margin":cost_limit,
            "cost_headroom_usd_per_1000":cost_limit-cost_per_1000 if cost_per_1000 is not None else None,
            "latency_scope": "all_attempts_including_failures", "p50_latency_seconds": p50,
            "p99_latency_seconds": percentile(latencies, 0.99),
            "slow_count": sum(row["path"] == "slow" for row in rows),
            "slow_share": sum(row["path"] == "slow" for row in rows) / count if routing_complete else None,
            "usage_complete": usage_complete, "routing_complete": routing_complete,
            "average_prompt_tokens": total_prompt / count if usage_complete else None,
            "average_completion_tokens": total_completion / count if usage_complete else None,
            "average_total_tokens": (total_prompt + total_completion) / count if usage_complete else None,
            "average_new_tokens": sum(row["n_new_tokens"] for row in rows) / count if new_tokens_complete else None,
            "pricing_model_key": price_key, "observed_models": observed_models,
            "price_model_matches": price_model_matches, "reference_prices": prices,
            "estimated_total_cost_usd": cost, "estimated_cost_per_1000_decisions_usd": cost_per_1000,
            "by_type": by_type, "limits": limits, "local_reference_limits_met": eligible,
            "qualification_note": "Local measurements only; evaluator hardware measurements determine admission."}


def project_cost(rows,config,P):
    if P is None:return None
    if isinstance(P,bool) or not isinstance(P,(int,float)) or not math.isfinite(P) or P<=0:raise ValueError('P must be positive')
    prices=config['pricing']['models'][config['pricing']['model_key']]
    known=all(isinstance(prices.get(k),(int,float)) and prices[k]>=0 for k in ('input_per_million','output_per_million'))
    result={'P':P,'by_basis':{},'assumption':'APC: additive input overhead; no APC: proportional input; output unchanged; long tail added'}
    cap=config['limits']['cost_per_1000_decisions_usd']*(1-config['pricing']['margin'])
    tail=config['bench'].get('long_tail',{'count':23,'total_decisions':1624})
    for basis in ('engine','each_once','submitted'):
        values=[];outputs=[];valid=bool(rows) and known and all(r.get('usage_complete') for r in rows)
        modes=set()
        if valid:
            for r in rows:
                usage=r.get('usage',{}).get('alternatives',{}).get(basis);f=r.get('fast_prompt_tokens')
                if not usage or not isinstance(f,(int,float)) or f<=0:valid=False;break
                apc=r.get('prefix_caching')
                if basis=='engine' and apc is None:valid=False;break
                additive=basis=='engine' and apc is True
                values.append(max(0,P+usage['input_tokens']-f) if additive else usage['input_tokens']*P/f)
                outputs.append(usage['output_tokens']);modes.add('additive' if additive else 'ratio')
        cost=None
        if valid:
            regular=(statistics.mean(values)*prices['input_per_million']+statistics.mean(outputs)*prices['output_per_million'])/1e6
            long_input=config['limits']['max_prompt_tokens'];probe=1 if config['model']['backend']=='vllm' else 0
            long_cost=(long_input*prices['input_per_million']+probe*prices['output_per_million'])/1e6
            cost=((tail['total_decisions']-tail['count'])*regular+tail['count']*long_cost)/tail['total_decisions']*1000
        result['by_basis'][basis]={'cost_per_1000':cost,'method':sorted(modes),
            'gate':'UNKNOWN' if cost is None else 'PASS' if cost<=.85*cap else 'WARN' if cost<=cap else 'FAIL'}
    return result


def margin_threshold_for_share(margins,target):
    if not margins or not 0<target<1 or any(not math.isfinite(v) or v<0 for v in margins):
        raise ValueError('Need finite nonnegative margins and target in (0,1)')
    ordered=sorted(margins); k=math.ceil(target*len(ordered)); lower=ordered[k-1]
    higher=next((v for v in ordered[k:] if v>lower),None)
    # JSON-safe equivalent of infinity when all remaining margins are tied.
    threshold=lower+(higher-lower)/2 if higher is not None else math.nextafter(lower,math.inf)
    if not math.isfinite(threshold): raise ValueError("No finite threshold can include the boundary tie")
    return threshold,sum(v<threshold for v in margins)/len(margins)


def binomial_ci(successes,n):
    if not n: return None
    z=1.959963984540054; p=successes/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,center-radius),min(1,center+radius)]


def margin_auroc(rows):
    positive=[-r['fast_margin'] for r in rows if r.get('correct') is False]
    negative=[-r['fast_margin'] for r in rows if r.get('correct') is True]
    if not positive or not negative: return None
    return sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))


def calibrate_margin(rows,target,health):
    if health.get('mode')!='fast_only': raise ValueError('Margin calibration requires fast_only service')
    if not rows or any(r.get('request_order')!='criteria' or not r['ok'] or not r['usage_complete'] or r.get('path')!='fast'
            or isinstance(r.get('fast_margin'),bool) or not isinstance(r.get('fast_margin'),(int,float))
            or not math.isfinite(r['fast_margin']) for r in rows):
        raise ValueError('Every calibration task must succeed with complete margin and usage')
    kinds={r.get('margin_kind') for r in rows}
    if len(kinds)!=1 or next(iter(kinds)) not in ('logratio','probdiff'):
        raise ValueError('Inconsistent margin kind')
    threshold,achieved=margin_threshold_for_share([r['fast_margin'] for r in rows],target)
    def stats(subset):
        n=len(subset); selected=sum(r['fast_margin']<threshold for r in subset)
        result={'n':n,'selected':selected,'share':selected/n if n else None,
                'binomial_ci95':binomial_ci(selected,n),'auroc_fast_error':margin_auroc(subset)}
        for side in ('below','above'):
            labeled=[r for r in subset if r.get('correct') is not None and (r['fast_margin']<threshold)==(side=='below')]
            result[side+'_fast_accuracy']=sum(r['correct'] for r in labeled)/len(labeled) if labeled else None
        return result
    identity=health.get('threshold_identity',{})
    if any(k not in identity for k in ('model_key','quant','adapter','prompt_sha','backend','margin_kind','max_new_tokens','max_prompt_tokens','request_contract')):
        raise ValueError('Health lacks threshold identity')
    return dict(identity,request_contract='typesafe.criteria.v1',margin_kind=next(iter(kinds)),target=target,threshold=threshold,
        achieved=achieved,n=len(rows),placeholder=False,overall=stats(rows),
        by_type={k:stats([r for r in rows if r['type']==k]) for k in sorted({r['type'] for r in rows})},
        by_tier={str(k):stats([r for r in rows if r['tier']==k]) for k in sorted({r['tier'] for r in rows},key=str)})


def compare_order_accuracy(criteria_rows,labels_rows):
    left={r['id']:r for r in criteria_rows};right={r['id']:r for r in labels_rows}
    ids=[i for i in left if i in right and left[i]['type']=='choice' and left[i].get('expected') is not None]
    if not ids or any(not left[i]['ok'] or not right[i]['ok'] for i in ids):
        return {'count':len(ids),'criteria_accuracy':None,'labels_accuracy':None,'labels_minus_criteria':None}
    a=sum(left[i]['correct'] for i in ids)/len(ids);b=sum(right[i]['correct'] for i in ids)/len(ids)
    return {'count':len(ids),'criteria_accuracy':a,'labels_accuracy':b,'labels_minus_criteria':b-a}


def calibrate_temperature(rows):
    # Only replay raw scores from the official request order; never calibrate
    # already floored/temperature-scaled probabilities.
    if not rows or any(not r['ok'] or r.get('request_order')!='criteria' or not isinstance(r.get('raw_label_scores'),dict) or r.get('expected') is None for r in rows):
        raise ValueError('Temperature fit needs successful labeled criteria-order raw scores')
    result={}
    grid=[2**(i/8) for i in range(-32,41)]
    for kind in TYPES:
        subset=[r for r in rows if r['type']==kind]
        if not subset: result[kind]={'count':0,'fitted':False};continue
        if kind=='noul':
            correct=sum(max(r['raw_label_scores'],key=r['raw_label_scores'].get)==r['expected'] for r in subset)
            result[kind]={'count':len(subset),'p_cal':max(.5,min(1-1e-6,correct/len(subset))),'fitted':True};continue
        def loss(t):
            total=0
            for r in subset:
                scores=r['raw_label_scores'];top=max(scores.values())
                total+=math.log(sum(math.exp((v-top)/t) for v in scores.values()))-(scores[r['expected']]-top)/t
            return total/len(subset)
        t=min(grid,key=loss);result[kind]={'count':len(subset),'temperature':t,'nll':loss(t),'fitted':True}
    return {'request_contract':'typesafe.criteria.v1','by_type':result,'status':'candidate; held-out validation required'}


def fit_calibration(fast_rows, routed_rows, model_key):
    import copy
    if not fast_rows or not routed_rows:raise ValueError('Both fast and routed rows are required')
    for rows in (fast_rows,routed_rows):
        if len({r.get('id') for r in rows})!=len(rows):raise ValueError('Duplicate calibration IDs')
        for r in rows:
            identity=r.get('threshold_identity') or {}
            if identity.get('model_key')!=model_key:raise ValueError('Calibration model identity mismatch')
            if not r.get('ok') or r.get('request_order')!='criteria' or r.get('expected') is None:
                raise ValueError('Calibration requires successful labeled criteria-order rows')
    keys=('model_key','quant','adapter','backend','prompt_sha','margin_kind','max_prompt_tokens','request_contract')
    identity=fast_rows[0]['threshold_identity']
    if any(any(r['threshold_identity'].get(k)!=identity.get(k) for k in keys) for r in fast_rows+routed_rows):
        raise ValueError('Mixed calibration identities')
    if any(r.get('path')!='fast' or r.get('calibration_block') not in ('fast',None) for r in fast_rows):
        raise ValueError('fast-run must contain only fast decisions')
    def prepared(rows,field):
        result=[]
        for r in rows:
            scores=r.get(field)
            if scores is None and field=='fast_scores':scores=r.get('raw_label_scores')
            if not isinstance(scores,dict) or r['expected'] not in scores or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in scores.values()):
                raise ValueError('Missing or invalid pre-temperature label scores')
            result.append(dict(r,raw_label_scores=scores))
        return result
    sources={'fast':prepared(fast_rows,'fast_scores'),
             'slow':prepared([r for r in routed_rows if r.get('calibration_block')=='slow'],'slow_scores'),
             'fallback':prepared([r for r in routed_rows if r.get('calibration_block')=='fallback'],'fast_scores')}
    blocks={}
    for name,rows in sources.items():
        fit=calibrate_temperature(rows)['by_type'] if rows else {}
        block={'placeholder':False,'status':'candidate; held-out validation required',
               'temperature_by_kind':{'choice':1.,'score':1.,'noul':'one_bin'},'noul_calibration':{'p_cal':.5},'by_type':{}}
        for kind in TYPES:
            item=fit.get(kind,{'count':0,'fitted':False});n=item['count']
            meta=dict(item,status='insufficient' if n<20 else 'candidate')
            if name=='fallback' and n<20:
                source=blocks['fast'];meta['inherited_from']='fast'
                if kind=='noul':block['noul_calibration']=copy.deepcopy(source['noul_calibration'])
                else:block['temperature_by_kind'][kind]=source['temperature_by_kind'][kind]
            elif item.get('fitted'):
                if kind=='noul':block['noul_calibration']['p_cal']=item['p_cal']
                else:block['temperature_by_kind'][kind]=item['temperature']
            if n<20:block['placeholder']=True
            block['by_type'][kind]=meta
        blocks[name]=block
    return {'status':'candidate; no production overwrite; validate held-out data',
            'models':{model_key:dict(blocks,placeholder=any(b['placeholder'] for b in blocks.values()),
                                   fitted_for=identity)}}


def read_run_rows(prefix):
    if __package__:from .core import local_output
    else:from core import local_output
    path=local_output(prefix)
    if path.suffix!='.jsonl':path=path.with_name(path.name+'.jsonl')
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_candidate(path,value):
    if __package__:from .core import local_output
    else:from core import local_output
    target=local_output(path);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as out:json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False)
    return target


def apply_margin_candidate(config, report, tier=None):
    import copy
    if __package__:from .core import apply_profile,threshold_identity,apply_budget_tier
    else:from core import apply_profile,threshold_identity,apply_budget_tier
    result=copy.deepcopy(config);key=report.get('model_key')
    result['model']['model_key']=key;apply_profile(result);apply_budget_tier(result,tier)
    identity=threshold_identity(result)
    keys=('model_key','quant','adapter','backend','prompt_sha','margin_kind','max_prompt_tokens','request_contract')
    if report.get('placeholder',True) or any(report.get(k)!=identity[k] for k in keys):raise ValueError('Margin candidate identity mismatch')
    value=report['threshold']
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Invalid threshold')
    fit={k:report[k] for k in keys};fit.update(placeholder=False,max_new_tokens=identity['max_new_tokens'],reasoning_words=identity['reasoning_words'])
    target=result['profiles'][key]['routing']
    target.update(margin_threshold=value,margin_threshold_fitted_for=fit)
    result['routing'].update(margin_threshold=value,margin_threshold_fitted_for=fit)
    return result


def get_health(endpoint,timeout,path="/health"):
    parts=urllib.parse.urlsplit(endpoint)
    url=urllib.parse.urlunsplit((parts.scheme,parts.netloc,path,'',''))
    with urllib.request.urlopen(url,timeout=timeout) as response:
        return json.load(response)


def output_paths(prefix, calibration_only=False):
    candidate = Path(prefix).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(ROOT) or candidate == ROOT:
        raise ValueError("output prefix must stay inside submission_v2")
    if calibration_only:
        target=candidate.with_name(candidate.name+'.margin_calibration.json')
        if target.exists(): raise FileExistsError('Calibration output exists')
        target.parent.mkdir(parents=True,exist_ok=True)
        return target
    rows_path = candidate.with_name(candidate.name + ".jsonl")
    summary_path = candidate.with_name(candidate.name + ".summary.json")
    if rows_path.exists() or summary_path.exists():
        raise FileExistsError("output already exists; choose a new prefix")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    return rows_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--public", type=Path)
    source.add_argument("--data", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--endpoint", help="Runtime HTTP endpoint; never written to output")
    parser.add_argument("--output-prefix")
    parser.add_argument("--price-model", help="Key in config.pricing.models")
    parser.add_argument('--original-types',action='store_true')
    parser.add_argument('--concurrency',type=int,nargs='+')
    parser.add_argument('--calibrate-margin',action='store_true')
    parser.add_argument('--engine-stats',action='store_true')
    parser.add_argument('--compare-label-order',action='store_true')
    parser.add_argument('--target-slow-share',type=float)
    parser.add_argument('--P',type=float)
    parser.add_argument('--fit-calibration',action='store_true')
    parser.add_argument('--fast-run',type=Path)
    parser.add_argument('--routed-run',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--model-key')
    parser.add_argument('--apply-margin',type=Path)
    parser.add_argument('--tier',choices=('normal','hard','long'))
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if args.fit_calibration:
        if not all((args.fast_run,args.routed_run,args.output,args.model_key)):parser.error('Fit needs fast-run, routed-run, output and model-key')
        write_candidate(args.output,fit_calibration(read_run_rows(args.fast_run),read_run_rows(args.routed_run),args.model_key))
        return 0
    if args.apply_margin:
        if not args.output:parser.error('Apply margin needs a new output config')
        write_candidate(args.output,apply_margin_candidate(config,json.loads(args.apply_margin.read_text(encoding='utf-8')),args.tier))
        return 0
    if not args.endpoint or not (args.public or args.data):parser.error('Benchmark needs endpoint and public or data')
    if args.P is not None:config['bench']['projection_prompt_tokens']=args.P
    if args.original_types: config['bench']['original_types']=True
    if args.price_model:
        config["pricing"]["model_key"] = args.price_model
    if args.public:
        index = args.index or Path(config["bench"]["index"])
        if not index.is_absolute():
            index = config_path.parent / index
        tasks = load_public_tasks(index, args.public, config["bench"]["expected_public_count"])
    else:
        if args.index:
            parser.error("--index requires --public")
        tasks = load_jsonl_tasks(args.data)
    levels=args.concurrency or config['bench']['concurrency']
    if any(n<1 for n in levels): parser.error('Concurrency must be positive')
    if args.calibrate_margin and config['bench'].get('original_types'):parser.error('Calibration requires the official request types')
    if args.calibrate_margin and levels!=[1]: parser.error('Margin calibration is serial')
    if args.calibrate_margin and args.engine_stats:parser.error('Use a separate serial run for engine reconciliation')
    endpoint = endpoint_url(args.endpoint)
    paths = output_paths(args.output_prefix or config["bench"]["output_prefix"],calibration_only=args.calibrate_margin)
    health=get_health(endpoint,config['bench']['timeout_seconds'])
    if __package__: from .core import apply_profile
    else: from core import apply_profile
    config['model']['model_key']=health['model_key']
    apply_profile(config)
    if args.tier:
        if __package__:from .core import apply_budget_tier
        else:from core import apply_budget_tier
        apply_budget_tier(config,args.tier)
    if args.price_model: config['pricing']['model_key']=args.price_model
    if args.calibrate_margin:
        if health.get('mode')!='fast_only': parser.error('Requires fast_only service')
        target=args.target_slow_share if args.target_slow_share is not None else config['bench']['target_slow_share']
        calibration_path=paths
        # Existing-file check precedes inference; exclusive creation prevents races.
        rows=run_benchmark(tasks,endpoint,config)
        report=calibrate_margin(rows,target,health)
        report['temperature_fit']=calibrate_temperature(rows)
        if args.compare_label_order:
            import copy
            other=copy.deepcopy(config);other['bench']['label_order']=True
            report['order_accuracy']=compare_order_accuracy(rows,run_benchmark(tasks,endpoint,other))
        with calibration_path.open('x',encoding='utf-8') as output:
            json.dump(report,output,ensure_ascii=False,indent=2,allow_nan=False)
        print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        return 0
    stats_path=paths[0].with_name(paths[0].stem+'.engine_stats.json')
    if args.engine_stats and stats_path.exists():raise FileExistsError('Engine snapshot exists')
    # Claim both paths before requests, using exclusive creation for preservation.
    with paths[0].open("x", encoding="utf-8") as writer, paths[1].open("x", encoding="utf-8") as summary_writer:
        summaries={}
        snapshots={"health":health,"by_concurrency":{}}
        stats_path=paths[0].with_name(paths[0].stem+".engine_stats.json")
        if args.engine_stats and stats_path.exists(): raise FileExistsError("Engine snapshot exists")
        for n in dict.fromkeys(levels):
            if args.engine_stats: before=get_health(endpoint,config['bench']['timeout_seconds'],'/engine_stats')
            rows=run_benchmark(tasks,endpoint,config,concurrency=n)
            if args.engine_stats: snapshots['by_concurrency'][str(n)]={'before':before,'after':get_health(endpoint,config['bench']['timeout_seconds'],'/engine_stats')}
            for row in rows:
                row['concurrency']=n
                row['prefix_caching']=health.get('prefix_caching')
                writer.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            writer.flush()
            summaries[str(n)]=summarize(rows,config,expected_count=len(tasks))
            if args.compare_label_order:
                import copy
                other=copy.deepcopy(config);other['bench']['label_order']=True
                summaries[str(n)]['order_accuracy']=compare_order_accuracy(rows,run_benchmark(tasks,endpoint,other,concurrency=n))
        if args.engine_stats:
            with stats_path.open('x',encoding='utf-8') as output: json.dump(snapshots,output,indent=2,allow_nan=False)
        summary=dict(summaries[str(levels[0])],by_concurrency=summaries)
        summary_writer.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if all(v["complete"] for v in summary["by_concurrency"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
