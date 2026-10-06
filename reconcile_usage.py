"""提交前用量对账：只读 bench 输出与引擎计数快照，不发推理请求，不读请求内容。

python -B exp/submission_v2/reconcile_usage.py --run runs/r34_on [--run-off runs/r34_off] [--tol 0.001]
退出码：0 通过；1 等式不成立；2 前置条件不满足（数据不全、计数器缺失）。
"""
import argparse, json, sys, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COUNTERS = ('prompt_tokens', 'generation_tokens', 'prefix_cache_queries', 'prefix_cache_hits',
            'num_preemptions', 'request_success')
FIELDS = ('input', 'output', 'total', 'cached', 'submitted', 'calls', 'generated')


def resolve(prefix):
    p = Path(prefix).expanduser()
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise SystemExit('prefix must stay inside submission_v2')
    return p


def load(prefix):
    p = resolve(prefix)
    rows = [json.loads(x) for x in p.with_name(p.name + '.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    stats = json.loads(p.with_name(p.name + '.engine_stats.json').read_text(encoding='utf-8'))
    rows = [r for r in rows if r.get('concurrency', 1) == 1]     # 只对串行档对账
    if not rows:
        sys.exit(2)
    return p, rows, stats


def delta(stats):
    level = stats['by_concurrency']['1']
    before, after = level['before']['counters'], level['after']['counters']
    missing = [k for k in COUNTERS if k not in before or k not in after]
    if missing:
        raise ValueError('engine counters missing: '+','.join(missing))
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for d in (before,after) for v in (d[k] for k in COUNTERS)):raise ValueError('Invalid counter')
    return {k: after[k] - before[k] for k in COUNTERS}


def row_usage(row):
    u = row.get('usage') or {}
    if not isinstance(u,dict):return False,{}
    d = u.get('input_tokens_details') or {}
    if not isinstance(d,dict):return False,{}
    v = dict(input=u.get('input_tokens'), output=u.get('output_tokens'), total=u.get('total_tokens'),
             cached=d.get('cached_tokens'), submitted=d.get('engine_submitted_tokens'),
             calls=d.get('engine_calls'), generated=d.get('engine_generated_tokens'))
    ok = (row.get('ok') and u.get('accounting') == 'engine' and not u.get('usage_estimated')
          and d.get('cached_tokens_known', True) and all(type(v[k]) is int and v[k] >= 0 for k in FIELDS))
    return bool(ok), v


def close(x, y, tol):
    return abs(x - y) <= tol * max(1, abs(y))


def reconcile_one(rows, stats, tol):
    fails, parsed = [], [row_usage(r) for r in rows]
    if not rows or len({r.get('id') for r in rows})!=len(rows) or any(r.get('concurrency',1)!=1 for r in rows) or not all(ok for ok, _ in parsed):
        return {'precondition': 'all rows ok, engine basis, exact usage'}, ['precondition']
    U = [v for _, v in parsed]
    S = {k: sum(v[k] for v in U) for k in FIELDS}
    if S['generated']!=S['output']:fails.append('generated/output identity broken')
    level=stats.get('by_concurrency',{}).get('1',{})
    source=level.get('after',{}).get('source')
    if source=='hf_forward_hook':
        try:
            before,after=level['before']['counters'],level['after']['counters']
            if after.get('uncounted_forwards',0)>before.get('uncounted_forwards',0):
                return {'precondition':'HF uncounted forwards'},['precondition']
            D={k:after[k]-before[k] for k in ('prefill_tokens','forward_tokens')}
            ds=[r['usage']['input_tokens_details'] for r in rows]
            prefill=sum(d['hf_prefill_tokens'] for d in ds)
            calls=sum(d['generate_nonempty_calls'] for d in ds)
        except (KeyError,TypeError): return {'precondition':'HF counters missing'},['precondition']
        if S['cached'] or any(v['input']!=v['submitted'] or v['total']!=v['input']+v['output'] for v in U): fails.append('HF row identity broken')
        if D['prefill_tokens']!=prefill: fails.append('HF initial prefill mismatch')
        if D['forward_tokens']!=S['input']+S['output']-calls: fails.append('HF forward identity broken')
        return dict(sums=S,engine_delta=D,counter_mode='hf_forward'),fails
    try: D = delta(stats)
    except (KeyError,ValueError,TypeError): return {'precondition':'engine counters missing'},['precondition']
    if any(v<0 for v in D.values()): return {'precondition':'counter reset'},['precondition']
    if D['num_preemptions']:
        return {'precondition':'preemption during run'},['precondition']
    if D['request_success'] != S['calls']:
        return {'precondition':'stray, warmup or missed engine calls'},['precondition']
    bad = [r['id'] for r, v in zip(rows, U) if v['input'] != v['submitted'] - v['cached'] or v['total'] != v['input'] + v['output']]
    if bad:
        fails.append('row identity broken: %s' % bad[:5])
    if close(S['submitted'], D['prompt_tokens'], tol):
        mode, prefill = 'full_prompt', D['prompt_tokens'] - D['prefix_cache_hits']
    elif close(S['submitted'] - S['cached'], D['prompt_tokens'], tol):
        mode, prefill = 'computed_only', D['prompt_tokens']
    else:
        mode, prefill = 'unknown', None
        fails.append('sum(engine_submitted) matches neither engine prompt-counter definition')
    if not close(S['cached'], D['prefix_cache_hits'], tol):
        fails.append('sum(cached_tokens) != delta prefix_cache_hits')
    if prefill is not None and not close(S['input'], prefill, tol):            # 等式一
        fails.append('E1: sum(usage.input_tokens) != engine prefill')
    if not close(S['output'], D['generation_tokens'], tol):
        fails.append('sum(usage.output_tokens) != delta generation_tokens')
    return dict(sums=S, engine_delta=D, counter_mode=mode, engine_prefill=prefill), fails


def reconcile_pair(on_rows, off_rows, off_stats, tol):
    fails = []
    if off_stats.get('health', {}).get('prefix_caching') is not False:
        fails.append('off run did not disable prefix caching')
    if not all(row_usage(r)[0] for r in on_rows+off_rows): return {},['precondition']
    if len({r['id'] for r in on_rows})!=len(on_rows) or len({r['id'] for r in off_rows})!=len(off_rows): return {},['precondition']
    on = {r['id']: row_usage(r)[1] for r in on_rows}
    off = {r['id']: row_usage(r)[1] for r in off_rows}
    if any(v['cached'] for v in off.values()):
        fails.append('off run reports cached tokens')
    matched = [i for i in on if i in off and on[i]['submitted'] == off[i]['submitted'] and on[i]['output'] == off[i]['output']]
    coverage = len(matched) / max(1, len(on),len(off))
    if coverage < 0.99:
        fails.append('decode drift: only %.1f%% rows identical between runs' % (100 * coverage))
    diff = sum(off[i]['input'] - on[i]['input'] for i in matched)
    cached = sum(on[i]['cached'] for i in matched)
    if not close(diff, cached, tol):                                             # 等式二
        fails.append('E2: sum(input_off) - sum(input_on) != sum(cached_on)')
    return dict(matched=len(matched), coverage=coverage, input_diff=diff, cached_on=cached), fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--run-off')
    ap.add_argument('--tol', type=float, default=0.001)
    a = ap.parse_args(argv)
    if not 0<=a.tol<=0.001: ap.error('tol must be within [0, 0.001]')
    try: path, rows, stats = load(a.run)
    except (OSError,ValueError,KeyError): return 2
    report, fails = reconcile_one(rows, stats, a.tol)
    if a.run_off:
        try: _, off_rows, off_stats = load(a.run_off)
        except (OSError,ValueError,KeyError):return 2
        off_report, off_fails = reconcile_one(off_rows, off_stats, a.tol)
        pair, pair_fails = reconcile_pair(rows, off_rows, off_stats, a.tol)
        report.update(off=off_report, pair=pair)
        fails += ['off: ' + f for f in off_fails] + pair_fails
    report.update(tolerance=a.tol, passed=not fails, submission_ready=bool(a.run_off) and not fails, failures=fails)
    with path.with_name(path.name + '.reconcile.json').open('x', encoding='utf-8') as out:   # 独占创建，不覆盖
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'passed': not fails, 'failures': fails}, ensure_ascii=False))
    return 0 if not fails else (2 if any('precondition' in f for f in fails) else 1)


if __name__ == '__main__':
    raise SystemExit(main())
