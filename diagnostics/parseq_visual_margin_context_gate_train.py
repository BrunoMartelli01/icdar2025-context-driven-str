"""Offline TRAIN margin-gated reranking; no model imports or inference."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path

from diagnostics.caption_routing_train import seal_write
from diagnostics.contextual_ocr_correction import verify
from diagnostics.lavis_captioner_candidates import digest, sha
from diagnostics.parseq_context_reranker_train import dedupe_candidates, per_crop_oracle


def read(path):
    return json.loads(Path(path).read_text())


def key(row):
    return str(row['sample_id']), row['candidate_id']


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite cached score')
    return value


def attach_visual(row, ar):
    """Observable-only exact lookup. Never substitute rank or casefold probability."""
    if not ar or len({r['text'].casefold() for r in ar}) != len(ar):
        raise ValueError('Empty or nonunique AR beam')
    values = [finite(x['logprob']) for x in ar]
    if any(b > a for a,b in zip(values, values[1:])):
        raise ValueError('Unsorted AR beam')
    cands = row['candidates']
    if cands[0]['text'] != row['v1_text']:
        raise ValueError('KEEP is not exact V1')
    matches = [(i,x) for i,x in enumerate(ar, 1) if x['text'] == row['v1_text']]
    reason = None
    if not row['v1_text'].strip(): reason = 'empty_v1'
    elif not matches:
        reason = ('case_only_reference_mismatch' if any(x['text'].casefold() == row['v1_text'].casefold() for x in ar)
                  else 'reference_absent_from_saved_beam')
    reference = finite(matches[0][1]['logprob']) if matches else None
    deltas = [0.0]
    for c in cands[1:]:
        rank = c['ar_rank']
        if not 1 <= rank <= 10 or ar[rank-1]['text'] != c['text'] or finite(c['ar_logprob']) != values[rank-1]:
            raise ValueError('Candidate visual score/rank identity mismatch')
        deltas.append(values[rank-1]-reference if reference is not None else None)
    return {'sample_id': row['sample_id'], 'candidate_id': row['candidate_id'],
            'v1_text': row['v1_text'], 'candidates': cands, 'scores': row['scores'],
            'delta_visual': deltas, 'reference_logprob': reference,
            'reference_rank': matches[0][0] if matches else None, 'force_keep_reason': reason}


def select(row, condition, m, beta, tau, rule='combined'):
    if row['force_keep_reason']:
        return 0
    if rule not in ('combined', 'visual_only', 'visual_margin_gate_only'):
        raise ValueError('Unknown selector')
    best = 0; best_gain = 0.0
    for i,dv in enumerate(row['delta_visual'][1:], 1):
        if rule != 'visual_only' and dv < -m:
            continue
        if rule == 'visual_only': gain = dv
        else:
            q = row['scores'][condition]
            dc = q[i]['qwen_mean_logprob'] - q[0]['qwen_mean_logprob']
            gain = dc if rule == 'visual_margin_gate_only' else dv + beta*dc
        if gain > best_gain:
            best = i; best_gain = gain
    return best if best_gain > tau else 0


def evaluate_choices(labels, rows, choices, cfg):
    exact = recovered = damaged = 0
    for lab in labels:
        row = rows[key(lab)]
        ok = row['candidates'][choices[key(lab)]]['text'].casefold() == lab['gt_text'].casefold()
        exact += ok; recovered += ok and not lab['v1_exact']; damaged += lab['v1_exact'] and not ok
    changed = sum(i != 0 for i in choices.values())
    if exact != cfg['baseline_exact'] + recovered - damaged:
        raise RuntimeError('Recovery accounting mismatch')
    return {'exact': exact, 'eligible_gt_rows': len(labels), 'gain_vs_502': exact-cfg['baseline_exact'],
            'recovered': recovered, 'damaged': damaged, 'changed_crops': changed,
            'net': recovered-damaged, 'rate_eligible': exact/len(labels),
            'fixed_spatial_rate_all849': exact/cfg['total_gt_words']}


def margin_diagnostic(labels, rows, margins):
    result = []
    for lab in labels:
        if lab['v1_exact']: continue
        r = rows[key(lab)]
        hits = [i for i,x in enumerate(r['candidates']) if x['text'].casefold() == lab['gt_text'].casefold()]
        if not hits: continue
        i = hits[0]
        result.append({**lab, 'gt_candidate_index': i, 'gt_ar_rank': r['candidates'][i]['ar_rank'],
                       'reference_rank': r['reference_rank'], 'force_keep_reason': r['force_keep_reason'],
                       'delta_visual_gt': r['delta_visual'][i],
                       'delta_context_gt': {cond: s[i]['qwen_mean_logprob']-s[0]['qwen_mean_logprob']
                                            for cond,s in r['scores'].items()}})
    counts = [{'m': m, 'recoverable_gt_rows_inside_margin': sum(
        x['force_keep_reason'] is None and x['delta_visual_gt'] >= -m for x in result)} for m in margins]
    return {'recoverable_error_rows': len(result), 'missing_reference_rows': sum(x['force_keep_reason'] is not None for x in result),
            'within_margin': counts, 'per_recoverable_gt': result}


def load_sources(a, cfg):
    paths = {'coverage': a.coverage, **{n: a.score_root/n for n in cfg['source_hashes'] if n != 'coverage'}}
    for n,f in paths.items(): verify(f, cfg['source_hashes'][n])
    cov = read(a.coverage); protocol = read(a.score_root/'protocol.json')
    if cov['scope']['split'] != 'TRAIN' or cov['scope']['test_loaded'] or protocol['test_loaded'] or protocol['training']:
        raise RuntimeError('TRAIN-only frozen sources required')
    labels = read(a.score_root/'evaluation-labels.json')['rows']
    inputs = read(a.score_root/'inputs.json')['samples']
    by = {}; cov_labels = {}
    for lab in cov['details']:
        if not 100 <= int(lab['sample_id']) <= 328: raise RuntimeError('Non-TRAIN sample')
        k = key(lab); lk = (*k, lab['gt_index'])
        if lk in cov_labels: raise RuntimeError('Duplicate GT row')
        cov_labels[lk] = lab
        if k in by and (by[k]['ar_candidates'] != lab['ar_candidates'] or by[k]['v1_text'] != lab['v1_text']):
            raise RuntimeError('Shared crop has inconsistent evidence')
        by[k] = lab
    seen_labels = set()
    for lab in labels:
        lk = (*key(lab),lab['gt_index'])
        if lk in seen_labels: raise RuntimeError('Duplicate evaluation label')
        seen_labels.add(lk)
        if any(lab[f] != cov_labels[lk][f] for f in ('gt_text','v1_text','v1_exact')):
            raise RuntimeError('Frozen label disagreement')
        if lab['v1_exact'] != (lab['gt_text'].casefold() == lab['v1_text'].casefold()):
            raise RuntimeError('Baseline exact mismatch')
    if seen_labels != set(cov_labels) or len(labels) != 745 or sum(x['v1_exact'] for x in labels) != 502:
        raise RuntimeError('Population/control mismatch')
    rows = {}; hashes = {str(f.resolve()): sha(f) for f in paths.values()}
    for s in inputs:
        k = key(s)
        if k in rows or k not in by: raise RuntimeError('Missing/duplicate crop')
        expected = dedupe_candidates(by[k]['v1_text'],by[k]['ar_candidates'])
        if s['candidates'] != expected or s['v1_text'] != by[k]['v1_text']:
            raise RuntimeError('Prepared candidate set drift')
        f = a.score_root/'scores'/f'{int(s["sample_id"]):04d}-{s["candidate_id"].split(":")[-1]}.json'
        row = read(f)
        if key(row) != k or row['v1_text'] != s['v1_text']: raise RuntimeError('Shard identity drift')
        empty = not s['v1_text'].strip()
        if row['candidates'] != (expected[:1] if empty else expected): raise RuntimeError('Score/candidate mismatch')
        if set(row['scores']) != set(cfg['conditions']): raise RuntimeError('Score condition mismatch')
        for scores in row['scores'].values():
            if len(scores) != len(row['candidates']): raise RuntimeError('Score length mismatch')
            for i,(q,c) in enumerate(zip(scores,row['candidates'])):
                if q['candidate_index'] != i or q['text'] != c['text']: raise RuntimeError('Score alignment mismatch')
                mean = finite(q['qwen_mean_logprob']); total = finite(q['qwen_sum_logprob']); n = q['qwen_token_count']
                if n <= 0 and not empty: raise RuntimeError('Unscored nonempty candidate')
                if n and not math.isclose(mean*n,total,rel_tol=1e-5,abs_tol=1e-5):
                    raise RuntimeError('Qwen mean/sum mismatch')
        rows[k] = attach_visual(row, by[k]['ar_candidates']); hashes[str(f.resolve())] = sha(f)
    if set(rows) != set(by) or len(rows) != cfg['unique_crops']: raise RuntimeError('Incomplete crop scores')
    return rows,labels,hashes


def best_key(row):
    return (-row['exact'], row['damaged'], row['changed_crops'], row['m'], row['beta'], -row['tau'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','coverage','score-root','output-dir'): p.add_argument('--'+name,required=True,type=Path)
    a = p.parse_args(); cfg = read(a.config)
    if (cfg['mode'] != 'APPROXIMATE' or cfg['split'] != 'TRAIN' or cfg['topk'] != 10
            or (cfg['total_gt_words'],cfg['eligible_gt_rows'],cfg['unique_crops'],cfg['baseline_exact']) != (849,745,700,502)):
        raise RuntimeError('Only frozen TRAIN diagnostic supported')
    combinations = list(itertools.product(cfg['conditions'],cfg['m'],cfg['beta'],cfg['tau']))
    if len(combinations) != 128: raise RuntimeError('Expected predeclared 128 combinations')
    rows,labels,hashes = load_sources(a,cfg)
    dependencies = [Path(__file__),a.config,Path('diagnostics/parseq_context_reranker_train.py')]
    protocol = {'config':cfg,'source_sha256':hashes,'code_sha256':{str(f.resolve()):sha(f) for f in dependencies},
                'training':False,'inference':False,'test_loaded':False,'gt_in_selector':False}
    contract = digest(protocol)
    # Freeze all inputs and parameter combinations BEFORE computing diagnostic or grid metrics.
    seal_write(a.output_dir/'protocol.json',{**protocol,'contract_sha256':contract})
    availability = Counter(r['force_keep_reason'] or ('exact_reference_rank'+str(r['reference_rank'])) for r in rows.values())
    margins = margin_diagnostic(labels,rows,cfg['diagnostic_margins'])
    print('REFERENCE AVAILABILITY',dict(availability),flush=True)
    print('RECOVERABLE MARGINS',json.dumps(margins['within_margin']),flush=True)
    keep = {k:0 for k in rows}; baseline = evaluate_choices(labels,rows,keep,cfg)
    controls = [{'name':'KEEP',**baseline}]
    visual = {k:select(r,None,None,None,0,'visual_only') for k,r in rows.items()}
    controls.append({'name':'visual_only',**evaluate_choices(labels,rows,visual,cfg)})
    for cond,m in itertools.product(cfg['conditions'],cfg['m']):
        chosen = {k:select(r,cond,m,None,0,'visual_margin_gate_only') for k,r in rows.items()}
        controls.append({'name':'visual_margin_gate_only','condition':cond,'m':m,
                         **evaluate_choices(labels,rows,chosen,cfg)})
    grid = []; selections = {}
    for n,(cond,m,beta,tau) in enumerate(combinations,1):
        chosen = {k:select(r,cond,m,beta,tau) for k,r in rows.items()}
        name = f'{cond}/m={m}/beta={beta}/tau={tau}'
        grid.append({'id':name,'condition':cond,'m':m,'beta':beta,'tau':tau,
                     **evaluate_choices(labels,rows,chosen,cfg)})
        selections[name] = [chosen[k] for k in rows]
        if n%32 == 0: print(f'[CPU grid] {n}/128',flush=True)
    best = {cond:min([r for r in grid if r['condition']==cond],key=best_key) for cond in cfg['conditions']}
    interest = cfg['interest_criterion']
    passing = [r for r in grid if r['recovered']>=interest['min_recovered'] and r['damaged']<=interest['max_damaged'] and r['net']>=interest['min_net']]
    oracle = per_crop_oracle(labels,rows)
    if oracle != 532: raise RuntimeError('Frozen per-crop oracle mismatch')
    report = {'contract_sha256':contract,'mode':cfg['mode'],'classification':cfg['classification'],
              'claim_label':cfg['claim_label'],'scope':{'split':'TRAIN','total_gt_words':849,'eligible_gt_rows':745,'unique_crops':700},
              'baseline':baseline,'oracle_per_crop_exact':oracle,'reference_availability':dict(availability),
              'margin_diagnostic':margins,'controls':controls,'grid':grid,'best_train_diagnostic_by_condition':best,
              'interesting_configurations':passing,'decision':'GO_TRAIN_DIAGNOSTIC_ONLY' if passing else 'NO_GO',
              'pipeline_promoted':False,'table5_accuracy':None,
              'selection_crop_order':[list(k) for k in rows],'selected_candidate_indices':selections,
              'observables':list(rows.values()),'evaluation_labels':labels,
              'warning':'GT only in evaluation. Best entries are TRAIN-selected, not validated on held-out images. 104 unassociated GT remain failures in all849 rate. No new score/model inference.'}
    seal_write(a.output_dir/'results.json',report)
    print('BEST',json.dumps(best,indent=2)); print('DECISION',report['decision'],'passing',len(passing))
    print('JSON',a.output_dir/'results.json','SHA256',sha(a.output_dir/'results.json'))


if __name__=='__main__': main()
