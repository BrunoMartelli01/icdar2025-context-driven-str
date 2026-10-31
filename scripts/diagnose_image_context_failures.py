"""Post-hoc context-specificity, FP-removal and named-case diagnostics; no policy fitting."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from recognition.image_context_correction import apply_choices,filter_readings,overlap_fraction,select_subset
from scripts.run_contextual_boundary_selection import bootstrap
from scripts.run_image_context_correction import operation_label
from scripts.run_text_reconstruction_tests import annotations,read,sha,write


def diagnose(root):
    root=Path(root);c=read(root/'resolved_config.json');data=read(root/'pool.json');result=read(root/'results.json')
    scores=read(root/'scores.json');selection=read(root/'selection_frozen.json');fscores=read(root/'filter_scores.json');frozen=read(root/'filter_selection_frozen.json')
    out=dict(claim_label='approximate reproduction / post-hoc diagnosis',results_sha256=sha(root/'results.json'),
        diagnostic_code_sha256=sha(Path(__file__)),routing=result['routing'],cohorts={})
    names={'krispykreme','bananaleaf','holidayinn','multigrainflakes','authenticrakery','sanfrancisco','touristvehicle'}
    for phase in ('evaluation','test'):
        a=result['results'][phase]['boundary_wrong_context_correct_cap2'];b=result['results'][phase]['boundary_inventory_correct_cap2']
        context_comparison=bootstrap(a,b,c['bootstrap_draws'],c['seed']);coverage=Counter();deletions={};examples=[]
        for policy,record in frozen['selection'].items():
            choice=record['choices']['cap2'];counts=Counter()
            for s in data[phase]:
                sid=s['sample_id'];words=annotations(s);e=fscores[phase][sid]
                _,removed=filter_readings(s['blocks'],e['exact'],choice['threshold'],policy,e['presence'],choice['presence_threshold'])
                for i in removed:
                    proxy=any(overlap_fraction(s['blocks'][i]['bbox'],w['bbox'])>=c['alignment']['presence_gt_coverage'] for w in words)
                    counts['removed_with_care_coverage' if proxy else 'removed_without_care_coverage']+=1
                    counts['removed_empty']+=int(not s['blocks'][i]['text'])
            deletions[policy]=dict(counts)
        for s in data[phase]:
            words=annotations(s);groups=defaultdict(list)
            for index in select_subset(s['pool'],'boundary'):
                groups[tuple(s['pool'][index]['parents'])].append(index)
            for parents,indices in groups.items():
                labels=[operation_label(s,s['pool'][i],words,c) for i in indices]
                if any(r['label']==1 for r in labels):
                    coverage['beneficial_parent_groups']+=1
                    coverage['with_local_ocr']+=int(any(s['contexts'][parents[0]]))
                    others=[b['text'] for i,b in enumerate(s['blocks']) if i not in parents and b['text'].strip()]
                    coverage['with_any_other_ocr']+=int(bool(others))
                    expected=Counter(w['text'].casefold() for w in words)
                    observed=Counter(text.casefold() for text in others)
                    coverage['other_observations']+=len(others)
                    coverage['other_exact_gt_occurrences']+=sum((observed&expected).values())
            if not any(b['text'].casefold() in names for b in s['blocks']):continue
            record=dict(sample_id=s['sample_id'],original_transcript=[b['text'] for b in s['blocks']],models={})
            for name in ('boundary_word','boundary_inventory','boundary_global_context','expanded_global'):
                spec=selection['models'][name];indices=select_subset(s['pool'],spec['spec']['subset']);values=scores[name][phase][s['sample_id']]
                texts,chosen=apply_choices(s['blocks'],[s['pool'][i] for i in indices],values,spec['choices']['correct_cap2']['threshold'])
                record['models'][name]=dict(transcript=texts,selected=[dict(operation=s['pool'][indices[i]],score=values[i],
                    effect=operation_label(s,s['pool'][indices[i]],words,c)) for i in chosen])
            examples.append(record)
        out['cohorts'][phase]=dict(matched_inventory_versus_wrong_context=context_comparison,
            beneficial_context_coverage=dict(coverage),filter_removal_annotation_proxies=deletions,named_examples=examples)
    write(root/'failure_diagnosis.json',out)
    lines=['# Context specificity and failure diagnostics','','Approximate reproduction; frozen policies and no TEST-driven threshold changes.','',
        'The inventory-versus-wrong-image contrast has identical feature dimensions, candidate pools, training labels and HGB capacity.','']
    for phase,d in out['cohorts'].items():
        r=d['matched_inventory_versus_wrong_context'];lines.append(f"- {phase}: correct-image inventory minus wrong-image inventory F1 {r['delta_f1']*100:+.4f}points; conditional95%CI {[round(x*100,4) for x in r['conditional_95_interval']]}.")
        lines.append('  Beneficial-group context availability: '+json.dumps(d['beneficial_context_coverage'])+'.')
        lines.append('  Filter-removal proxies: '+json.dumps(d['filter_removal_annotation_proxies'])+'.')
    lines+=['','Care coverage is not physical text-presence truth: low coverage can include ignored, partial or unannotated text. Exact neighbour matching is an annotation proxy, not proof of semantic usefulness.',
        '', 'Named examples are selected post-hoc from previously discussed failures and cannot establish average generalization. Their complete original OCR context and selected decisions are preserved in failure_diagnosis.json.',
        '', 'No DeepSolo invocation occurs in this correction-only experiment: TEST0/300images,0.00%; no online routing latency is measured.']
    (root/'failure_diagnosis.md').write_text('\n'.join(lines)+'\n');return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();out=diagnose(args.run)
    print(json.dumps(dict(status='SUCCESS',results_sha256=out['results_sha256'])))


if __name__=='__main__':main()
