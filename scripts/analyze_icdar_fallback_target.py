"""Certify target F1 feasibility for the frozen ICDAR budget-router experts."""
import argparse
import json
from pathlib import Path
from recognition.fallback_feasibility import target_minimum,global_oracle,ranking_frontier,json_point
from scripts.icdar_budget_data import read,sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',required=True,choices=['approximate']);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--arm',required=True);p.add_argument('--target-f1',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent=read(a.run_root/'supplemental_audit.json')
    if parent['status']!='PASS' or parent['results_sha256']!=sha(a.run_root/'results.json'):raise ValueError('Unqualified frozen source results')
    source=a.run_root/'test_scores'/(a.dataset+'__'+a.arm+'.json');d=read(source);b=d['branches'];eligible=[s for s in b if d['scores'][s] is not None]
    out=dict(mode=a.mode,claim_label='approximate reproduction',result_class='complete-dataset cached diagnostic',
        dataset=a.dataset,arm=a.arm,target_f1=a.target_f1,source_sha256=sha(source),source_audit_sha256=sha(a.run_root/'supplemental_audit.json'),code_sha256=sha(__file__),
        metric='Unicode casefold+isalnum lexical multiset micro F1; inherited native256 ignore policy; no IoU')
    for name,population in [('unrestricted',None),('router_eligible',eligible)]:
        value=target_minimum(b,a.target_f1,population)
        if value['witness'] is not None:value['witness']=json_point(value['witness'])
        out[name]=dict(feasibility=value,global_oracle=json_point(global_oracle(b,population)))
    points=ranking_frontier(b,d['scores'],d['tie_keys']);feasible=[v for v in points if float(v['f1'])>=float(a.target_f1)]
    out['trained_rank']=dict(minimum_deepsolo_images=min((v['deepsolo_images'] for v in feasible),default=None),
        best=json_point(max(points,key=lambda v:(v['f1'],-v['deepsolo_images']))),frontier=[json_point(v) for v in points])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({name:dict(feasible=out[name]['feasibility']['feasible'],minimum_deepsolo_images=out[name]['feasibility']['minimum_deepsolo_images'],global_maximum=out[name]['global_oracle']['f1'],deepsolo_images=out[name]['global_oracle']['deepsolo_images'],total_images=out[name]['global_oracle']['total_images'],deepsolo_percentage=out[name]['global_oracle']['deepsolo_percentage']) for name in ['unrestricted','router_eligible']}))


if __name__=='__main__':main()
