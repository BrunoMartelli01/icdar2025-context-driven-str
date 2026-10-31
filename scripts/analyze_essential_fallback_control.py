"""Exploratory ordinary-DeepSolo control; no expert or router threshold search."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from recognition.deepsolo_word_f1 import apply_policy
from recognition.unet_corrected_token_interfaces import lexical_counts,summarize
from scripts.run_essential_icdar_transfer import truth_and_validity
from scripts.analyze_essential_router_thresholds import heavy_counts

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();c=read(args.config);root=Path(c['paths']['icdar_transfer']);plan=read(root/'resolved_configuration.json')
    source=read(plan['config']['paths']['deepsolo_profile'])['selected_policy'];ordinary=dict(source,score_threshold=.4,recognition_threshold=0.)
    result={};inputs={}
    for ds,samples in plan['population']['test'].items():
        values=[]
        for s in samples:
            gt,valid,_=truth_and_validity(s,ds,c['paths']['icdar2015_root']);path=root/'fallback'/(s['sample_id'].replace(':','__')+'.json')
            inputs[str(path)]=sha(path);values.append(heavy_counts(read(path),s,gt,valid,ordinary))
        result[ds]=dict(**summarize(values),deepsolo_images=len(samples),total_images=len(samples),deepsolo_percent=100.)
    fusion=Path(c['paths']['totaltext_native'])
    # Its plan pins the original processed native-export source used by the qualified ordinary endpoint.
    native_plan=read(fusion/'resolved_configuration.json');source_root=Path(native_plan['paths']['improved_root']);path=source_root/'processed_predictions.json'
    rows=read(path)['splits']['test'];inputs[str(path)]=sha(path);values=[]
    for r in rows:
        predictions=apply_policy(r['native'],ordinary)
        values.append(lexical_counts([dict(raw_text=p['text'] if p.get('invalid_envelope') is None else '') for p in predictions],r['ground_truth'],[p['ignored'] for p in predictions]))
    result['total_text']=dict(**summarize(values),deepsolo_images=len(rows),total_images=len(rows),deepsolo_percent=100.)
    report=dict(status='SUCCESS',mode='approximate',claim_label='approximate reproduction',ordinary_policy=ordinary,datasets=result,input_sha256=inputs,
        timing='Exploratory control added after seeing the cross-dataset filtered-profile endpoints. Uses the previously qualified ordinary0.4/no-character-pruning recipe, without fitting to the new TEST.',
        interpretation='Same native checkpoint and RGB. This control checks whether a gain over the router fallback also exceeds ordinary DeepSolo; it does not replace or retune the frozen primary fallback.',analysis_code_sha256=sha(__file__))
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:dict(f1=v['f1'],D=v['deepsolo_images'],N=v['total_images'],percent=v['deepsolo_percent']) for k,v in result.items()}))

if __name__=='__main__':main()
