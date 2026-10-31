"""Reuse valid historical controls only after exact input/function equivalence checks."""
import argparse
import ast
import json
from pathlib import Path
import shutil

from scripts.run_image_context_correction import PHASES,REPO,pairs_for
from scripts.run_text_reconstruction_tests import read,sha,write


def function_ast(path,name):
    return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def reuse(source,target,source_code):
    source,target,source_code=map(Path,(source,target,source_code))
    a,b=read(source/'protocol.json'),read(target/'protocol.json');c=read(target/'resolved_config.json')
    if a['config_sha256']!=b['config_sha256'] or a['data_sha256']!=b['data_sha256'] or a['runtime']!=b['runtime']:
        raise ValueError('Generator reuse requires identical resolved configuration, data and runtime')
    runner='scripts/run_image_context_correction.py'
    for name,expected in a['code_sha256'].items():
        if sha(source_code/name)!=expected:raise ValueError('Original execution source changed')
        if name!=runner and b['code_sha256'][name]!=expected:raise ValueError('Generator dependency changed')
    for name in ('generators','generate_neural'):
        if function_ast(source_code/runner,name)!=function_ast(REPO/runner,name):raise ValueError('Training/inference function changed')
    data=read(target/'data.json');folds=b['folds'];external=read(c['paths']['spell_vocabulary'])['labels'];stages=[]
    for fold in [*range(c['folds']),'final']:
        work=source/'generators'/f'historical_{fold}';old=read(work/'pairs.json');receipt=read(work/'receipt.json')
        train=[s for s in data['fit'] if fold=='final' or folds[s['sample_id']]!=fold]
        seed=c['seed']+(fold if isinstance(fold,int) else c['folds'])
        rows,details=pairs_for(train,c,'historical',seed,external,False)
        if rows!=old['records'] or details['training_ids']!=old['training_ids']:raise ValueError('Historical training inputs differ')
        if len(rows)!=c['generator']['training_pairs'] or receipt['updates']*c['generator']['batch_size']!=len(rows):raise ValueError('Historical update budget is not exact')
        if sha(work/'checkpoint.pt')!=receipt['checkpoint_sha256'] or sha(work/'predictions.json')!=receipt['predictions_sha256']:raise ValueError('Historical artifact changed')
        dest=target/'generators'/work.name
        if dest.exists():raise ValueError('Refuse to overwrite a reused stage')
        shutil.copytree(work,dest)
        checks={p.name:sha(p) for p in work.iterdir() if p.is_file()}
        if any(sha(dest/name)!=expected for name,expected in checks.items()):raise ValueError('Historical copy mismatch')
        stages.append(dict(stage=work.name,source=str(work),files_sha256=checks,records=len(rows),regenerated_records_exact=True))
    result=dict(status='PASS',source_protocol_sha256=sha(source/'protocol.json'),target_protocol_sha256=sha(target/'protocol.json'),
        scope='Historical controls only; affected random/empirical recipes must be trained anew.',unchanged_training_and_inference_ast=True,stages=stages)
    write(target/'generator_reuse.json',result);print(json.dumps(dict(status='PASS',reused_stages=len(stages))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True,type=Path);parser.add_argument('--target',required=True,type=Path);parser.add_argument('--source-code',required=True,type=Path)
    args=parser.parse_args();reuse(args.source,args.target,args.source_code)


if __name__=='__main__':main()
