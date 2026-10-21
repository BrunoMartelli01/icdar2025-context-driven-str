"""Lossless table export from completed approximate routing JSON; no inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(root):
    source=root/'results.json';data=json.loads(source.read_text())
    if data['status']!='SUCCESS' or data['reproduction_mode']!='approximate':
        raise ValueError('Only completed approximate results may be exported')
    rows=[]
    for split,groups in data['splits'].items():
        for family,points in groups['families'].items():
            for point_name,p in points.items():
                rows.append({'split':split,'family':family,'point':point_name,'threshold':p['threshold'],
                    'images':p['images'],'accepted_images':p['accepted_images'],'fallback_images':p['fallback_images'],
                    'good_images':p['accepted_good_images'],'bad_images':p['accepted_bad_images'],
                    'good_fraction':p['good_fraction_among_accepted'],'good_image_recall':p['good_image_recall'],
                    'predicted_words':p['combined']['detected_words'],'gt_words':p['combined']['gt_words'],
                    'tp':p['combined']['tp'],'fp':p['combined']['fp'],'fn':p['combined']['fn'],
                    'word_f1':p['combined']['f1'],'word_accuracy_tn0':p['combined']['accuracy_tn0'],
                    'accepted_predicted_words':p['accepted_words']['detected_words'],
                    'all_outputs':p['combined']['all_outputs'],'ignored_outputs':p['combined']['ignored_outputs'],
                    'unfilled_budget':p.get('unfilled_budget')})
    target=root/'analysis/results_table.csv';target.parent.mkdir(exist_ok=True)
    with target.open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    identity=sha(__file__);snapshot=target.parent/'code_snapshot'/(identity+'.py');snapshot.parent.mkdir(exist_ok=True)
    if not snapshot.exists():snapshot.write_bytes(Path(__file__).read_bytes())
    metadata={'reproduction_mode':'approximate','claim_label':'approximate reproduction','source_results_sha256':sha(source),
        'csv_sha256':sha(target),'export_code_sha256':identity,'rows':len(rows),'inference_performed':False,
        'note':'fractions0..1; good image means local per-image wordF1>=.8; train/test/selection/calibration kept separate'}
    target.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'path':str(target),**metadata}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',type=Path,required=True)
    export(parser.parse_args().run)
