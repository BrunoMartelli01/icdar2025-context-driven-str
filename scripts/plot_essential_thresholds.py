"""Standalone scientific F1/fallback frontiers; no model inference or selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

read=lambda p:json.loads(Path(p).read_text())

def count(p):return f"{p['deepsolo_images']}/{p['total_images']} ({p['deepsolo_percent']:.2f}%)"

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    original=read(args.root/'threshold_analysis_v1/results.json');oc=read(args.root/'threshold_analysis_v1/test_curves.json')
    matched=read(args.root/'threshold_analysis_yolo_matched_v1/results.json');mc=read(args.root/'threshold_analysis_yolo_matched_v1/test_curves.json')
    normalized=read(args.root/'threshold_analysis_normalized_v2/test_curves.json')
    names={'total_text':'Total-Text','icdar13':'ICDAR2013','icdar2015':'ICDAR2015'}
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    for row,expert in enumerate(['unet','yolo']):
        data=original if expert=='unet' else matched;curves=oc if expert=='unet' else mc
        for column,ds in enumerate(names):
            ax=axes[row,column];value=data['datasets'][ds][expert];curve=curves[ds][expert];n=curve[0]['total_images']
            ax.plot([p['deepsolo_percent'] for p in curve],[100*p['f1'] for p in curve],color='#155e75',lw=1.7,label='Attainable absolute-score cuts')
            comparison=normalized[ds][expert] if expert=='unet' else oc[ds][expert]
            ax.plot([p['deepsolo_percent'] for p in comparison],[100*p['f1'] for p in comparison],color='#9ca3af',ls=':',lw=1.2,
                label='Normalized-score control' if expert=='unet' else 'U-Net-head transfer control')
            baseline=value['all_deepsolo'];ax.axhline(100*baseline['f1'],color='#555',ls='--',lw=1,label='All DeepSolo: '+count(baseline))
            for point,marker,color,label in [(value['exact_half'],'o','#0f766e','Half-quota control'),
                (value['validation_selected']['empirical_loss_0.0']['test'],'D','#b45309','Validation no-loss cut'),
                (value['test_descriptive_best'],'*','#be123c','Best TEST cut: post hoc')]:
                ax.scatter(point['deepsolo_percent'],100*point['f1'],marker=marker,color=color,s=65 if marker!='*' else 130,zorder=5,
                    label=label+'\n'+count(point))
            ax.set_title(names[ds]+' / '+('corrected U-Net head' if expert=='unet' else 'YOLO-specific head'))
            ax.set_xlim(-2,102);ax.set_xlabel('Images sent to DeepSolo (percent)');ax.set_ylabel('Symbol-free lexical micro F1 (percent)')
            ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right')
            top=ax.secondary_xaxis('top',functions=(lambda x,n=n:x*n/100,lambda x,n=n:x*100/n));top.set_xlabel('DeepSolo image count / '+str(n),fontsize=8)
    fig.suptitle('Essential router: accuracy and fallback cost\nApproximate reproduction; cached logical routing; TEST optima are descriptive',fontsize=15)
    args.output.mkdir(exist_ok=False)
    fig.savefig(args.output/'essential_router_frontiers.png',dpi=180);fig.savefig(args.output/'essential_router_frontiers.svg');plt.close(fig)

if __name__=='__main__':main()
