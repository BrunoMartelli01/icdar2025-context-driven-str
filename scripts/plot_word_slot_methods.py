"""Standalone scientific PNG/PDF comparisons; no model imports or inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from recognition.word_slot_text import normalize_word_text


def read(path):return json.loads(Path(path).read_text())


def iou(a,b):
    if a is None:return 0.
    a=list(map(int,a));b=list(map(int,b))
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/max(union,1e-12)


def figures(root,output):
    results=read(root/'word_results.json');routes=read(root/'router_results.json');selected=read(root/'analysis_selection.json')
    normalization=read(root/'protocol.json')['config']['evaluation'].get('text_normalization','casefold')
    text_label='alphanumeric casefold' if normalization=='alphanumeric_casefold' else 'exact casefold with punctuation'
    primary=selected['primary_best_learned_router_oof'];arms=list(read(root/'threshold_selection.json'))
    labels=[a.replace('_',' ')+(' *' if a==primary else '') for a in arms]
    rows=np.arange(len(arms));deep=results['test']['deepsolo']['f1']*100
    fig,axes=plt.subplots(1,3,figsize=(18,10),sharey=True,gridspec_kw={'width_ratios':[1.3,1,1.2]})
    fig.suptitle('Word-slot U-Net + PARSeq and frozen DeepSolo',fontsize=18,y=.98)
    ax=axes[0]
    ax.barh(rows-.16,[results['test'][a]['f1']*100 for a in arms],height=.3,color='#4184b7',label='U-Net alone')
    ax.barh(rows+.16,[routes['test'][a]['word']['f1']*100 for a in arms],height=.3,color='#2c9874',label='Word router, both experts')
    for y,a in enumerate(arms):
        r=routes['test'][a]['word']
        ax.text(r['f1']*100-.5,y+.16,f"D {r['logical_deepsolo_calls']}/{r['images']} ({100*r['logical_deepsolo_calls']/r['images']:.1f}%)",ha='right',va='center',fontsize=7,color='white')
    ax.axvline(deep,color='#222',ls='--',lw=1.5,label=f'DeepSolo {deep:.2f}%')
    ax.set_xlabel('TEST word micro F1 (%)');ax.set_yticks(rows,labels);ax.invert_yaxis();ax.legend(loc='lower left',fontsize=9)
    ax.set_title('Standalone and routed outcome')
    ax=axes[1]
    ax.barh(rows,[results['test'][a]['unet_only'] for a in arms],height=.6,color='#2c9874')
    ax.set_xlabel('Correct U-Net words missed by DeepSolo');ax.set_title('Exclusive recoveries before routing')
    for y,a in enumerate(arms):ax.text(results['test'][a]['unet_only']+1,y,str(results['test'][a]['unet_only']),va='center',fontsize=8)
    ax=axes[2]
    ax.barh(rows-.16,[routes['test'][a]['image']['f1']*100 for a in arms],height=.3,color='#8470b3',label='Image router, variable fallback')
    ax.barh(rows+.16,[routes['test'][a]['image_fixed_half']['f1']*100 for a in arms],height=.3,color='#d39c48',label='Image router, 150 DeepSolo calls')
    for y,a in enumerate(arms):
        for offset,kind in ((-.16,'image'),(.16,'image_fixed_half')):
            r=routes['test'][a][kind]
            ax.text(r['f1']*100-.5,y+offset,f"D {r['logical_deepsolo_calls']}/{r['images']} ({100*r['logical_deepsolo_calls']/r['images']:.1f}%)",ha='right',va='center',fontsize=7,color='white')
    ax.axvline(deep,color='#222',ls='--',lw=1.5);ax.set_xlabel('TEST word micro F1 (%)');ax.set_title('Selective image fallback')
    ax.legend(loc='lower left',fontsize=9)
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.018,'Approximate research screen; one seed; calibration-only fitting; explored TEST300 / 2204 care words.\n'
             f'* Learned student selected by OOF calibration routed F1. All predictions count; bbox IoU >= 0.5 + {text_label}; cap 20 OCR crops.\n'
             'Word routing calls both experts. Image-call counts are cached simulations, not measured deployment latency.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.95));output.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(output/('comparison.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    details=read(root/'failure_analysis.json')['test'][primary];examples=details['examples']
    review=read(root/'annotation_visual_review.json').get('cases',{}) if (root/'annotation_visual_review.json').exists() else {}
    examples=sorted(examples,key=lambda r:-(r['bbox'][2]-r['bbox'][0])*(r['bbox'][3]-r['bbox'][1]))
    chosen=[];seen=set()
    for category in ('alphabet','missing','reading','any'):
        for case in examples:
            key=(case['sample_id'],case['gt_index'])
            if key in seen:continue
            allowed=(category=='any' or category=='alphabet' and case['outside_deepsolo_alphabet'] or
                     category=='missing' and case['deep_failure_category']=='no_retained_bbox_match' or
                     category=='reading' and case['deep_failure_category']=='overlapping_output_not_exact_or_assignment')
            if allowed:chosen.append(case);seen.add(key);break
    if not chosen:return
    fig,axes=plt.subplots(2,2,figsize=(14,10));thresholds=read(root/'threshold_selection.json')[primary]['chosen']
    for ax,case in zip(axes.flat,chosen):
        with Image.open(case['image_path']) as image:rgb=np.asarray(image.convert('RGB'))
        box=case['bbox'];w,h=box[2]-box[0],box[3]-box[1]
        x1=max(0,int(box[0]-w*.35));y1=max(0,int(box[1]-h*.7));x2=min(rgb.shape[1],int(box[2]+w*.35));y2=min(rgb.shape[0],int(box[3]+h*.7))
        ax.imshow(rgb[y1:y2,x1:x2]);ax.add_patch(Rectangle((box[0]-x1,box[1]-y1),w,h,fill=False,edgecolor='#2dd272',lw=2))
        raw=read(root/'predictions'/primary/'test'/(case['sample_id'].replace(':','__')+'.json'))['predictions']
        candidates=[p for p in raw if normalize_word_text(p['text'],normalization) and normalize_word_text(p['text'],normalization)==normalize_word_text(case['text'],normalization) and iou(p['bbox'],box)>=.5 and p['presence']>=thresholds['presence'] and p['confidence']>=thresholds['confidence']]
        if candidates:
            p=max(candidates,key=lambda p:iou(p['bbox'],box));b=p['bbox'];ax.add_patch(Rectangle((b[0]-x1,b[1]-y1),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#3d9fff',lw=1.5))
        status='no bbox match' if case['deep_failure_category']=='no_retained_bbox_match' else repr(case['deep_nearest_text'])
        title=f"{case['sample_id']} | GT/U-Net: {case['text']}\nDeepSolo: {status}"
        if case['sample_id'] in review:title+='\nPossible GT discrepancy: image appears '+review[case['sample_id']]['visual_reading_candidate']
        ax.set_title(title,fontsize=10);ax.axis('off')
    for ax in list(axes.flat)[len(chosen):]:ax.axis('off')
    fig.suptitle('GT-matched U-Net hypotheses ('+text_label+'): annotation (green), prediction (blue)',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.subplots_adjust(hspace=.4)
    for suffix in ('png','pdf'):fig.savefig(output/('exclusive_examples.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();figures(args.run.resolve(),args.output.resolve())
