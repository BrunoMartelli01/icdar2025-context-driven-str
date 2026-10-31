"""Plot saved common-pipeline results; no fitting or inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    rows=json.loads((root/'comparison.json').read_text());oracle=json.loads((root/'icdar13_gt_oracles.json').read_text())['results']
    fig,axes=plt.subplots(1,2,figsize=(15,6.5),layout='constrained')
    for ax,selected,title in [(axes[0],[r for r in rows if r['dataset']!='icdar13'],'Common procedure: accepted TT / IC15 targets'),
        (axes[1],[r for r in rows if r['dataset']=='icdar13'],'Same procedure: ICDAR2013 90% test')]:
        y=np.arange(len(selected));labels=[]
        ax.barh(y-.16,[r['baseline_f1_percent'] for r in selected],height=.29,color='#aab3bd',label='Previous qualified pipeline')
        ax.barh(y+.16,[r['f1_percent'] for r in selected],height=.29,color='#327b72',label='New common procedure')
        for index,row in enumerate(selected):
            version='YOLO + U-Net' if row['version']=='yolo_unet' else row['version'].upper()
            labels.append(f"{row['dataset']} / {version}\nDeepSolo {row['deepsolo_images']}/{row['total_images']} ({row['deepsolo_percentage']:.2f}%)")
            ax.text(row['f1_percent']+.2,index+.16,f"{row['f1_percent']:.2f}%",va='center',fontsize=9)
            ax.text(row['baseline_f1_percent']+.2,index-.16,f"{row['baseline_f1_percent']:.2f}%",va='center',fontsize=8,color='#555555')
        ax.set(yticks=y,yticklabels=labels,xlim=(65,100),xlabel='Punctuation-free lexical micro F1 (%)',title=title)
        ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.13),fontsize=8,ncol=2)
    axes[1].axvline(90,color='#913c2c',ls='--',label='Requested 90%')
    for i,r in enumerate([r for r in rows if r['dataset']=='icdar13']):
        ideal=oracle['icdar13__'+r['version']]['exact_budget']
        axes[1].plot(100*ideal['f1'],i+.16,marker='D',color='#c29031',ms=6,label='GT-perfect choice, same quota' if i==0 else None)
    axes[1].legend(loc='upper left',bbox_to_anchor=(0,-.13),fontsize=8,ncol=2)
    fig.suptitle('Approximate reproduction — full TEST datasets, fixed DeepSolo budgets\nLogical cached routing; GT-perfect choice is a privileged diagnostic; prior TEST exposure remains',fontsize=11)
    fig.savefig(root/'common_pipeline.png',dpi=160);fig.savefig(root/'common_pipeline.svg');plt.close(fig)
    print(root/'common_pipeline.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--report-root',type=Path,required=True);main(parser.parse_args().report_root)
