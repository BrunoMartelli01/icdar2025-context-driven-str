"""Render saved numerical results only, in an existing plotting environment."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root,report_name='report'):
    if Path(report_name).name!=report_name or report_name in ('.','..'):
        raise ValueError('Report name must be a single directory name')
    data=json.loads((root/report_name/'summary.json').read_text());rows={r['name']:r for r in data['results']}
    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
    light=['light_none','light_generic_keep','light_generic_conservative','mixed_totaltext_light','mixed_fixed_epoch_light']
    labels=['Original','Generic keep','Generic conservative','Mixed IC13 + TT (selected)','Mixed IC13 + TT (epoch 30)']
    for i,name in enumerate(light):
        r=rows[name];axes[0].barh(i,r['f1_percent'],color='#396eaa')
        axes[0].text(r['f1_percent']+.15,i,f"{r['f1_percent']:.2f}%",va='center',fontsize=10)
        labels[i]+=f"\nDeepSolo {r['deepsolo_images']}/{r['total_images']} ({r['deepsolo_percentage']:.2f}%)"
    axes[0].set(yticks=range(len(light)),yticklabels=labels,xlim=(0,100),xlabel='Lexical micro F1 (%)',title='Light expert controls, full TEST (233 images)')
    axes[0].invert_yaxis()
    names=['older_frozen_gate','seen_control','crossfit_control','crossfit_raw_no_guard']
    labels=['Older frozen gate','Seen FIT observations','Out-of-fold observations','OOF + raw evidence / no guard']
    for i,name in enumerate(names):
        r=rows[name];axes[1].barh(i,r['f1_percent'],color='#34836e')
        axes[1].text(r['f1_percent']+.15,i,f"{r['f1_percent']:.2f}%",va='center',fontsize=10)
        labels[i]+=f"\nDeepSolo {r['deepsolo_images']}/{r['total_images']} ({r['deepsolo_percentage']:.2f}%)"
    oracle=rows['current_gt_oracle']
    axes[1].axvline(oracle['f1_percent'],ls='--',color='#963627',label=f"GT oracle {oracle['f1_percent']:.2f}%\n61/233 (26.18%)")
    axes[1].set(yticks=range(len(names)),yticklabels=labels,xlim=(0,100),xlabel='Lexical micro F1 (%)',title='Trained gates, fixed experts')
    axes[1].invert_yaxis();axes[1].legend(loc='upper right',fontsize=9,frameon=False)
    crop=data['crop_diagnosis']['test']['0']
    groups=['correct_emitted_crop','no_viable_crop','viable_crop_wrong_reading','correct_crop_rejected_by_filter']
    labels=['Correct emitted viable crop','No crop at bbox IoU ≥ 0.5','Viable crop, wrong reading','Correct raw crop filtered']
    colors=['#34836e','#ac4a3d','#d08e35','#6475a4']
    axes[2].barh(range(4),[crop[k] for k in groups],color=colors)
    for i,k in enumerate(groups):
        axes[2].text(crop[k]+6,i,str(crop[k]),va='center')
    axes[2].set(yticks=range(4),yticklabels=labels,xlim=(0,850),xlabel='GT word identities',title='Spatial crop diagnosis, 1095 GT words\nDeepSolo 0/233 (0.00%)')
    axes[2].invert_yaxis()
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('ICDAR2013 hypothesis tests — approximate reproduction\nCached logical routing; GT oracle/crops are privileged diagnostics; spatial groups differ from lexical matching',fontsize=12)
    out=root/report_name/'hypothesis_tests.png';fig.savefig(out,dpi=160);plt.close(fig)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--report-name',default='report');a=p.parse_args();main(a.run_root,a.report_name)
