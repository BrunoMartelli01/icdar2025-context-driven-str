"""Dependency-free SVG export and offline interactive threshold explorer."""
from __future__ import annotations

from html import escape
import json
import math


def chart(points, reference, split):
    width,height,left,right,top,ph,gap=1320,1240,100,1260,100,230,145
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a;font-size:17px}.title{font-size:24px;font-weight:bold}.axis{font-size:15px}</style>',
        f'<text x="100" y="35" class="title">{escape(split)}: CLIP threshold, words, and performance</text>',
        '<text x="100" y="65">Approximate reproduction · care words · TN=0 · DeepSolo fallback at score ≥ 0.4</text>']
    final=[p['combined_model']['detected_words'] for p in points if p['combined_model']]
    word_top=max(500,math.ceil(max(final+[p['words_retained_parseq'] for p in points])/500)*500)
    panels=[('Images retained by PARSeq (%)',100,[('PARSeq','#1565c0',lambda p:100*p['coverage_images'])]),
        ('Word count',word_top,[('Final predictions','#1565c0',lambda p:p['combined_model']['detected_words'] if p['combined_model'] else None),
          ('Final correct (TP)','#218739',lambda p:p['combined_model']['tp'] if p['combined_model'] else None),
          ('Final incorrect (FP)','#c62828',lambda p:p['combined_model']['fp'] if p['combined_model'] else None),
          ('Retained by PARSeq','#6c757d',lambda p:p['words_retained_parseq'])]),
        ('Final performance (%)',100,[('Precision','#ef6c00',lambda p:100*p['combined_model']['accuracy_on_detected_words'] if p['combined_model'] else None),
          ('Accuracy TN=0','#7b1fa2',lambda p:100*p['combined_model']['accuracy_tn0'] if p['combined_model'] else None),
          ('Recall','#218739',lambda p:100*p['combined_model']['recall'] if p['combined_model'] else None)])]
    for n,(label,ymax,series) in enumerate(panels):
        y0=top+n*(ph+gap);x=lambda t:left+t*(right-left);y=lambda v:y0+ph-v/ymax*ph
        parts.append(f'<text x="{left}" y="{y0-13}" class="title">{label}</text>')
        for i in range(6):
            v=i*ymax/5;yy=y(v)
            parts.extend([f'<line x1="{left}" y1="{yy}" x2="{right}" y2="{yy}" stroke="#e0e0e0"/>',
                          f'<text x="{left-12}" y="{yy+5}" text-anchor="end" class="axis">{v:g}</text>'])
        for i in range(11):
            xx=x(i/10);parts.append(f'<text x="{xx}" y="{y0+ph+24}" text-anchor="middle" class="axis">{i/10:.1f}</text>')
        parts.append(f'<line x1="{x(reference)}" y1="{y0}" x2="{x(reference)}" y2="{y0+ph}" stroke="#333" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="680" y="{y0+ph+50}" text-anchor="middle">Image acceptance threshold</text>')
        for k,(name,color,fn) in enumerate(series):
            # Inclusive acceptance is left-continuous; draw evaluated score
            # breakpoints as a pre-step curve, including grid points.
            path=[];previous=None
            for p in points:
                value=fn(p)
                if value is None:
                    previous=None;continue
                xx,yy=x(p['threshold']),y(value)
                if previous is None:path.extend([f'M {xx:.3f} {yy:.3f}'])
                else:path.extend([f'L {previous:.3f} {yy:.3f}',f'L {xx:.3f} {yy:.3f}'])
                previous=xx
            parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            xx=left+k*290;yy=y0+ph+83
            parts.extend([f'<line x1="{xx}" y1="{yy-5}" x2="{xx+22}" y2="{yy-5}" stroke="{color}" stroke-width="3"/>',
                          f'<text x="{xx+29}" y="{yy}">{escape(name)}</text>'])
    parts.append(f'<text x="100" y="{height-12}" class="axis">Vertical line: previous threshold {reference:g}. Missing words are FN; no threshold selection on TEST.</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


def render(output,result,reference):
    (output/'threshold_curve.svg').write_text(chart(result['splits']['test']['full_curve'],reference,'TEST300'))
    # Embedded JSON enables local-file viewing with no server, fetch or network.
    data=json.dumps(result,allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLIP threshold curves</title><style>
body{font:16px system-ui,sans-serif;margin:30px auto;max-width:1300px;padding:0 20px;color:#17202a;background:#fafafa}
h1{font-size:26px}select,input{font:inherit}input{width:80%}.cards{display:flex;gap:15px;flex-wrap:wrap;margin:25px 0}
.card{background:white;border:1px solid #ddd;padding:16px;min-width:150px}.card b{display:block;font-size:24px}
table{border-collapse:collapse;background:white;width:100%;margin:25px 0}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}
#notice{color:#a13b08}.active{background:#e3f2fd}img{width:100%;height:auto}
</style><h1>CLIP router: threshold, predicted words, and performance</h1>
<p>Approximate reproduction. TRAIN: 125 previously analyzed images; TEST: all 300 official images. Frozen CDF/predictions; internal DeepSolo threshold = 0.4.</p>
<label>Split <select id="split"><option value="test">TEST300</option><option value="train">TRAIN125</option></select></label>
<p><label>Threshold <b id="threshold"></b><br><input id="slider" type="range" min="0" step="1"></label></p>
<p id="notice"></p><div class="cards" id="cards"></div>
<p>Precision = TP/(TP+FP). Accuracy TN=0 = TP/(TP+FP+FN). Words = care predictions, after evaluation-only ignore handling.</p>
<table><thead><tr><th>Threshold</th><th>PARSeq images</th><th>DeepSolo images</th><th>PARSeq words</th><th>Final words</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Accuracy TN=0</th></tr></thead><tbody id="rows"></tbody></table>
<p>Exploratory curves: these TEST results do not select a new threshold.</p>
<img src="threshold_curve.svg" alt="Complete TEST curves"><script id="data" type="application/json">DATA_JSON</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),select=document.getElementById('split'),slider=document.getElementById('slider');
const num=v=>v===null||v===undefined?'—':Number(v).toLocaleString('en-GB'),pct=v=>v===null||v===undefined?'—':(100*v).toFixed(2)+'%';
function card(label,value){return '<div class="card">'+label+'<b>'+value+'</b></div>'}
function draw(){let s=data.splits[select.value],p=s.full_curve[Number(slider.value)],c=p.combined_model;
 document.getElementById('threshold').textContent=p.threshold.toFixed(6);
 document.getElementById('notice').textContent=c?'':'Final performance unavailable: missing '+p.missing_deepsolo_sample_ids.length+' DeepSolo inferences at this point. No image is omitted.';
 document.getElementById('cards').innerHTML=card('PARSeq images',p.parseq_images+'/'+p.images)+card('PARSeq words',num(p.words_retained_parseq))+card('Final words',num(c?.detected_words))+card('Final precision',pct(c?.accuracy_on_detected_words))+card('Accuracy TN=0',pct(c?.accuracy_tn0));
 document.getElementById('rows').innerHTML=s.grid.map(q=>{let m=q.combined_model;return '<tr class="'+(Math.abs(q.threshold-p.threshold)<1e-12?'active':'')+'"><td>'+q.threshold.toFixed(6)+'</td><td>'+q.parseq_images+'</td><td>'+q.deepsolo_images+'</td><td>'+q.words_retained_parseq+'</td><td>'+num(m?.detected_words)+'</td><td>'+num(m?.tp)+'</td><td>'+num(m?.fp)+'</td><td>'+num(m?.fn)+'</td><td>'+pct(m?.accuracy_on_detected_words)+'</td><td>'+pct(m?.accuracy_tn0)+'</td></tr>'}).join('');}
function reset(){let points=data.splits[select.value].full_curve;slider.max=points.length-1;slider.value=points.findIndex(p=>Math.abs(p.threshold-REFERENCE_VALUE)<1e-12);draw()}
select.onchange=reset;slider.oninput=draw;reset();
</script></html>'''.replace('DATA_JSON',data).replace('REFERENCE_VALUE',repr(reference))
    (output/'explore.html').write_text(page)
