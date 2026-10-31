"""Approximate GT-free context interventions, repeated glyphs and structural checks."""
from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, sobel

from recognition.grounded_context_router import line_members, local_views, intersection, box_area


def summarize(values, prefix):
    x=np.asarray(values,float)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite inference evidence: '+prefix)
    return {prefix+'_mean':float(x.mean()) if len(x) else 0.,prefix+'_minimum':float(x.min()) if len(x) else 0.,
            prefix+'_missing':float(not len(x))}


def context_views(rgb, job, donor_rgb, settings):
    """Exactly preserve target pixels, location, canvas and framing across views."""
    base=local_views(rgb,job,settings['mask_value'])['local']
    x1,y1,_,_=job['context_box'];a,b,c,d=job.get('protected_box',job['target_box'])
    if not x1<=a<c<=job['context_box'][2] or not y1<=b<d<=job['context_box'][3]:
        raise ValueError('Protected text patch must stay inside the fixed context canvas')
    target=base[b-y1:d-y1,a-x1:c-x1].copy()
    donor=np.asarray(Image.fromarray(donor_rgb).resize((base.shape[1],base.shape[0]),Image.Resampling.BICUBIC)).copy()
    neutral=np.full_like(base,settings['mask_value'])
    foreign_without_target=donor.copy()
    for view in (donor,neutral): view[b-y1:d-y1,a-x1:c-x1]=target
    return {'original':base,'neutral':neutral,'foreign':donor,'foreign_without_target':foreign_without_target}


def expanded_job(job,image_wh,settings):
    out=dict(job);w,h=image_wh;a,b,c,d=job['target_box'];height=d-b
    x1,y1,x2,y2=job['context_box'];pad=settings['context_padding_heights']*height
    out['context_box']=[max(0,math.floor(x1-pad)),max(0,math.floor(y1-pad)),min(w,math.ceil(x2+pad)),min(h,math.ceil(y2+pad))]
    guard=settings['protected_padding_heights']*height
    out['protected_box']=[max(out['context_box'][0],math.floor(a-guard)),max(out['context_box'][1],math.floor(b-guard)),
        min(out['context_box'][2],math.ceil(c+guard)),min(out['context_box'][3],math.ceil(d+guard))]
    return out


def swap_features(probes):
    stability=[];margins=[];support=[];logit_range=[];foreign_evidence=[]
    for p in probes:
        ys=np.array([p['answers'][k][0]['yes_probability'] for k in ('original','neutral','foreign')])
        ms=np.array([p['answers'][k][0]['yes_probability']-p['answers'][k][1]['yes_probability'] for k in ('original','neutral','foreign')])
        stability.append(1-float(np.ptp(ys)));margins.append(float(ms.min()));support.append(float(ys.min()))
        odds=[p['answers'][k][0]['yes_logprob']-p['answers'][k][0]['no_logprob'] for k in ('original','neutral','foreign')]
        logit_range.append(float(np.ptp(odds)))
        foreign_evidence.append(p['answers']['foreign'][0]['yes_probability']-p['answers']['foreign_without_target'][0]['yes_probability'])
    return {**summarize(stability,'swap_stability'),**summarize(margins,'swap_margin'),
            **summarize(support,'swap_support'),**summarize(logit_range,'swap_logit_range'),
            **summarize(foreign_evidence,'swap_foreign_evidence')}


def permutation_features(words):
    values={k:[] for k in ('permutation_support','permutation_agreement','permutation_context_gain','permutation_variation')}
    for w in words:
        if w.get('abstention'): continue
        p=np.asarray(w['target_probabilities'],float);ids=np.asarray(w['argmax_ids'],int)
        if p.ndim!=2 or ids.shape!=p.shape: raise ValueError('Permutation token alignment differs')
        # EOS is saved but excluded from these character evidence summaries.
        n=len(w['text'])
        if not n: continue
        values['permutation_support'].append(float(p[:,:n].min(axis=0).mean()))
        values['permutation_agreement'].append(float((ids[:,:n]==ids[0:1,:n]).mean()))
        values['permutation_context_gain'].append(float((p[:-1,:n].mean(axis=0)-p[-1,:n]).mean()))
        values['permutation_variation'].append(float(p[:,:n].std(axis=0).mean()))
    return {key:value for k,v in values.items() for key,value in summarize(v,k).items()}


def glyph_patch(rgb, box, shape):
    x1,y1,x2,y2=box
    if not 0<=x1<x2<=rgb.shape[1] or not 0<=y1<y2<=rgb.shape[0]: raise ValueError('Invalid character proxy box')
    patch=np.asarray(Image.fromarray(rgb[y1:y2,x1:x2]).convert('L').resize(tuple(shape),Image.Resampling.BILINEAR),float)
    # Edge magnitude reduces brightness/polarity dependence; it is not a learned font invariant.
    edge=np.hypot(sobel(patch,axis=0),sobel(patch,axis=1)).ravel();norm=np.linalg.norm(edge)
    return (edge/norm).tolist() if norm>1e-8 else None


def glyph_features(words,rows,settings,geometry):
    row_by_id={r['candidate_id']:i for i,r in enumerate(rows)};items=[]
    for w in words:
        for c in w.get('characters',[]):
            if c['vector'] is not None and c['probability']>=settings['minimum_character_probability']:
                items.append({'candidate_id':w['candidate_id'],**c})
    items=sorted(items,key=lambda c:(-c['probability'],c['candidate_id'],c['position']))[:settings['maximum_characters']]
    pairs=[]
    for i,a in enumerate(items):
        ia=row_by_id[a['candidate_id']];neighbors=set(line_members(rows,ia,geometry))
        for b in items[i+1:]:
            ib=row_by_id[b['candidate_id']]
            if ia==ib or ib not in neighbors: continue
            overlap=intersection(rows[ia]['bbox'],rows[ib]['bbox'])/min(box_area(rows[ia]['bbox']),box_area(rows[ib]['bbox']))
            if overlap>settings['maximum_word_overlap']: continue
            sim=float(np.dot(a['vector'],b['vector']))
            if sim>=settings['minimum_visual_similarity']:
                pairs.append({'left':[a['candidate_id'],a['position']],'right':[b['candidate_id'],b['position']],
                    'similarity':sim,'disagree':a['character'].casefold()!=b['character'].casefold()})
    mass=sum(p['similarity'] for p in pairs)
    return {'glyph_contradiction':sum(p['similarity'] for p in pairs if p['disagree'])/mass if mass else 0.,
        'glyph_pair_density':len(pairs)/max(len(items),1),'glyph_log_pairs':math.log1p(len(pairs)),
        'glyph_missing':float(not pairs)}, {'characters_considered':len(items),'pairs':pairs}


def structure_features(rows,object_labels,geometry):
    text=' '.join(r['raw_text'] for r in rows);scene=' '.join(object_labels).casefold()
    menu=bool(re.search(r'\b(menu|prices?|pricelist)\b',scene+' '+text.casefold()))
    hours=bool(re.search(r'\b(open|closed|hours|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b',text.casefold()))
    groups=sorted({tuple(line_members(rows,i,geometry)) for i in range(len(rows))});records=[]
    prices=orphan_prices=times=orphan_times=invalid_times=0
    price_pattern=r'(?:[$€£]\s*\d+(?:[.,]\d+)?|\d+[.,]\d{2}\s*[$€£]?)'
    for group in groups:
        s=' '.join(rows[i]['raw_text'] for i in group)
        price=bool(re.search(price_pattern,s));clock=[list(pair) for pair in re.findall(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)',s)]
        letters=bool(re.search(r'[A-Za-z]{2,}',s))
        bad_time=any(int(h)>23 or int(m)>59 for h,m in clock)
        prices+=price;orphan_prices+=bool(menu and price and not letters)
        times+=len(clock);orphan_times+=bool(hours and len(clock)==1);invalid_times+=bool(hours and bad_time)
        records.append({'candidate_ids':[rows[i]['candidate_id'] for i in group],'text':s,'price':price,'times':clock,
            'orphan_price':bool(menu and price and not letters),'orphan_time':bool(hours and len(clock)==1),'invalid_time':bool(hours and bad_time)})
    return {'structure_menu_context':float(menu),'structure_hours_context':float(hours),
        'structure_log_prices':math.log1p(prices),'structure_log_times':math.log1p(times),
        'structure_orphan_price':orphan_prices/max(prices,1),'structure_orphan_time':orphan_times/max(len(groups),1),
        'structure_invalid_time':invalid_times/max(times,1),'structure_missing':float(not(menu or hours))},records


def edge_mask(image,settings):
    gray=np.asarray(image.convert('L').resize(tuple(settings['shape']),Image.Resampling.BILINEAR),float)
    edge=np.hypot(sobel(gray,axis=0),sobel(gray,axis=1))
    if float(edge.max())<settings['minimum_edge_magnitude']: return None
    return edge>=max(settings['minimum_edge_magnitude'],float(np.quantile(edge,settings['edge_quantile'])))


def chamfer(a,b):
    if a is None or b is None or not a.any() or not b.any(): return None
    # Symmetric nearest-edge distance, normalized by the canvas diagonal.
    return float((distance_transform_edt(~a)[b].mean()+distance_transform_edt(~b)[a].mean())/(2*np.hypot(*a.shape)))


class ContextRenderer:
    def __init__(self,fonts,settings):
        self.settings=settings;self.fonts=[ImageFont.truetype(p,size=settings['font_size']) for p in fonts]
        self.styles=[(i,a) for i in range(len(fonts)) for a in settings['angles']];self.cache={}

    def rendered(self,text,index):
        key=(text,index)
        if key not in self.cache:
            if not text.strip() or any(ord(c)<32 or ord(c)>126 for c in text): return None
            font_id,angle=self.styles[index];font=self.fonts[font_id];box=font.getbbox(text)
            if box[2]<=box[0] or box[3]<=box[1]: return None
            im=Image.new('L',(box[2]-box[0]+4,box[3]-box[1]+4),255)
            ImageDraw.Draw(im).text((2-box[0],2-box[1]),text,font=font,fill=0)
            if angle: im=im.rotate(angle,expand=True,fillcolor=255,resample=Image.Resampling.BICUBIC)
            self.cache[key]=edge_mask(im,self.settings)
        return self.cache[key]

    def infer(self,rgb,rows,geometry):
        real={r['candidate_id']:edge_mask(Image.fromarray(rgb[r['bbox'][1]:r['bbox'][3],r['bbox'][0]:r['bbox'][2]]),self.settings) for r in rows}
        margins=[];residuals=[];records=[]
        for i,row in enumerate(rows):
            peers=[j for j in line_members(rows,i,geometry) if j!=i and rows[j]['confidence']>=self.settings['minimum_neighbor_confidence']]
            peers=sorted(peers,key=lambda j:(-rows[j]['confidence'],rows[j]['candidate_id']))[:self.settings['maximum_neighbors']]
            if not peers: records.append({'candidate_id':row['candidate_id'],'abstention':'no_confident_same_line_neighbor'});continue
            scores=[]
            for font in range(len(self.styles)):
                distances=[chamfer(real[rows[j]['candidate_id']],self.rendered(rows[j]['raw_text'],font)) for j in peers]
                scores.append(float(np.mean(distances)) if all(x is not None for x in distances) else None)
            available=[j for j,x in enumerate(scores) if x is not None]
            if not available: records.append({'candidate_id':row['candidate_id'],'abstention':'unsupported_neighbor_rendering'});continue
            font=min(available,key=lambda j:(scores[j],j));text=row['raw_text']
            foil=text[::-1] if text.casefold()!=text[::-1].casefold() else text[:-1]+('X' if text[-1:].casefold()!='x' else 'Z')
            correct=chamfer(real[row['candidate_id']],self.rendered(text,font));wrong=chamfer(real[row['candidate_id']],self.rendered(foil,font))
            if correct is None or wrong is None: records.append({'candidate_id':row['candidate_id'],'abstention':'unsupported_target_rendering'});continue
            margins.append(wrong-correct);residuals.append(correct)
            records.append({'candidate_id':row['candidate_id'],'style_index':font,'font_index':self.styles[font][0],
                'angle':self.styles[font][1],'neighbor_ids':[rows[j]['candidate_id'] for j in peers],
                'neighbor_font_distances':scores,'target_distance':correct,'foil_distance':wrong,'foil':foil})
        return {**summarize(margins,'render_margin'),**summarize(residuals,'render_residual'),
            'render_checked_fraction':len(margins)/max(len(rows),1)},records


def feature_schemas(core,groups):
    schemas={'core':list(core)}
    for name,values in groups.items(): schemas[name]=list(core)+sorted(values)
    schemas['all']=list(core)+sorted(k for values in groups.values() for k in values)
    if len(schemas['all'])!=len(set(schemas['all'])): raise ValueError('Context feature names overlap')
    return schemas
