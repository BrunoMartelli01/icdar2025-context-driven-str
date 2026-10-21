"""Parse the pinned VimTS word-GT variant, distinct from CTW line GT."""

from recognition.scene_text_datasets import SceneTextInstance
import cv2
import numpy as np
import unicodedata
from PIL import Image, ImageDraw


def parse_vimts_word_gt(text: str) -> tuple[SceneTextInstance, ...]:
    instances = []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        if ",####" not in raw:
            raise ValueError(f"VimTS word GT line{number} lacks the coordinate/text delimiter")
        geometry, transcript = raw.split(",####", 1)
        numbers = [int(x) for x in geometry.split(",")]
        if len(numbers) < 6 or len(numbers) % 2:
            raise ValueError(f"VimTS word GT line{number} has invalid polygon coordinates")
        transcript = transcript.strip()
        if not transcript or (transcript != "###" and any(x.isspace() for x in transcript)):
            raise ValueError(f"VimTS line{number} is empty or contains multiple whitespace-separated words")
        points = tuple(zip(numbers[::2], numbers[1::2]))
        if transcript != "###" and len(set(points)) < 3:
            raise ValueError(f"Degenerate care word at line{number}")
        instances.append(SceneTextInstance(points, transcript, transcript == "###", "word"))
    if not instances:
        raise ValueError("VimTS word GT contains no instances")
    return tuple(instances)


def restore_split_single_token_lines(words: tuple[SceneTextInstance, ...],
                                    lines: tuple[SceneTextInstance, ...]):
    """Correct split-word candidates only when their text reconstructs a source token.

    ASSUMPTION: literal whitespace in the original transcription defines its
    token count. Punctuation/case normalization is used solely to check a
    proposed merge, never to invent word boundaries or alter original text.
    Geometry is the original annotated line (a single token), not a new guess.
    """
    def canonical(text):
        return "".join(c for c in text.casefold() if unicodedata.category(c)[0] in "LN")

    assigned = {}
    ambiguous = []
    for word_id, word in enumerate(words):
        if word.ignore:
            continue
        x0,y0,x1,y1 = word.bbox_xyxy
        center = ((x0+x1)/2, (y0+y1)/2)
        candidates = [i for i,line in enumerate(lines) if not line.ignore and
                      cv2.pointPolygonTest(np.asarray(line.polygon_xy,dtype=np.int32),center,False) >= 0]
        if len(candidates) == 1:
            assigned.setdefault(candidates[0],[]).append(word_id)
        elif len(candidates) > 1:
            ambiguous.append(word_id)
    replaced = set()
    replacements = {}
    changes = []
    for line_id, parts in assigned.items():
        source = lines[line_id]
        if len(source.transcription.split()) != 1 or len(parts) < 2:
            continue
        points=np.asarray(source.polygon_xy,dtype=float)
        half=len(points)//2
        centerline=(points[:half]+points[half:][::-1])/2
        segments=np.diff(centerline,axis=0)
        lengths=np.linalg.norm(segments,axis=1)
        arc=np.concatenate(([0.0],np.cumsum(lengths)))
        def position(word_id):
            x0,y0,x1,y1=words[word_id].bbox_xyxy
            center=np.array([(x0+x1)/2,(y0+y1)/2])
            denom=np.sum(segments*segments,axis=1)
            t=np.divide(np.sum((center-centerline[:-1])*segments,axis=1),denom,
                        out=np.zeros_like(denom),where=denom>0).clip(0,1)
            closest=centerline[:-1]+t[:,None]*segments
            idx=int(np.argmin(np.sum((closest-center)**2,axis=1)))
            return (float(arc[idx]+t[idx]*lengths[idx]),word_id)
        order=sorted(parts,key=position)
        combined="".join(words[i].transcription for i in order)
        if canonical(combined) != canonical(source.transcription) or not canonical(combined):
            continue
        replacements[min(parts)] = SceneTextInstance(source.polygon_xy,source.transcription,False,"word")
        replaced.update(parts)
        changes.append({"original_line_id":line_id+1,"original_transcription":source.transcription,
                        "vimts_word_ids":[i+1 for i in order],
                        "vimts_transcriptions":[words[i].transcription for i in order]})
    result=tuple(replacements[i] if i in replacements else word for i,word in enumerate(words)
                 if i not in replaced or i in replacements)
    return result,{"merged_original_single_token_lines":len(changes),"changes":changes,
                   "ambiguous_vimts_word_ids":[i+1 for i in ambiguous]}


def remove_redundant_word_annotations(words: tuple[SceneTextInstance, ...], size_wh: tuple[int,int]):
    """Record and remove exact duplicates and fully contained lexical fragments.

    A smaller care polygon may be removed only if every rasterized pixel is
    inside a larger care polygon and its normalized text is a strict substring
    of that larger word. Unrelated overlaps are preserved for explicit failure.
    No failed image is dropped and the source archive remains immutable.
    """
    def canonical(text):
        return "".join(c for c in text.casefold() if unicodedata.category(c)[0] in "LN")
    unique=[]; source_ids=[]; seen={}; duplicate_records=[]
    for i,word in enumerate(words,1):
        if word in seen:
            duplicate_records.append({"source_word_id":i,"kept_source_word_id":seen[word],
                                      "transcription":word.transcription})
            continue
        seen[word]=i;unique.append(word);source_ids.append(i)
    removed={};nested_records=[];equal_text_nested=[]
    for i,child in enumerate(unique):
        if i in removed or child.ignore or not canonical(child.transcription):continue
        cx0,cy0,cx1,cy1=child.bbox_xyxy
        for j,parent in enumerate(unique):
            if i==j or j in removed or parent.ignore or len(canonical(child.transcription))>len(canonical(parent.transcription)):
                continue
            if canonical(child.transcription) not in canonical(parent.transcription):continue
            px0,py0,px1,py1=parent.bbox_xyxy
            if not (px0<=cx0<=cx1<=px1 and py0<=cy0<=cy1<=py1):continue
            # Compare only a local raster; inclusive integer polygon edges.
            canvas_size=(cx1-cx0+1,cy1-cy0+1)
            child_mask=Image.new('1',canvas_size,0);parent_mask=Image.new('1',canvas_size,0)
            ImageDraw.Draw(child_mask).polygon([(x-cx0,y-cy0) for x,y in child.polygon_xy],fill=1)
            ImageDraw.Draw(parent_mask).polygon([(x-cx0,y-cy0) for x,y in parent.polygon_xy],fill=1)
            c=np.asarray(child_mask,dtype=bool);p=np.asarray(parent_mask,dtype=bool)
            if c.any() and np.all(~c|p):
                if canonical(child.transcription)==canonical(parent.transcription):
                    # Same word at the same location with nested alternative
                    # boundaries: retain the tighter source polygon. For
                    # raster-equivalent regions, keep the earlier source row.
                    parent_size=(px1-px0+1,py1-py0+1)
                    whole_parent=Image.new('1',parent_size,0)
                    ImageDraw.Draw(whole_parent).polygon([(x-px0,y-py0) for x,y in parent.polygon_xy],fill=1)
                    if np.count_nonzero(np.asarray(whole_parent))>np.count_nonzero(c) or source_ids[j]>source_ids[i]:
                        removed[j]=i
                        equal_text_nested.append({'source_word_id':source_ids[j],
                            'kept_source_word_id':source_ids[i],'transcription':parent.transcription})
                    continue
                removed[i]=j
                nested_records.append({"source_word_id":source_ids[i],"parent_source_word_id":source_ids[j],
                                       "transcription":child.transcription,"parent_transcription":parent.transcription})
                break
    return tuple(word for i,word in enumerate(unique) if i not in removed), {
        "exact_duplicates_removed":len(duplicate_records),"exact_duplicates":duplicate_records,
        "equal_text_nested_duplicates_removed":len(equal_text_nested),"equal_text_nested_duplicates":equal_text_nested,
        "contained_lexical_fragments_removed":len(nested_records),"contained_fragments":nested_records}
