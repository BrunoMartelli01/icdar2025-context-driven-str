"""Shared, explicit APPROXIMATE word-U-Net and fusion experiment boundaries."""

from collections import Counter
import hashlib
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def word_counts(rows, ground_truth, ignored):
    if len(rows) != len(ignored):
        raise ValueError('Prediction/ignore cardinality differs')
    care = [r for r, skip in zip(rows, ignored) if not skip]
    a = Counter(normalize(r['raw_text']) for r in care if normalize(r['raw_text']))
    b = Counter(normalize(r['text']) for r in ground_truth if normalize(r['text']))
    tp = sum((a & b).values())
    return dict(tp=tp, fp=len(care)-tp, fn=len(ground_truth)-tp,
                detected_words=len(care), gt_words=len(ground_truth),
                ignored_outputs=sum(ignored), all_outputs=len(rows))


def aggregate_counts(rows):
    rows = list(rows)
    totals = {key: sum(row[key] for row in rows) for key in
              ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'ignored_outputs', 'all_outputs')}
    denominator = 2*totals['tp'] + totals['fp'] + totals['fn']
    return dict(totals, images=len(rows), f1=2*totals['tp']/denominator if denominator else 0.)


def validate_roles(records, partitions):
    """Disallow both ID and decoded-RGB overlap, including the complete TEST."""
    known, rgb = set(), set()
    for role in ('fit', 'selection', 'calibration', 'test'):
        ids = partitions[role]
        if len(ids) != len(set(ids)) or set(ids) & known:
            raise ValueError('Duplicate sample or overlapping roles: '+role)
        hashes = {records[s]['rgb_sha256'] for s in ids}
        if hashes & rgb:
            raise ValueError('RGB overlap between roles: '+role)
        known.update(ids)
        rgb.update(hashes)


def checked_mode(config_path, config):
    if Path(config_path).resolve().parent.name != 'approximate' or config['mode'] != 'approximate':
        raise ValueError('An explicit configs/approximate profile is required')
    if config['claim_label'] != 'approximate reproduction':
        raise ValueError('Approximate claim label is required')


def annotation(record):
    """Keep inherited native256 ignore/care, independent of prediction generation."""
    from PIL import Image
    from scripts.icdar_budget_data import annotation as icdar_annotation, native_mask
    if not record['sample_id'].startswith('total_text:'):
        return icdar_annotation(record)
    from recognition.scene_text_datasets import parse_totaltext_annotation
    from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
    if file_hash(record['annotation_path']) != record['annotation_sha256']:
        raise ValueError('Changed Total-Text annotation')
    instances = parse_totaltext_annotation(Path(record['annotation_path']))
    instances, _ = clip_instances_to_pixel_extent(instances, tuple(record['image_wh']))
    # Use the qualified native256 validity boundary. Repainting clipped polygons
    # would both change that boundary and fail on legitimate degenerate ignores.
    validity_path = record.get('evaluation_validity_path', record['validity_path'])
    validity_sha = record.get('evaluation_validity_sha256', record['validity_sha256'])
    if file_hash(validity_path) != validity_sha:
        raise ValueError('Changed Total-Text evaluation validity')
    with Image.open(validity_path) as source:
        validity = source.convert('L')
    truth = [dict(text=w.transcription, bbox=[w.bbox_xyxy[0], w.bbox_xyxy[1],
             w.bbox_xyxy[2]+1, w.bbox_xyxy[3]+1]) for w in instances if not w.ignore]
    return truth, native_mask(validity, 256)
