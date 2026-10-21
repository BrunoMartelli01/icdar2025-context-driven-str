"""Training-facing V3 wrapper: transport guard evidence through DataLoader workers."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from segmentation.data import file_sha256
from segmentation.guarded_word_gt import make_guarded_dataset


class AuditedGuardedWordDataset(Dataset):
    """Delegate targets unchanged; attach collatable provenance and diagnostics."""
    def __init__(self, dataset, recipe_path):
        self.source=dataset
        self.transform=dataset.transform
        self.manifest=dataset.manifest
        self.paths=dataset.paths
        self.recipe_sha256=file_sha256(Path(recipe_path))

    def __len__(self):
        return len(self.source)

    def __getitem__(self,index):
        item=self.source[index]
        audit=self.transform.last_audit
        item.update(target_variant='guarded-word-targets-v3',target_recipe_sha256=self.recipe_sha256,
            native_owners=torch.from_numpy(self.transform.last_owners.copy()),
            gt_guard_stats=dict(audit['stats']),
            gt_guard_audit_json=json.dumps(audit,sort_keys=True,separators=(',',':')))
        return item


def make_audited_guarded_dataset(config_path, *, augmentation_enabled, verify_checksums=True):
    dataset=make_guarded_dataset(config_path,augmentation_enabled=augmentation_enabled,
                                 verify_checksums=verify_checksums)
    return AuditedGuardedWordDataset(dataset,config_path)
