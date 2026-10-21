"""Approximate segmentation-reproduction components.

Nothing exported by this package is evidence of an exact paper implementation.
"""

from .data import SegmentationDataset, SegmentationManifest, load_segmentation_manifest

__all__ = ["SegmentationDataset", "SegmentationManifest", "load_segmentation_manifest"]
