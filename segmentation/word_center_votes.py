"""APPROXIMATE endpoint-consensus/neighbor loss for the unchanged offset U-Net."""

from dataclasses import dataclass, field, fields
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from segmentation.word_center_offsets import WordCenterOffsetTrainingConfig, word_center_offset_loss


@dataclass(frozen=True)
class CenterVoteLossParams:
    beta: float
    regression_weight: float
    consensus_weight: float
    neighbor_weight: float
    separation_margin_fraction: float
    centroid_tolerance_pixels: float
    competition_chunk_pixels: int

    def __post_init__(self):
        for name in ("beta", "regression_weight", "centroid_tolerance_pixels"):
            if type(getattr(self,name)) not in (int,float) or not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("Positive finite vote-loss parameter required: " + name)
        for name in ("consensus_weight", "neighbor_weight"):
            if type(getattr(self,name)) not in (int,float) or not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("Nonnegative finite vote-loss weight required: " + name)
        if (type(self.separation_margin_fraction) not in (int,float) or not np.isfinite(self.separation_margin_fraction)
            or not 0 <= self.separation_margin_fraction < 1):
            raise ValueError("Separation margin must be a fraction in [0,1)")
        if type(self.competition_chunk_pixels) is not int or self.competition_chunk_pixels < 1:
            raise ValueError("Positive integer competition chunk size required")


@dataclass(frozen=True)
class CenterVoteTrainingConfig(WordCenterOffsetTrainingConfig):
    loss: str = "word_balanced_coherent_center_votes"
    regression_weight: float = field(kw_only=True)
    consensus_weight: float = field(kw_only=True)
    neighbor_weight: float = field(kw_only=True)
    separation_margin_fraction: float = field(kw_only=True)
    centroid_tolerance_pixels: float = field(kw_only=True)
    competition_chunk_pixels: int = field(kw_only=True)

    def __post_init__(self):
        if self.loss != "word_balanced_coherent_center_votes":
            raise ValueError("Unsupported coherent vote training loss")
        # Validate unchanged optimizer/protocol fields without admitting new
        # loss settings into the historical v1 configuration or source files.
        common = {item.name: getattr(self, item.name) for item in fields(WordCenterOffsetTrainingConfig)}
        common["loss"] = "word_balanced_center_offset_SmoothL1"
        WordCenterOffsetTrainingConfig(**common)
        self.vote_params()

    def vote_params(self):
        return CenterVoteLossParams(beta=self.offset_beta, **{
            item.name: getattr(self, item.name) for item in fields(CenterVoteLossParams) if item.name != "beta"})


def load_vote_config(path, *, mode, variant):
    path = Path(path).resolve()
    namespace = Path(__file__).resolve().parents[1] / "configs/approximate"
    if mode != "approximate" or namespace not in path.parents:
        raise ValueError("Coherent votes require explicit approximate mode and configs/approximate/; no fallback")
    cfg = json.loads(path.read_text())
    if (cfg["schema_version"] != 2 or cfg["reproduction_mode"] != mode
        or cfg["claim_label"] != "approximate reproduction"
        or cfg["architecture_identity"] != "experimental-word-center-offset-agunet-v1"
        or cfg["target"] != "visible_word_centroid_offsets_xy_divided_by_native_wh"
        or cfg["loss"]["name"] != "word_balanced_coherent_center_votes"
        or cfg["loss"]["aggregation"] != "mean_words_per_image_then_mean_supervised_images"
        or set(cfg["variants"]) != {"control", "coherent"} or variant not in cfg["variants"]):
        raise ValueError("Unsupported matched center-vote experiment/variant")
    cfg["variant"] = variant
    for weights in cfg["variants"].values():
        settings = {**cfg["loss"],**weights}
        CenterVoteLossParams(**{item.name:settings[item.name] for item in fields(CenterVoteLossParams)})
    cfg["loss"].update(cfg["variants"][variant])
    params_from_config(cfg)
    if cfg["variants"]["control"] != {"regression_weight": 1.0, "consensus_weight": 0.0, "neighbor_weight": 0.0}:
        raise ValueError("Control must preserve the historical regression objective")
    if (cfg["variants"]["coherent"]["regression_weight"] != 1.0
        or cfg["variants"]["coherent"]["consensus_weight"] <= 0
        or cfg["variants"]["coherent"]["neighbor_weight"] <= 0):
        raise ValueError("Coherent branch requires anchored regression and both declared vote terms")
    from localize.word_center_offsets import validate_vote_config
    validate_vote_config(cfg["voting"])
    return cfg


def params_from_config(cfg):
    return CenterVoteLossParams(**{item.name: cfg["loss"][item.name] for item in fields(CenterVoteLossParams)})


def _closest_rival(votes, centers, chunk_pixels):
    # Rival selection is a discrete argmin. Only the selected distance needs
    # a gradient; candidate tensors do not accumulate in the autograd graph.
    with torch.no_grad():
        indices = torch.cat([((chunk[:, None, :] - centers[None, :, :])**2).sum(-1).argmin(1)
                             for chunk in votes.split(chunk_pixels)])
    return (votes - centers[indices]).square().sum(1)


def center_vote_objective(raw_offsets, targets, owners, validity, *, params):
    """Regression + robust within-word consensus + adaptive rival margin.

    Votes use isotropic native coordinates divided by max(H,W). The existing
    regression keeps dx/W,dy/H. Means are equal words per supervised image,
    then equal supervised images. Background/ignore receives no gradient.
    Coincident GT centers have no discriminating xy target: omit those rival
    pairs and report ambiguous words, rather than inventing separate centers.
    """
    if not isinstance(params, CenterVoteLossParams):
        raise ValueError("Explicit CenterVoteLossParams required")
    if raw_offsets.ndim != 4 or min(raw_offsets.shape) < 1:
        raise ValueError("Nonempty Bx2xHxW vote predictions required")
    regression = word_center_offset_loss(raw_offsets, targets, owners, validity, beta=params.beta)
    prediction = raw_offsets.tanh()
    h, w = raw_offsets.shape[-2:]
    scale = max(h, w)
    yy, xx = torch.meshgrid(torch.arange(h, device=raw_offsets.device, dtype=raw_offsets.dtype),
                            torch.arange(w, device=raw_offsets.device, dtype=raw_offsets.dtype), indexing="ij")
    xy = torch.stack((xx, yy), dim=-1) / scale
    factor = raw_offsets.new_tensor([w, h]) / scale
    image_terms = []; surviving = competing = coincident = 0
    for index in range(len(raw_offsets)):
        valid = validity[index, 0].bool()
        ids = torch.unique(owners[index][valid & (owners[index] > 0)])
        if not len(ids):
            continue
        masks = [(owners[index] == owner) & valid for owner in ids]
        centers = torch.stack([xy[mask].mean(0) for mask in masks])
        word_terms = []
        for k, mask in enumerate(masks):
            truth_votes = xy[mask] + targets[index].detach().permute(1, 2, 0)[mask] * factor
            if torch.any((truth_votes - centers[k]).abs() * scale > params.centroid_tolerance_pixels):
                raise ValueError("Offsets do not point to their paired visible word centroid")
            votes = xy[mask] + prediction[index].permute(1, 2, 0)[mask] * factor
            consensus = F.smooth_l1_loss(votes, votes.mean(0).expand_as(votes), beta=params.beta, reduction="mean")
            gaps = torch.linalg.vector_norm(centers - centers[k], dim=1)
            other = torch.arange(len(centers), device=centers.device) != k
            ambiguous = other & (gaps * scale <= params.centroid_tolerance_pixels)
            coincident += int(ambiguous.any())
            eligible = other & ~ambiguous
            separation = votes.sum() * 0
            with torch.no_grad():
                native_votes = votes.detach() * scale
                endpoint = torch.linalg.vector_norm((prediction[index].detach()-targets[index].detach())[:, mask]
                                                     * raw_offsets.new_tensor([w,h])[:,None], dim=0).mean()
                scatter = torch.linalg.vector_norm(native_votes-native_votes.mean(0), dim=1).square().mean().sqrt()
                wrong = votes.new_zeros(())
            if eligible.any():
                rival = _closest_rival(votes, centers[eligible], params.competition_chunk_pixels)
                own = (votes - centers[k]).square().sum(1)
                gap = gaps[eligible].min()
                # Squared-distance contrast has a restoring gradient even past
                # the rival center. A difference of Euclidean distances has a
                # flat region there. Divide by gap to retain distance units.
                margin = params.separation_margin_fraction * gap.square()
                separation = (F.relu(own-rival+margin)/gap).mean()
                wrong = (rival.detach() < own.detach()).to(votes.dtype).mean()
                competing += 1
            word_terms.append(torch.stack((consensus,separation,endpoint,scatter,wrong)))
        image_terms.append(torch.stack(word_terms).mean(0))
        surviving += len(ids)
    if image_terms:
        consensus, separation, endpoint, scatter, wrong = torch.stack(image_terms).mean(0)
    else:
        zero = raw_offsets.sum() * 0
        consensus = separation = zero
        endpoint = scatter = wrong = zero.detach()
    total = params.regression_weight*regression + params.consensus_weight*consensus + params.neighbor_weight*separation
    return {"loss":total,"offset_loss":regression,"consensus_loss":consensus,"separation_loss":separation,
        "endpoint_error_native_pixels":endpoint,"endpoint_scatter_native_pixels":scatter,"wrong_neighbor_vote_rate":wrong,
        "supervised_images":len(image_terms),"surviving_words":surviving,"competing_words":competing,
        "coincident_centroid_words":coincident}


OBJECTIVE_METRICS = ("loss","offset_loss","consensus_loss","separation_loss",
                     "endpoint_error_native_pixels","endpoint_scatter_native_pixels","wrong_neighbor_vote_rate")


def train_center_vote_epoch(model, batches, optimizer, device, *, params):
    model.train()
    totals = {name:0. for name in OBJECTIVE_METRICS}
    active = samples = steps = empty = words = competing = coincident = 0
    ids = []
    for batch in batches:
        local = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        raw = model(local["image"])
        stats = center_vote_objective(raw,local["center_offsets"],local["owners_native"],local["validity"],params=params)
        if not all(torch.isfinite(stats[name]) for name in OBJECTIVE_METRICS):
            raise FloatingPointError("Nonfinite coherent center-vote objective/diagnostic")
        count = stats["supervised_images"]
        if count:
            stats["loss"].backward();optimizer.step();steps += 1
        else:
            empty += 1
        for name in totals:totals[name] += float(stats[name].detach()) * count
        active += count;samples += len(raw);words += stats["surviving_words"]
        competing += stats["competing_words"];coincident += stats["coincident_centroid_words"]
        ids.extend(batch["sample_id"])
    if not samples:raise ValueError("Empty center-vote training loader")
    return {"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        **{name:total/active if active else 0. for name,total in totals.items()},"samples":samples,
        "supervised_images":active,"surviving_words":words,"competing_words":competing,"coincident_centroid_words":coincident,
        "steps":steps,"empty_batches_without_update":empty,
        "sample_order_sha256":sha256("".join(x+"\n" for x in ids).encode()).hexdigest()}


def evaluate_center_votes(model, batches, device, *, params, export_dir=None):
    model.eval();rows = []
    with torch.inference_mode():
        for batch in batches:
            local = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            raw = model(local["image"])
            for index, sample_id in enumerate(batch["sample_id"]):
                stats = center_vote_objective(raw[index:index+1],local["center_offsets"][index:index+1],
                    local["owners_native"][index:index+1],local["validity"][index:index+1],params=params)
                if not all(torch.isfinite(stats[name]) for name in OBJECTIVE_METRICS):
                    raise FloatingPointError("Nonfinite center-vote validation metric")
                active = stats["supervised_images"] > 0
                row = {"sample_id":sample_id,"expected_words":int(batch["expected_word_count"][index]),
                    **{name:float(stats[name]) if active else None for name in OBJECTIVE_METRICS},
                    **{name:stats[name] for name in ("surviving_words","competing_words","coincident_centroid_words")}}
                if export_dir is not None:
                    directory = Path(export_dir)/sample_id.replace(":","__");directory.mkdir(parents=True,exist_ok=True)
                    arrays = {"normalized_offsets":raw[index].tanh().cpu().numpy(),
                        "target_offsets":local["center_offsets"][index].cpu().numpy(),
                        "owners":local["owners_native"][index].cpu().numpy(),"validity":local["validity"][index,0].bool().cpu().numpy()}
                    path = directory/"offsets.npz"
                    if path.exists():
                        with np.load(path,allow_pickle=False) as cached:
                            if set(cached.files) != set(arrays) or any(not np.array_equal(cached[k],v) for k,v in arrays.items()):
                                raise ValueError("Existing coherent offset export differs: "+sample_id)
                    else:np.savez_compressed(path,**arrays)
                    from scripts.prune_segmentation_checkpoints import sha256_file
                    row["export_sha256"] = sha256_file(path)
                rows.append(row)
    active = [row for row in rows if row["offset_loss"] is not None]
    if not rows or not active:raise ValueError("Validation needs samples and surviving word supervision")
    counts = {name:sum(row[name] for row in rows) for name in ("expected_words","surviving_words","competing_words","coincident_centroid_words")}
    return {"aggregate":{**{name:sum(row[name] for row in active)/len(active) for name in OBJECTIVE_METRICS},
        **counts,"samples":len(rows),"supervised_images":len(active),"empty_images":len(rows)-len(active),
        "lost_words":counts["expected_words"]-counts["surviving_words"]},"per_image":rows}
