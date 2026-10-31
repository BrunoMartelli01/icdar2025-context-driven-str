"""Approximate target70 supervision, accumulated updates and frozen OCR replay."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from recognition.icdar2015_word_f1 import aggregate_counts, sample_counts
from recognition.deepsolo_word_f1 import normalize_word
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer, verify_parseq_runtime_source)
from scripts.evaluate_icdar2015_totaltext_transfer_word_f1 import crop_hash
from scripts.train_icdar2015_totaltext_transfer import write_json
from segmentation.data import file_sha256
from segmentation.icdar2015_improvements import binary_proposals
from segmentation.training import SegmentationTrainingConfig


def load_profile(path):
    config=json.loads(path.read_text())
    if (path.resolve().parent.name!="approximate" or config["reproduction_mode"]!="approximate"
            or config["claim_label"]!="approximate reproduction"):
        raise ValueError("Explicit approximate target70 profile required")
    if config["training"]["precision"]!="float32" or config["validation_ocr_device"]!="cpu":
        raise ValueError("Qualified FP32/CPU validation OCR protocol required")
    if config["word_validation_interval"]<1 or config["training"]["epochs"]<2:
        raise ValueError("Positive word validation cadence and at least two epochs required")
    return config


@dataclass(frozen=True)
class Target70TrainingConfig(SegmentationTrainingConfig):
    def __post_init__(self):
        if self.learning_rate<=0 or self.batch_size<1 or self.learning_rate_schedule!="cosine_per_epoch":
            raise ValueError("Explicit positive target70 LR/batch and cosine schedule required")
        fields=asdict(self)
        fields.update(learning_rate=1e-4,batch_size=8,learning_rate_schedule="none")
        SegmentationTrainingConfig(**fields)


def training_config(config,branch):
    t=config["training"]
    return Target70TrainingConfig(epochs=t["epochs"],seed=t["seed"],adam_betas=tuple(t["adam_betas"]),
        adam_epsilon=t["adam_epsilon"],learning_rate_schedule=t["learning_rate_schedule"],
        checkpoint_selection="validation_word_f1",deterministic_algorithms=True,optimizer=t["optimizer"],
        learning_rate=t["learning_rate"],weight_decay=t["weight_decay"],batch_size=branch["physical_batch_size"],
        loss="masked_BCEWithLogitsLoss",separation_weight=t["separation_weight"])


class TrainingSupervision(Dataset):
    def __init__(self,owner_dataset,*,negative_ignore):
        self.owner_dataset=owner_dataset;self.negative_ignore=negative_ignore

    def __len__(self):return len(self.owner_dataset)

    def __getitem__(self,index):
        item=self.owner_dataset[index]
        if not self.negative_ignore:
            item["ignored_negative"]=torch.zeros_like(item["validity"])
            return item
        base=self.owner_dataset.base;sample=base.manifest.samples[index]
        with Image.open(base.data_root/sample.image_path) as image:size=image.size
        # Replay the exact paired random geometry. This map excludes padding and
        # rotation fill, but includes original TRAIN ignored text as background.
        _,_,canvas=base.transform.transform_with_validity(Image.new("RGB",size),
            Image.new("L",size),Image.new("L",size,255),index)
        ignored=(canvas-item["validity"]).clamp_min(0)
        if torch.any((item["mask"]>0)&(ignored>0)):
            raise ValueError("Care/ignore supervision conflict")
        item["ignored_negative"]=ignored;item["validity"]=canvas
        return item


def supervised_loss(logits,batch,*,separation_weight,ignore_negative_weight):
    target,valid,separator,ignored=[batch[k].to(logits.device) for k in ("mask","validity","separation","ignored_negative")]
    if logits.shape!=target.shape:raise ValueError("Logit/target shape differs")
    if torch.any((target>0)&((valid==0)|(ignored>0))):raise ValueError("Invalid positive supervision")
    weights=valid*(1+(separation_weight-1)*separator)
    if ignore_negative_weight>0:weights=weights+(ignore_negative_weight-1)*ignored
    if torch.any(weights<0):raise ValueError("Negative supervision weight")
    return (F.binary_cross_entropy_with_logits(logits,target,reduction="none")*weights).sum()/weights.sum().clamp_min(1)


def set_learning_rates(optimizer,config,epoch):
    n=config["training"]["epochs"]
    ratio=config["training"]["minimum_learning_rate_ratio"]
    factor=ratio+(1-ratio)*.5*(1+math.cos(math.pi*(epoch-1)/(n-1)))
    for group in optimizer.param_groups:group["lr"]=group["initial_lr"]*factor
    return {g["group_name"]:g["lr"] for g in optimizer.param_groups}


def train_epoch(network,batches,optimizer,config,branch,*,device,max_steps=None):
    network.train();optimizer.zero_grad(set_to_none=True)
    accumulation=branch["gradient_accumulation"];window=0;window_samples=0
    samples=steps=microbatches=0;total=0.;ids=[]
    def update():
        nonlocal window,window_samples,steps
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:parameter.grad.div_(window_samples)
        if branch["gradient_clip_norm"]>0:
            torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],branch["gradient_clip_norm"],error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1;window=window_samples=0
    for batch in batches:
        if max_steps is not None and steps>=max_steps:break
        images=batch["image"].to(device);logits=network(images)
        loss=supervised_loss(logits,batch,separation_weight=config["training"]["separation_weight"],
            ignore_negative_weight=branch["ignore_negative_weight"])
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite target70 loss")
        count=len(images);(loss*count).backward();window+=1;window_samples+=count
        samples+=count;microbatches+=1;total+=float(loss.detach())*count;ids.extend(batch["sample_id"])
        if window==accumulation:update()
    if window:update()
    if not samples:raise ValueError("No training samples")
    return {"loss":total/samples,"samples":samples,"steps":steps,"microbatches":microbatches,
        "sample_order_sha256":sha256("".join(x+"\n" for x in ids).encode()).hexdigest()}


class ValidationOCR:
    """GT-blind crop recognition cache, shared only for identical RGB/decoder."""
    def __init__(self,args,config):
        if (file_sha256(args.parseq_checkpoint)!=PARSEQ_CHECKPOINT_SHA256
                or file_sha256(args.parseq_source_archive)!=PARSEQ_RUNTIME_ARCHIVE_SHA256):
            raise ValueError("Qualified PARSeq artifact mismatch")
        runtime=verify_parseq_runtime_source(args.parseq_source,source_type=PARSEQ_ARCHIVE_SOURCE,
            source_archive=args.parseq_source_archive)
        self.config=config;self.args=args;self.cache=args.output_dir/"validation_ocr_cache"
        self.protocol={"device":"cpu","precision":"float32","runtime":runtime,
            "checkpoint_sha256":PARSEQ_CHECKPOINT_SHA256,"batch_size":config["parseq_batch_size"],
            "adapter_sha256":file_sha256(Path(__file__).resolve().parents[1]/"recognition/parseq.py")}
        self.protocol_sha=sha256(json.dumps(self.protocol,sort_keys=True).encode()).hexdigest()
        path=args.output_dir/"validation_ocr_protocol.json"
        if path.exists() and json.loads(path.read_text())!=self.protocol:raise ValueError("OCR protocol changed")
        if not path.exists():write_json(path,self.protocol)
        self.recognizer=None

    def evaluate(self,details,refs,manifest,root,shape,mask_dir):
        images={};samples={s.sample_id:s for s in manifest.samples};boxes={};pending={}
        for row in details["per_image"]:
            key=row["sample_id"];path=root/samples[key].image_path
            if file_sha256(path)!=samples[key].image_sha256:raise ValueError("Validation RGB changed")
            with Image.open(path) as im:images[key]=np.array(im.convert("RGB"))
            path=mask_dir/row["mask_file"]
            if file_sha256(path)!=row["mask_sha256"]:raise ValueError("Validation mask changed")
            with np.load(path) as maps:prediction=maps["prediction"]
            proposals,_=binary_proposals(prediction,refs[key]["image_wh"],shape)
            if len(proposals)!=row["proposals"]:raise ValueError("Proposal count differs")
            for p in proposals:
                x1,y1,x2,y2=p["bbox"];rgb=images[key][y1:y2,x1:x2];digest=crop_hash(rgb)
                p["crop_rgb_sha256"]=digest;cache=self.cache/(digest+".json")
                if cache.exists():
                    old=json.loads(cache.read_text())
                    if old["ocr_protocol_sha256"]!=self.protocol_sha or old["crop_rgb_sha256"]!=digest:
                        raise ValueError("OCR cache identity mismatch")
                else:pending.setdefault(digest,np.ascontiguousarray(rgb))
            boxes[key]=proposals
        items=list(pending.items());size=self.config["parseq_batch_size"]
        if items and self.recognizer is None:
            self.recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=self.args.parseq_source,
                checkpoint=self.args.parseq_checkpoint,device="cpu",runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                runtime_source_archive=self.args.parseq_source_archive))
        for start in range(0,len(items),size):
            chunk=items[start:start+size];outputs=self.recognizer.recognize_candidate_batch([rgb for _,rgb in chunk])
            if len(outputs)!=len(chunk):raise RuntimeError("Incomplete validation OCR")
            for (key,_),output in zip(chunk,outputs):
                write_json(self.cache/(key+".json"),{"ocr_protocol_sha256":self.protocol_sha,"crop_rgb_sha256":key,**output.to_dict()})
        rows=[]
        for row in details["per_image"]:
            key=row["sample_id"];predictions=[]
            for p in boxes[key]:
                path=self.cache/(p["crop_rgb_sha256"]+".json");o=json.loads(path.read_text())
                predictions.append({**p,"text":o["raw_text"],"sequence_confidence":o["sequence_confidence_native"],
                    "mean_token_log_probability":o["mean_token_log_probability"],"ocr_sha256":file_sha256(path)})
            words=refs[key]["ground_truth"]
            rows.append({"sample_id":key,"image_sha256":samples[key].image_sha256,
                "annotation_sha256":refs[key]["annotation_sha256"],"predictions":predictions,"ground_truth":words,
                "counts":sample_counts(predictions,words,iou_threshold=self.config["matching_iou_threshold"])})
        aggregate=aggregate_counts([r["counts"] for r in rows])
        if aggregate["geometry"]["f1"]!=details["aggregate"]["proposal_f1_iou_50"]:
            raise ValueError("Word/geometry replay disagrees")
        calibrated=calibrate_emission(rows,self.config)
        return {"reproduction_mode":"approximate","claim_label":"approximate reproduction",
            "result_class":"TRAIN validation correct-word F1","aggregate":aggregate,"per_image":rows,
            "emitted_aggregate":calibrated["aggregate"],"emission_policy":calibrated["policy"],
            "emission_calibration":calibrated["candidates"],
            "ocr_protocol":self.protocol,"failed_samples":0}


def emitted_predictions(predictions,policy):
    kept=[]
    for p in predictions:
        # Empty decodes have no character log probability. Confidence zero
        # keeps them in the raw threshold-zero population, and rejects them
        # only when a positive output threshold was selected on validation.
        mean_log=p["mean_token_log_probability"]
        confidence=(p["sequence_confidence"] if policy["confidence"]=="sequence_confidence"
            else (math.exp(mean_log) if mean_log is not None else 0.))
        if confidence>=policy["threshold"] and len(normalize_word(p["text"]))>=policy["minimum_alphanumeric_length"]:
            kept.append(p)
    return kept


def calibrate_emission(rows,config):
    grid=config["emission_policy_grid"];candidates=[]
    for field in ("sequence_confidence","mean_token_confidence"):
        for threshold in grid[field]:
            for minimum in grid["minimum_alphanumeric_length"]:
                policy={"confidence":field,"threshold":threshold,"minimum_alphanumeric_length":minimum}
                counts=[sample_counts(emitted_predictions(row["predictions"],policy),row["ground_truth"],
                    iou_threshold=config["matching_iou_threshold"]) for row in rows]
                candidates.append({"policy":policy,"aggregate":aggregate_counts(counts)})
    best=sorted(candidates,key=lambda x:(-x["aggregate"]["alphanumeric_casefold"]["f1"],
        -x["aggregate"]["alphanumeric_casefold"]["recall"],x["policy"]["minimum_alphanumeric_length"],
        x["policy"]["threshold"],x["policy"]["confidence"]))[0]
    return {**best,"candidates":candidates}


def word_selection_key(raw,emitted,geometry,epoch):
    return (emitted["alphanumeric_casefold"]["f1"],raw["alphanumeric_casefold"]["f1"],
        geometry["proposal_f1_iou_50"],-epoch)
